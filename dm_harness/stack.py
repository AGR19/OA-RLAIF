"""The servers of a dm_harness run, each a child process with its log in outputs/full_eval/logs/:

  decider   pplx-decider-v1.2 behind POST /v1/systemone (dm_harness.decider_server, .venv-decider), port 8090
  vllm      `vllm serve` with the All-merged weights, data-parallel over its GPUs, served as oa-rlaif-e4b, port 8100
  jah       the JevAgentHarness proxy (`jah serve`), port 8787, between the agent and vLLM
  upstream  DM_FAKE=lexical only: a lexical stand-in for vLLM (dm_harness.fake_upstream), port 8100

GPUs: DM_GPUS (arm A, every GPU for the decision model), DM_DECIDER_GPUS / DM_GEMMA_GPUS (arm B; default: the first
half for the decision model, the rest for vLLM).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from oa_rlaif.paths import ROOT, out_dir

from full_eval.schedule import root

DECIDER_PORT, VLLM_PORT, JAH_PORT = 8090, 8100, 8787
SERVED = "oa-rlaif-e4b"
DECIDER_REPO = "perplexity-ai/pplx-decider-v1.2"


def fake() -> bool:
    return os.environ.get("DM_FAKE") == "lexical"


def decider_dir() -> Path:
    return Path(os.environ.get("DM_DECIDER_DIR") or out_dir() / "dm_models" / "pplx-decider-v1.2")


def gpus() -> list[str]:
    if os.environ.get("DM_GPUS"):
        return [g for g in os.environ["DM_GPUS"].split(",") if g]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        n = sum(1 for line in out.splitlines() if line.startswith("GPU"))
    except (OSError, subprocess.SubprocessError):
        n = 0
    return [str(i) for i in range(max(1, n))]


def split_gpus() -> tuple[list[str], list[str]]:
    """Arm B: (decision-model GPUs, vLLM GPUs)."""
    all_gpus = gpus()
    half = max(1, len(all_gpus) // 2)
    dec = [g for g in os.environ.get("DM_DECIDER_GPUS", ",".join(all_gpus[:half])).split(",") if g]
    gem = [g for g in os.environ.get("DM_GEMMA_GPUS", ",".join(all_gpus[half:] or all_gpus)).split(",") if g]
    return dec, gem


def _start(name: str, args: list[str], env: dict | None = None) -> subprocess.Popen:
    (root() / "logs").mkdir(parents=True, exist_ok=True)
    log = open(root() / "logs" / f"{name}.log", "a")
    print(f"[stack] starting {name}: {' '.join(args[:4])} ...", flush=True)
    return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}, cwd=ROOT)


def wait_http(url: str, proc: subprocess.Popen, minutes: float, name: str) -> None:
    for _ in range(int(minutes * 12)):
        if proc.poll() is not None:
            raise SystemExit(f"{name} stopped during start-up; see outputs/full_eval/logs/{name}.log")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"[stack] {name} is up", flush=True)
                    return
        except OSError:
            pass
        time.sleep(5)
    proc.terminate()
    raise SystemExit(f"{name} did not start within {minutes:g} minutes; see outputs/full_eval/logs/{name}.log")


def start_decider(devices: list[str]) -> subprocess.Popen:
    """The decision model on these GPUs (one worker each); loading and compiling take a few minutes."""
    py = str(ROOT / ".venv-decider" / "bin" / "python") if not fake() else sys.executable
    args = [py, "-m", "dm_harness.decider_server", "--checkpoint", str(decider_dir()), "--devices", ",".join(devices),
            "--port", str(DECIDER_PORT)]
    proc = _start("decider", args, {"PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"})
    wait_http(f"http://127.0.0.1:{DECIDER_PORT}/health", proc, 40, "decider")
    smoke()
    return proc


def smoke() -> dict:
    """The model card's own example through the server (the wire format JevAgentHarness sends, with its extra
    'model' key): it must come back with a probability per option, else the run stops before any class."""
    body = {"state": "My integration keeps failing with a 401 since we rotated our API key. Please help.",
            "model": "pplx-decider-v1.2",
            "questions": {"team": {"type": "choice", "instructions": "Which team should handle this request?",
                                   "criteria": {"billing": "Charges and refunds", "support": "Technical integration errors",
                                                "sales": "Questions about buying a product"}}}}
    req = urllib.request.Request(f"http://127.0.0.1:{DECIDER_PORT}/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    probs = (out.get("answers") or {}).get("team", {}).get("probabilities") or {}
    if set(probs) != {"billing", "support", "sales"}:
        raise SystemExit(f"the decision model's smoke test failed: {str(out)[:300]}")
    print(f"[stack] decider smoke test: {max(probs, key=probs.get)} "
          f"({', '.join(f'{k} {v:.2f}' for k, v in probs.items())})", flush=True)
    return out


def start_vllm(devices: list[str]) -> subprocess.Popen:
    """All merged on these GPUs, one data-parallel rank per GPU, one endpoint for both tiers (thinking on / off)."""
    if fake():
        args = [sys.executable, "-m", "dm_harness.fake_upstream", "--port", str(VLLM_PORT)]
        proc = _start("upstream", args)
        wait_http(f"http://127.0.0.1:{VLLM_PORT}/v1/models", proc, 2, "upstream")
        return proc
    from full_eval.models import model_path

    args = [str(Path(sys.executable).parent / "vllm"), "serve", str(model_path("all_merged")), "--port", str(VLLM_PORT),
            "--served-model-name", SERVED, "--max-model-len", os.environ.get("DM_MAX_LEN", "32768"),
            "--data-parallel-size", str(len(devices)),
            "--gpu-memory-utilization", os.environ.get("FULL_EVAL_GPU_UTIL", "0.88"),
            "--max-num-seqs", os.environ.get("DM_MAX_SEQS", "256"),
            "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0})]
    proc = _start("vllm", args, {"CUDA_VISIBLE_DEVICES": ",".join(devices)})
    wait_http(f"http://127.0.0.1:{VLLM_PORT}/v1/models", proc, 20, "vllm")
    return proc


def start_jah(config: Path) -> subprocess.Popen:
    args = [str(Path(sys.executable).parent / "jah"), "serve", "-c", str(config)]
    # the vendored source on the path as well: macOS can hide a venv's .pth files, which Python then skips
    src = f"{ROOT / 'third_party' / 'JevAgentHarness' / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    proc = _start("jah", args, {"PYTHONPATH": src})
    wait_http(f"http://127.0.0.1:{JAH_PORT}/stats", proc, 2, "jah")
    return proc


def stop(*procs: subprocess.Popen | None) -> None:
    for p in procs:
        if p is not None and p.poll() is None:
            p.terminate()
    for p in procs:
        if p is not None:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
