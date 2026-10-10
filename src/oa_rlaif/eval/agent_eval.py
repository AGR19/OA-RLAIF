"""Evaluate a model inside mini-SWE-agent 2.4.6 (plan steps 8, 11, 15): it runs `om` in a sandbox per episode.

    python -m oa_rlaif.eval.agent_eval --episodes data/eval/heldout_anatomy.jsonl --n 200 \
        --model hosted_vllm/oa-rlaif-e4b --api-base http://localhost:8000/v1 --env docker

Each episode gets its own task directory with the PUBLIC episode (no gold; harness.sandbox.public_episode), mounted
at /task (docker), owned by the sandbox user (user: machines without Docker, e.g. Vast.ai containers) or used
directly (local: tests). The submission (TSV from `om submit`) is scored against the gold kept in this process;
trajectories are saved for inspection. --env defaults to OA_AGENT_ENV (written by scripts/sandbox_setup.sh), else
docker.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from ..harness.sandbox import SANDBOX_USER, describe, public_episode
from ..io import read_jsonl, read_yaml
from ..paths import ROOT, out_dir
from ..progress import stage


def instance_text(episode: dict) -> str:
    relation = "classified under" if episode.get("relation") == "classified_under" else "equivalent to"
    return (f"Align source entity {episode['source_iri']} ({episode['pair_id']}). Find the candidate target it is "
            f"{relation}, or NONE. Use `om show` and `om candidates`, then submit.")


def make_environment(kind: str, task_dir: Path, cfg: dict):
    env_cfg = cfg["environment"]
    if kind == "user":
        from ..harness.sandbox import make_user_sandbox

        return make_user_sandbox(task_dir, timeout=env_cfg.get("timeout", 30))
    if kind == "docker":
        from minisweagent.environments.docker import DockerEnvironment

        return DockerEnvironment(image=env_cfg["image"], cwd="/task", timeout=env_cfg.get("timeout", 30),
                                 env=env_cfg.get("env", {}),
                                 run_args=["--rm", "--network", "none", "-v", f"{task_dir.resolve()}:/task"])
    from minisweagent.environments.local import LocalEnvironment

    # PYTHONPATH: the `om` subprocess must import oa_rlaif even if the venv's .pth files are skipped (macOS sync)
    src = str(Path(__file__).resolve().parents[2])
    env = {"OM_TASK": str(task_dir / "episode.json"), "PAGER": "cat",
           "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
           "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}
    return LocalEnvironment(cwd=str(task_dir), env=env, timeout=env_cfg.get("timeout", 30))


def parse_submission(submission: str) -> dict | None:
    lines = [line for line in (submission or "").strip().splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith("source_iri"):
        return None
    rows = [line.split("\t") for line in lines[1:4]]
    source, target, relation, confidence = rows[0]
    return {"source": source, "target": target, "targets": [r[1] for r in rows], "relation": relation,
            "confidence": float(confidence)}


def run(episodes: list[dict], model, env_kind: str, workdir: Path, cfg: dict) -> list[dict]:
    from minisweagent.agents.default import DefaultAgent

    agent_cfg = {k: v for k, v in cfg["agent"].items()}
    results = []
    with stage("agent_eval", total=len(episodes)) as bar:
        for episode in episodes:
            run_dir = workdir / episode["id"]  # trajectory and outputs (the repo side)
            run_dir.mkdir(parents=True, exist_ok=True)
            if env_kind == "user":
                from ..harness.sandbox import prepare_task_dir

                task_dir = prepare_task_dir(workdir.name, episode["id"])
            else:
                task_dir = run_dir
            # the agent can read every file in its task folder: give it the public episode only, never the gold
            (task_dir / "episode.json").write_text(json.dumps(public_episode(episode)))
            if env_kind == "user":
                import pwd

                entry = pwd.getpwnam(SANDBOX_USER)
                os.chown(task_dir / "episode.json", entry.pw_uid, entry.pw_gid)
            env = make_environment(env_kind, task_dir, cfg)
            agent = DefaultAgent(model, env, **agent_cfg)
            try:
                outcome = agent.run(task=instance_text(episode))
            finally:
                getattr(env, "cleanup", lambda: None)()
            decision = parse_submission(outcome.get("submission", ""))
            target = decision["target"] if decision else None
            targets = set(decision["targets"]) if decision else set()
            gold = set(episode.get("gold") or [])
            correct = target is not None and ((target == "NONE" and not gold) or (bool(gold) and targets == gold))
            results.append({"id": episode["id"], "pair_id": episode["pair_id"], "exit": outcome.get("exit_status"),
                            "target": target, "correct": bool(correct), "steps": len(agent.messages)})
            (run_dir / "trajectory.json").write_text(json.dumps(agent.messages, default=str)[:2_000_000])
            bar.advance()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", default="hosted_vllm/oa-rlaif-e4b")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--env", choices=["docker", "user", "local"], default=os.environ.get("OA_AGENT_ENV") or "docker")
    parser.add_argument("--tag", default="agent")
    args = parser.parse_args()
    cfg = read_yaml(ROOT / "configs" / "mini_swe_agent.yaml")
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

    model_cfg = dict(cfg["model"])
    model_cfg.pop("model_class", None)
    model_cfg.update(model_name=args.model, model_kwargs={**model_cfg.get("model_kwargs", {}), "api_base": args.api_base})
    model = LitellmTextbasedModel(**model_cfg)
    episodes = list(read_jsonl(args.episodes))
    random.Random(7).shuffle(episodes)
    workdir = out_dir() / "agent_runs" / args.tag
    results = run(episodes[: args.n], model, args.env, workdir, cfg)
    accuracy = sum(r["correct"] for r in results) / max(1, len(results))
    summary = {"episodes": len(results), "accuracy": round(accuracy, 4),
               "submitted": sum(r["exit"] == "Submitted" for r in results), "tag": args.tag, **describe(args.env)}
    (workdir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    (workdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
