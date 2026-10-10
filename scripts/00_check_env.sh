#!/usr/bin/env bash
# LAMBDA: verify GPUs, CUDA arch (B200 = sm_100), libraries, Docker, data and the local A4B judge weights.
. "$(dirname "$0")/_common.sh"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
$PY - <<'PY'
import torch, trl, transformers, vllm, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      "arch", [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])
print("trl", trl.__version__, "transformers", transformers.__version__, "vllm", vllm.__version__, "peft", peft.__version__)
from trl.experimental.sdft import SDFTTrainer  # noqa: F401
print("SDFTTrainer import ok")
PY
docker image inspect oa-rlaif-om:latest >/dev/null && echo "docker image ok"
FAIL=0
for f in data/sdft/train.jsonl data/grpo/train.jsonl data/eval/heldout_anatomy.jsonl data/raw/anatomy-dataset-2/mouse.owl; do
  [ -s "$f" ] && echo "ok $f" || { echo "MISSING $f"; FAIL=1; }
done
ls data/dral/*.npz >/dev/null 2>&1 && echo "dral npz: $(ls data/dral/*.npz | wc -l)" || { echo "MISSING data/dral/*.npz"; FAIL=1; }
$PY -c "import torch, sys; n = torch.cuda.device_count(); print('gpus', n); sys.exit(0 if n >= 4 else 1)" || { echo "need 4 GPUs"; FAIL=1; }
$PY -m pytest -q tests/test_harness.py -k docker -p no:cacheprovider || { echo "docker sandbox check failed"; FAIL=1; }
$PY - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo in ("google/gemma-4-E4B-it", os.environ.get("JUDGE_LOCAL_MODEL", "google/gemma-4-26B-A4B-it")):
    try:
        snapshot_download(repo, local_files_only=True)
        print("weights cached:", repo)
    except Exception:
        print("MISSING weights:", repo, "(run scripts/lambda_setup.sh)")
PY
$PY -m oa_rlaif.progress status
[ "$FAIL" = 0 ] || { echo "environment check FAILED"; exit 1; }
echo "environment check passed"
