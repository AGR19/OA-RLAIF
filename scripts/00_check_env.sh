#!/usr/bin/env bash
# GPU box (Lambda or CoreWeave): verify GPUs and their layout, shared memory, libraries, Docker, data and the local
# A4B judge weights. Any missing item fails the check.
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
FAIL=0
log "GPU layout: train on $TRAIN_GPUS ($NTRAIN processes), vLLM on $VLLM_GPU, A4B judge on $JUDGE_GPU, eval tp $EVAL_TP"
shm_gb="$(df -Pk /dev/shm | awk 'NR == 2 { print int($2 / 1048576) }')"
[ "${shm_gb:-0}" -ge 8 ] && echo "ok /dev/shm ${shm_gb} GB" || { echo "/dev/shm is ${shm_gb} GB; need >= 8 (pods: mount an emptyDir with medium: Memory)"; FAIL=1; }
AGENT_ENV="${OA_AGENT_ENV:-docker}"
log "agent sandbox: $AGENT_ENV${OA_AGENT_NET:+ (network: $OA_AGENT_NET)}"
if [ "$AGENT_ENV" = docker ]; then
  docker info >/dev/null 2>&1 && echo "ok docker daemon" || { echo "docker daemon not reachable (scripts/sandbox_setup.sh)"; FAIL=1; }
  docker image inspect oa-rlaif-om:latest >/dev/null 2>&1 && echo "ok docker image" || { echo "MISSING docker image oa-rlaif-om:latest"; FAIL=1; }
else
  id oa-agent >/dev/null 2>&1 && [ -x /opt/om/bin/om ] && echo "ok user sandbox" || { echo "user sandbox missing (scripts/sandbox_setup.sh)"; FAIL=1; }
fi
for f in data/sdft/train.jsonl data/grpo/train.jsonl data/eval/heldout_anatomy.jsonl data/raw/anatomy-dataset-2/mouse.owl; do
  [ -s "$f" ] && echo "ok $f" || { echo "MISSING $f"; FAIL=1; }
done
ls data/dral/*.npz >/dev/null 2>&1 && echo "dral npz: $(ls data/dral/*.npz | wc -l)" || { echo "MISSING data/dral/*.npz"; FAIL=1; }
$PY -c "import torch, sys; n = torch.cuda.device_count(); print('gpus', n); sys.exit(0 if n >= int(sys.argv[1]) else 1)" "${OA_MIN_GPUS:-4}" \
  || { echo "need at least ${OA_MIN_GPUS:-4} GPUs; nvidia-smi shows $NGPU"; FAIL=1; }
# the full agent loop in the sandbox the evaluations will use (scripted model, no GPU)
SANDBOX_TEST=docker_sandbox; [ "$AGENT_ENV" = user ] && SANDBOX_TEST=user_sandbox
$PY -m pytest -q -rs tests/test_harness.py -k "$SANDBOX_TEST" -p no:cacheprovider | tee outputs/logs/check_sandbox.log || { echo "agent sandbox check failed"; FAIL=1; }
grep -q "1 passed" outputs/logs/check_sandbox.log || { echo "agent sandbox test did not pass (skipped or failed)"; FAIL=1; }
$PY - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo in ("google/gemma-4-E4B-it", os.environ.get("JUDGE_LOCAL_MODEL", "google/gemma-4-26B-A4B-it")):
    try:
        snapshot_download(repo, local_files_only=True)
        print("weights cached:", repo)
    except Exception:
        print("MISSING weights:", repo, "(run the setup script)")
        open("outputs/logs/check_weights.failed", "w").close()
PY
[ -f outputs/logs/check_weights.failed ] && { rm -f outputs/logs/check_weights.failed; FAIL=1; }
$PY -m oa_rlaif.progress status
[ "$FAIL" = 0 ] || { echo "environment check FAILED"; exit 1; }
echo "environment check passed"
