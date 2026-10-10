# Vast.ai runbook ($500 total, stop at $450)

The same pipeline as on Lambda (`LAMBDA_RUNBOOK.md`), started with one command: `scripts/vast_all.sh`. It runs the same 16 stages and pushes the same private Hugging Face repos. It also handles what is different on Vast.ai.

| Vast.ai difference | What the scripts do |
|---|---|
| Each host sets its own price; storage (and on some hosts, internet traffic) is billed too | Ask once for the instance's $/hr. Show what the $450 stop buys. The budget guard counts the machine's wall-clock time from the first start, and the dry run fits the OPSD, GRPO and SFT slots to the money left. |
| A normal instance is itself an unprivileged Docker container, so it cannot run Docker | The agent evaluations use the **user sandbox** (below). On a VM instance, where Docker can run, Docker is installed and used instead. |
| Any GPU count and type | The GPU layout follows the count and size. On GPUs under 150 GB, the A4B judge gets its own GPU. |
| The disk size is fixed when you create the instance | The setup checks for at least 180 GB free and stops early if there is less. |
| `/dev/shm` depends on the host | The setup checks for at least 8 GB (NCCL and vLLM need it). |

## 1. Choose the instance

- **GPUs:** at least 4. B200 or H200 is best; H100 also works (the judge then gets its own GPU). 8 GPUs train faster, but the price is what limits the budget.
- **Disk:** at least **250 GB**. It cannot grow later.
- **Image:** any Ubuntu 22.04 or 24.04 image with CUDA, for example a Vast.ai PyTorch or CUDA template. Choose SSH access.
- **Host:** a datacenter or verified host with fast download (the setup downloads about 80 GB) and no or low per-TB internet price. Prefer **on-demand** to interruptible: an interrupted instance stops the run, and the stage it was in starts again on the next run.

## 2. Run

Have ready a GitHub token (read access to `AGR19/OA-RLAIF`) and a Hugging Face token **with write access**. SSH into the instance. Vast.ai's SSH login normally opens a tmux session; if your prompt is not inside tmux, run `tmux new -s run` first. Then:

```bash
cd /workspace 2>/dev/null || cd ~; git clone https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF
```

```bash
scripts/vast_all.sh
```

1. It asks for the **$/hr of this instance** (shown on the Vast.ai Instances page) and shows what the budget buys. Then it asks you to confirm.
2. The setup installs the system packages and checks `/dev/shm` and the disk. It sets up the agent sandbox and self-tests it. Then it installs the pinned Python packages and downloads the E4B and A4B weights.
3. It asks once for your **Hugging Face token**. After that, everything runs without input, with the live dashboard.

If SSH drops, reconnect; tmux keeps the run going (`tmux attach`). If a step fails, fix the cause and run `scripts/vast_all.sh` again; finished steps are skipped.

**When it prints DONE:** your GGUF is at `huggingface.co/<you>/oa-rlaif-e4b-gguf` (private). Then **destroy the instance** in the Vast.ai console. A stopped instance still bills for its disk.

## 3. The agent sandbox on Vast.ai

The mini-SWE-agent evaluation (Table 2) runs shell commands written by the model. On Lambda and CoreWeave they run in Docker with no network. A normal Vast.ai instance cannot start Docker, so `scripts/sandbox_setup.sh` sets up the **user sandbox** (`src/oa_rlaif/harness/sandbox.py`):

| Protection | Docker (Lambda, CoreWeave, Vast VM) | User sandbox (Vast container) |
|---|---|---|
| What the agent can read | its task folder only | its task folder; the repo (data, gold, outputs, caches) is closed to the `oa-agent` user |
| The answer in the task folder | never (public fields only) | never (public fields only) |
| User | `agent` in the container | `oa-agent`, unprivileged, empty environment |
| Network | none | none if the container allows `unshare -n`; otherwise **open** |

The self-test shows which case applies. Table 2's caption in the report states the sandbox and the network setting, so the results are reported honestly. For Docker-grade isolation, rent a VM instance; the same command then installs and uses Docker.

## 4. What the budget buys

The $450 stop divided by the instance price gives the hours. About 6 h go to everything except training: setup, checks, dry run, baseline, three evaluations, export and uploads. Training gets the rest, up to the planned 10 h (OPSD 4 h, GRPO 4 h, SFT 2 h):

  training hours ≈ min(10, 450 / price − 6)

| Example price (not a quote) | Hours for $450 | Training time |
|---|---:|---:|
| $15/h | 30 | 10 h (full slots) |
| $25/h | 18 | 10 h (full slots) |
| $35/h | 12.9 | ≈ 6.9 h |
| $60/h | 7.5 | ≈ 1.5 h |

The dry run fixes the exact slots from the measured speed and the money really left (`outputs/step_plan.json["budget"]`).
