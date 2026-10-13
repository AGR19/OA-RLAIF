# Vast.ai runbook ($500 total, stop at $450)

The same pipeline as on Lambda (`LAMBDA_RUNBOOK.md`), started with one command: `scripts/vast_all.sh`. It runs the same 16 stages and pushes the same private Hugging Face repos. It also handles what is different on Vast.ai.

| Vast.ai difference | What the scripts do |
|---|---|
| Each host sets its own price; storage (and on some hosts, internet traffic) is billed too | The cost calculator reads this instance's price (`dph_total`) from the Vast.ai API with the key Vast places inside every instance, and asks only if that fails. Spend is the machine's billed time from its start; the training stages share the money left (OPSD 40%, GRPO 40%, SFT 20%). |
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

1. It reads the **price of this instance** from the Vast.ai API (or asks for the $/hr on the Instances page) and shows what the budget buys. Then it asks you to confirm.
2. The setup installs the system packages and checks `/dev/shm` and the disk. It sets up the agent sandbox and self-tests it. Then it installs the pinned Python packages and downloads the E4B and A4B weights.
3. It asks once for your **Hugging Face token**. After that, everything runs without input, with the live dashboard.

If SSH drops, reconnect; tmux keeps the run going (`tmux attach`). If a step fails, fix the cause and run `scripts/vast_all.sh` again; finished steps are skipped.

**When it prints DONE:** your GGUF is at `huggingface.co/<you>/oa-rlaif-e4b-gguf` (private). Then **destroy the instance** in the Vast.ai console. A stopped instance still bills for its disk.

**Optional: keep the SFT comparison model.** The one-command run uploads only its scores. To also keep its weights, run `scripts/push_sft.sh` after step 13 (and again after step 14 to add its evaluation). It creates two private repos: `<you>/oa-rlaif-e4b-sft` (LoRA adapter + evaluation + logs) and `<you>/oa-rlaif-e4b-sft-merged` (merged model, about 16 GB, resumable upload). `scripts/push_sft.sh --dry-run` lists what would be uploaded.

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

The calculator (`src/oa_rlaif/cost.py`) works in dollars, not time slots. At the end of the dry run, the money left before the $450 stop, minus what the remaining evaluations, export and uploads will cost at this price (about 4.3 h of machine time), less a 5% reserve, is the **training money**. It is shared out OPSD 40%, GRPO 40%, SFT 20%. Each stage runs the steps its share pays for (price × measured seconds per step), and never more than its full epochs. At the start of each training stage the share is checked again against the money really left, so an overrun in OPSD makes GRPO shorter instead of breaking the budget. Roughly:

  training money ≈ 0.95 × ($450 − 5.8 h × price)

| Example price (not a quote) | Training money | OPSD / GRPO / SFT | Training time |
|---|---:|---:|---:|
| $15/h | $345 | $138 / $138 / $69 | ≈ 23.0 h |
| $25/h | $290 | $116 / $116 / $58 | ≈ 11.6 h |
| $35/h | $235 | $94 / $94 / $47 | ≈ 6.7 h |
| $60/h | $97 | $39 / $39 / $19 | ≈ 1.6 h |

On a cheap machine each stage can run until its full epochs (more steps, longer wall-clock time). `python -m oa_rlaif.cost status` shows the live numbers.
