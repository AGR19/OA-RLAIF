# CoreWeave runbook ($500 total, stop at $450)

The same pipeline as on Lambda (`LAMBDA_RUNBOOK.md`), started with one command: `scripts/coreweave_all.sh`. It runs the same 16 stages and pushes the same private Hugging Face repos. It also handles the ways CoreWeave differs from a Lambda instance.

| CoreWeave difference | What the scripts do |
|---|---|
| You pay per machine hour, and prices vary by contract | The cost calculator records the price of this machine. It asks you, and refuses a price that is implausible for the GPUs `nvidia-smi` reports. It shows what the $450 stop buys before anything runs. Spend is the machine's billed time from its own start, across restarts. |
| A higher price than Lambda's | The plan is in dollars: the training stages share what is left after the fixed stages, so the run still ends with the GGUF, every evaluation and the SFT row inside the budget. If too little is left for training, the dry run stops and says why. |
| Docker is often missing, and pods cannot always run it | `scripts/sandbox_setup.sh` installs Docker Engine and starts it. It detects the machine type (case table below). |
| 8 GPUs per node, sometimes H100/H200 instead of B200 | The GPU layout follows the GPU count and size (layout table below). |
| Pods get a 64 MB `/dev/shm` | The setup stops with the pod-spec fix, because NCCL and vLLM crash with so little shared memory. |
| Pods restart, and the root disk is temporary | The Hugging Face and uv caches are kept in `.cache/` next to the repo. Put the repo on the persistent volume, and the run resumes after a restart. |

## 1. The machine you need

Any Ubuntu (or Debian) machine with at least 4 GPUs, at least 180 GB free on the volume that holds the repo (250 GB recommended), and internet access for the setup.

| You have | Docker | What you do |
|---|---|---|
| **Virtual Server or bare-metal node** (SSH, systemd, sudo) | Installed from Docker's apt repository and started with systemd | Nothing extra |
| **CKS pod or DevPod, privileged** | Installed, then `dockerd` is started in the background | Nothing extra |
| **CKS pod, not privileged** | Cannot run | Use the pod spec below, or a Docker-in-Docker sidecar (option 2 in the script's message) |
| **A pod with a dind sidecar** | Uses the sidecar through `DOCKER_HOST=tcp://localhost:2375` in `.env` | Mount the workspace at the **same path** in both containers, because each agent episode bind-mounts its folder into the sandbox |
| **SUNK (Slurm) login node** | Not supported by default (CoreWeave's Docker-in-Docker for SUNK needs an admin change) | Run in a CKS pod or on a Virtual Server instead |

A CKS pod spec with everything the run needs. Adjust the image, the volume claim and the node labels for your cluster:

```yaml
apiVersion: v1
kind: Pod
metadata: {name: oa-rlaif}
spec:
  containers:
    - name: main
      image: nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04   # any Ubuntu image with the NVIDIA runtime
      command: ["sleep", "infinity"]
      securityContext: {privileged: true}                 # Docker for the agent sandbox
      resources:
        limits: {nvidia.com/gpu: 8}                       # 4 also works; add cpu/memory to suit your node
      volumeMounts:
        - {name: work, mountPath: /mnt/work}              # clone the repo here
        - {name: dshm, mountPath: /dev/shm}
  volumes:
    - {name: work, persistentVolumeClaim: {claimName: oa-rlaif-work}}   # >= 250 GB
    - {name: dshm, emptyDir: {medium: Memory, sizeLimit: 128Gi}}
```

GPU layout (decided in `scripts/_common.sh` from `nvidia-smi`):

| GPUs | Training | vLLM (rollouts, agents) | A4B judge |
|---|---|---|---|
| 4 × B200 | 0–2 | 3 | 3 (shared, as on Lambda) |
| 8 × B200 | 0–6 | 7 | 7 (shared) |
| 8 × H100 / H200 (< 150 GB) | 0–5 | 7 | 6 (its own GPU: the 52 GB judge does not fit next to the policy) |

## 2. Run

Have ready a GitHub token (read access to `AGR19/OA-RLAIF`) and a Hugging Face token **with write access**. Open a shell on the machine (`kubectl exec -it oa-rlaif -- bash`, or SSH), then:

```bash
tmux new -s run
```

```bash
cd /mnt/work && git clone https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF
```

```bash
scripts/coreweave_all.sh
```

1. It asks for the **hourly price of this machine** and shows what the budget buys. Then it asks you to confirm.
2. The setup installs the system packages and Docker, and checks `/dev/shm` and the disk. It builds the sandbox image and self-tests it with no network and a bind mount. Then it installs the pinned Python packages and downloads the E4B and A4B weights.
3. It asks once for your **Hugging Face token**. After that, everything runs without input, with the same live dashboard as on Lambda.

If your connection drops, run `tmux attach -t run`. If a step fails, fix the cause and run `scripts/coreweave_all.sh` again; finished steps are skipped. After a pod restart, run the same command: Docker is reinstalled or restarted automatically.

**When it prints DONE:** your GGUF is at `huggingface.co/<you>/oa-rlaif-e4b-gguf` (private). Then **release the node**: delete the pod or instance, or scale its node pool to 0. Billing continues until you do.

## 3. What the budget buys

The calculator (`src/oa_rlaif/cost.py`) works in dollars, not time slots. At the end of the dry run, the money left before the $450 stop, minus what the remaining evaluations, export and uploads will cost at this price (about 4.3 h of machine time), less a 5% reserve, is the **training money**. It is shared out OPSD 40%, GRPO 40%, SFT 20%. Each stage runs the steps its share pays for (price × measured seconds per step), and never more than its full epochs. At the start of each training stage the share is checked again against the money really left, so an overrun in OPSD makes GRPO shorter instead of breaking the budget. Roughly:

  training money ≈ 0.95 × ($450 − 5.8 h × price)

| Machine price | Training money | OPSD / GRPO / SFT | Training time |
|---|---:|---:|---:|
| $27.16/h (Lambda 4×B200, for comparison) | $278 | $111 / $111 / $56 | ≈ 10.2 h |
| $34.11/h (8×B200, lowest published CoreWeave spot rate) | $240 | $96 / $96 / $48 | ≈ 7.0 h |
| $34.40/h (4 GPUs at $8.60 per GPU-hour) | $238 | $95 / $95 / $48 | ≈ 6.9 h |
| $68.80/h (8×B200 on-demand list price) | $48 | $19 / $19 / $10 | ≈ 0.7 h |

These are third-party figures for CoreWeave's rate card; enter what you actually pay. At the 8×B200 on-demand list price, $500 does not pay for a useful training run. Use spot capacity, a 4-GPU machine billed per GPU, or a contract price.

## 4. Results

The outputs are the same as on Lambda (`LAMBDA_RUNBOOK.md`, section 3). `outputs/step_plan.json["budget"]` records the price, the money that was left and each stage's share; `outputs/cost_ledger.json` holds the price source and the machine's billed time. `python -m oa_rlaif.cost status` prints them.
