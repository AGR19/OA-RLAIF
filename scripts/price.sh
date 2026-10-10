#!/usr/bin/env bash
# Cloud GPU box: record the hourly price of THIS machine once (OA_GPU_USD_PER_HOUR in .env) and show what the budget
# buys.   scripts/price.sh coreweave|vast
# The budget guard and the dry-run planner use this price, so it must be what you actually pay per hour for the
# machine (it bills whether or not its GPUs are busy). OA_YES=1 skips the confirmation.
set -euo pipefail
cd "$(dirname "$0")/.."
PLATFORM="${1:-coreweave}"
touch .env
n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"; n="${n:-0}"
saved="$(grep -E '^OA_GPU_USD_PER_HOUR=' .env | tail -n 1 | cut -d= -f2- | tr -d '"' || true)"
price="${OA_GPU_USD_PER_HOUR:-$saved}"

if [ -z "$price" ]; then
  if [ "$PLATFORM" = vast ]; then
    echo "This machine has $n GPU(s). Enter the \$/hr shown for this instance on the Vast.ai console (Instances"
    echo "page; it includes the disk). Hosts that charge for internet traffic bill the ~80 GB of downloads on top."
    read -r -p "Hourly price in USD: " price
  else
    list="$(awk -v n="$n" 'BEGIN { printf "%.2f", n * 8.60 }')"
    echo "This machine has $n GPU(s). CoreWeave's on-demand list price for B200 is about \$8.60 per GPU-hour"
    echo "(an 8x HGX B200 node: about \$68.80/h). Contracts, spot capacity and credits differ."
    echo "Enter what you pay per hour for this machine. If you pay for a whole node, enter the node price even if"
    echo "this pod uses fewer GPUs."
    read -r -p "Hourly price in USD [$list]: " price
    price="${price:-$list}"
  fi
fi
awk -v p="$price" 'BEGIN { exit !(p + 0 > 0) }' || { echo "not a price: '$price'"; exit 1; }
sed -i '/^OA_GPU_USD_PER_HOUR=/d' .env
echo "OA_GPU_USD_PER_HOUR=$price" >> .env
grep -q '^OA_BILLABLE=' .env || echo 'OA_BILLABLE=1' >> .env

# What the budget buys at this price (estimates from the Lambda dashboard; evaluations do not speed up with more GPUs)
awk -v p="$price" -v stop="${OA_BUDGET_STOP_USD:-450}" 'BEGIN {
  hours = stop / p; fixed = 6.0; slots = 10.0
  train = hours - fixed; if (train < 0) train = 0; if (train > slots) train = slots
  printf "\nAt $%.2f/h the $%d stop buys %.1f h of this machine.\n", p, stop, hours
  printf "Setup, checks, evaluations, export and uploads need about %.0f h; training gets about %.1f h of the planned\n", fixed, train
  printf "%.0f h (OPSD 4, GRPO 4, SFT 2). The dry run fixes the exact slots from the measured speed and the money left.\n", slots
  if (train < 0.5) printf "\nWARNING: that leaves almost no training time; the dry run will stop with the reason.\n" \
                          "Use a machine with fewer GPUs, spot capacity, or a lower price.\n"
}'

# confirm once, before anything is billed for this run
if [ ! -f outputs/stages/run.start ] && [ "${OA_YES:-0}" != 1 ] && [ -t 0 ]; then
  read -r -p "Continue? [y/N] " ok
  case "$ok" in y|Y|yes) ;; *) echo "stopped; nothing ran"; exit 1 ;; esac
fi
