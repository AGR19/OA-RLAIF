"""Cost calculator for a GPU run: the real price of this machine, what it has cost so far, and a plan in dollars.

    python -m oa_rlaif.cost price --platform lambda|coreweave|vast   # find and record the price (asks if it must)
    python -m oa_rlaif.cost status                                   # price, spend, money left, plan in dollars

1. Price of THIS instance, never a built-in default, trusted in this order:
     a. the provider's API: Vast.ai (CONTAINER_ID and CONTAINER_API_KEY, which Vast sets inside each instance) or
        Lambda (LAMBDA_API_KEY in .env; the instance whose IP is one of this machine's IPs);
     b. OA_GPU_USD_PER_HOUR given in the shell (a value in .env is only offered as a suggestion);
     c. the price recorded earlier for the SAME GPUs (a repo moved to another machine asks again);
     d. asking the user.
   Every price that is not from the API is checked against the GPUs that nvidia-smi reports: a price per GPU-hour
   outside the plausible range for that GPU model is refused unless confirmed (for example $27/h for one A100-40GB).
   Recorded with its source and the GPUs in outputs/cost_ledger.json.
2. Spend: GPU clouds bill per hour of the MACHINE, from its own start, not from when our scripts started. The
   start is the start of PID 1 (VM boot, or the container start). Each machine start is one segment; a restart
   opens a new one. Spend = sum over segments of (length x price), plus the API costs in the budget ledger.
3. Plan in dollars (train.common.plan_steps): the money left minus the fixed stages still to come is shared out
   (configs/train.yaml budget_plan: OPSD 40%, GRPO 40%, SFT 20%); steps = dollars / (price x measured seconds per
   step). The plan is made at the end of the dry run and re-checked against the money really left at the start
   of each training stage.

Standard library only: run_all.sh uses it with the system Python before anything is installed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# A price given in the shell, read before .paths loads .env into the environment: a stale .env value must never
# pass for one the user just gave.
SHELL_PRICE = os.environ.get("OA_GPU_USD_PER_HOUR", "").strip()

from . import jsonfile  # noqa: E402
from .paths import ROOT, out_dir  # noqa: E402

# Plausible on-demand prices per GPU-hour (USD), wide on purpose: they only catch a wrong machine or a typo.
PLAUSIBLE_PER_GPU = [
    (r"GB300|B300", 2.5, 25.0),
    (r"GB200|B200", 2.0, 18.0),
    (r"H200", 1.5, 14.0),
    (r"H100|H800|GH200", 1.0, 12.0),
    (r"A100.*80|A100-SXM4-80|A100 80", 0.5, 6.0),
    (r"A100", 0.3, 4.0),
    (r"L40|L4|A10|A6000|RTX|4090|5090|3090", 0.1, 3.0),
]
UNKNOWN_RANGE = (0.1, 25.0)
LAMBDA_API_BASES = ("https://cloud.lambda.ai/api/v1", "https://cloud.lambdalabs.com/api/v1")
VAST_API_BASE = "https://console.vast.ai/api/v0"

# The plan in dollars (single source; train.common.plan_steps uses it). Hours of the stages that do not train are
# estimates on 4xB200 (scripts/dashboard.py); the money for them is set aside at the real price.
PLAN = {
    "shares": {"sdft": 0.4, "grpo": 0.4, "sft": 0.2},        # of the training money: OPSD, GRPO, SFT comparison
    "reserve": 0.05,                                          # of the money left, kept back for overruns
    "setup_hours": 1.5,                                       # setup, data, checks, GGUF check, dry run
    # non-training hours still to come at each point: baseline, 3 evaluations, export, uploads, report
    "fixed_hours_after": {"plan": 4.3, "sdft": 3.5, "grpo": 2.8, "sft": 0.9},
    "min_training_usd": 25.0,                                 # below this the dry run stops and says why
    "order": ("sdft", "grpo", "sft"),
}
LABELS = {"sdft": "OPSD", "grpo": "GRPO", "sft": "SFT"}


# ---- machine ----------------------------------------------------------------------------------------------------
def gpus() -> list[tuple[str, int]]:
    """(name, memory GB) per GPU from nvidia-smi; empty when there is none."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        name, _, mem = line.rpartition(",")
        if name.strip():
            rows.append((name.strip(), int(float(mem.strip() or 0) / 1024)))
    return rows


def machine_start() -> float | None:
    """When this machine (VM) or container started: the start time of PID 1. None off Linux."""
    try:
        btime = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                     if line.startswith("btime "))
        fields = Path("/proc/1/stat").read_text().rsplit(")", 1)[1].split()
        ticks = int(fields[19])  # field 22 of /proc/[pid]/stat: starttime, in clock ticks after boot
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def local_ips() -> set[str]:
    try:
        return set(subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=10).stdout.split())
    except (OSError, subprocess.SubprocessError):
        return set()


def plausible(usd_per_hour: float, machine_gpus: list[tuple[str, int]]) -> tuple[bool, str]:
    """Is this hourly price believable for these GPUs? Returns (ok, explanation)."""
    if not machine_gpus:
        return usd_per_hour > 0, "no GPUs found to check the price against"
    name = machine_gpus[0][0]
    lo, hi = next(((lo, hi) for pattern, lo, hi in PLAUSIBLE_PER_GPU if re.search(pattern, name, re.I)), UNKNOWN_RANGE)
    n = len(machine_gpus)
    per_gpu = usd_per_hour / n
    text = (f"${usd_per_hour:.2f}/h for {n} x {name} = ${per_gpu:.2f} per GPU-hour; "
            f"believable range for this GPU: ${lo:.2f}-${hi:.2f} per GPU-hour (${lo * n:.2f}-${hi * n:.2f}/h)")
    return lo <= per_gpu <= hi, text


# ---- provider APIs ------------------------------------------------------------------------------------------------
def _get_json(url: str, headers: dict, timeout: float = 15) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https endpoints)
        return json.loads(response.read().decode())


def env_file() -> Path:
    return Path(os.environ.get("OA_ENV_FILE") or ROOT / ".env")


def _env(name: str) -> str:
    """From the environment, else from .env (the runner does not export secrets)."""
    if os.environ.get(name):
        return os.environ[name]
    path = env_file()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def vast_price() -> tuple[float, str] | None:
    """Vast.ai: this instance's dph_total ($/h, GPU + disk) through the instance's own API key."""
    instance, key = os.environ.get("CONTAINER_ID", ""), os.environ.get("CONTAINER_API_KEY", "")
    if not (instance and key):
        return None
    try:
        data = _get_json(f"{VAST_API_BASE}/instances/{instance}/", {"Authorization": f"Bearer {key}"})
    except Exception as error:  # noqa: BLE001 (any API problem falls back to asking)
        print(f"[cost] Vast.ai API not reachable ({type(error).__name__}); asking for the price instead")
        return None
    record = data.get("instances", data)
    record = record[0] if isinstance(record, list) and record else record
    price = record.get("dph_total") if isinstance(record, dict) else None
    if not price:
        return None
    return float(price), f"Vast.ai API, instance {instance} (dph_total; internet traffic is billed separately)"


def lambda_price() -> tuple[float, str] | None:
    """Lambda: the price of the running instance whose IP belongs to this machine (needs LAMBDA_API_KEY)."""
    key = _env("LAMBDA_API_KEY")
    if not key:
        return None
    auth = {"Authorization": "Basic " + base64.b64encode(f"{key}:".encode()).decode()}
    data, last_error = None, None
    for base in LAMBDA_API_BASES:
        try:
            data = _get_json(f"{base}/instances", auth)
            break
        except Exception as error:  # noqa: BLE001
            last_error = error
    if data is None:
        print(f"[cost] Lambda API not reachable ({type(last_error).__name__}); asking for the price instead")
        return None
    running = [i for i in data.get("data", []) if i.get("status", "active") == "active"]
    mine = local_ips()
    match = [i for i in running if {i.get("ip"), i.get("private_ip")} & mine]
    note = "matched by IP"
    if not match and len(running) == 1:
        match, note = running, "the only active instance on the account"
    if len(match) != 1:
        print(f"[cost] Lambda API: could not tell which of {len(running)} active instances is this one; asking instead")
        return None
    itype = match[0].get("instance_type", {})
    cents = itype.get("price_cents_per_hour")
    if cents is None:
        return None
    return cents / 100, f"Lambda API, {itype.get('name', '?')} ({note})"


# ---- ledger and meter ---------------------------------------------------------------------------------------------
class Meter:
    """The machine's billed time and price, in a JSON ledger next to the budget ledger. Several processes (one per
    GPU, plus the dashboard) use it at once: every change is a locked read-modify-write (jsonfile.update), and every
    reading starts from the file, so no process works from a stale copy."""

    EMPTY = {"price": None, "segments": []}

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else out_dir() / "cost_ledger.json"
        self.data = self._read()

    def _read(self) -> dict:
        return jsonfile.read(self.path) or {"price": None, "segments": []}

    @property
    def price(self) -> float | None:
        return (self.data.get("price") or {}).get("usd_per_hour")

    @staticmethod
    def _note(data: dict, start: float, now: float) -> None:
        """Open the segment of this machine start, or extend it; it takes the current price."""
        price = data["price"]["usd_per_hour"]
        current = next((s for s in data["segments"] if abs(s["start"] - start) < 5), None)
        if current is None:
            data["segments"].append({"start": start, "last_seen": now, "usd_per_hour": price})
        else:
            current["last_seen"] = max(current["last_seen"], now)
            current["usd_per_hour"] = price

    def set_price(self, usd_per_hour: float, source: str, machine_gpus: list[tuple[str, int]]) -> None:
        record = {"usd_per_hour": round(usd_per_hour, 4), "source": source, "set_at": time.time(),
                  "gpus": gpu_signature(machine_gpus)}
        start, now = machine_start(), time.time()

        def change(data):
            data.setdefault("segments", [])
            data["price"] = record
            if start is not None:
                self._note(data, start, now)
        self.data = jsonfile.update(self.path, change, default=lambda: {"price": None, "segments": []})

    def observe(self, now: float | None = None) -> None:
        """Note that this machine is up now (the main training process and the dashboard call this)."""
        self.data = self._read()
        start = machine_start()
        if self.price is None or start is None:
            return  # nothing is recorded before a price is set
        now = now or time.time()

        def change(data):
            if data.get("price"):
                self._note(data, start, now)
        self.data = jsonfile.update(self.path, change, default=lambda: {"price": None, "segments": []})

    def spent(self, now: float | None = None) -> float:
        """Dollars billed so far: closed segments up to when they were last seen; the live one up to `now`.
        With the same `now`, every process gets the same number (OA_PLAN_TIME uses this)."""
        self.data = self._read()
        if self.price is None:
            return 0.0
        now = now or time.time()
        live = machine_start()
        total = 0.0
        for s in self.data["segments"]:
            end = now if live is not None and abs(s["start"] - live) < 5 else s["last_seen"]
            total += max(0.0, end - s["start"]) / 3600 * s["usd_per_hour"]
        return total


# ---- price discovery (CLI) ----------------------------------------------------------------------------------------
def gpu_signature(machine_gpus: list[tuple[str, int]]) -> list[str]:
    return [f"{n} ({m} GB)" for n, m in machine_gpus]


def discover(platform: str, meter: "Meter", machine_gpus: list[tuple[str, int]]) -> tuple[float, str, bool] | None:
    """(price, source, from_api) from the most trusted source available, or None (then ask)."""
    api = vast_price() if platform == "vast" else lambda_price() if platform == "lambda" else None
    if api:
        return api[0], api[1], True
    if SHELL_PRICE:
        return float(SHELL_PRICE), "OA_GPU_USD_PER_HOUR (shell)", False
    recorded = meter.data.get("price") or {}
    if recorded.get("usd_per_hour") and recorded.get("gpus") == gpu_signature(machine_gpus):
        return recorded["usd_per_hour"], recorded.get("source", "recorded earlier"), False
    if recorded.get("usd_per_hour"):
        print(f"[cost] the recorded price (${recorded['usd_per_hour']}/h) was for other GPUs "
              f"({', '.join(recorded.get('gpus') or ['?'])}); asking again")
    return None


def ask(platform: str, machine_gpus: list[tuple[str, int]]) -> tuple[float, str]:
    hints = {"lambda": "the $/hr of this instance type on the Lambda console (or put LAMBDA_API_KEY in .env and the "
                       "price is read automatically)",
             "coreweave": "what you pay per hour for this machine (for a whole node, the node price)",
             "vast": "the $/hr shown for this instance on the Vast.ai Instances page"}
    if not sys.stdin.isatty():
        raise SystemExit(f"[cost] no price for this machine. Set OA_GPU_USD_PER_HOUR to {hints.get(platform, 'its $/hr')}.")
    desc = ", ".join(sorted({f"{sum(1 for g in machine_gpus if g[0] == n)} x {n}" for n, _ in machine_gpus})) or "no GPUs found"
    print(f"[cost] This machine: {desc}. Enter {hints.get(platform, 'its $/hr')}.")
    hint = _env("OA_GPU_USD_PER_HOUR")
    value = input(f"Hourly price in USD{f' [.env has {hint}]' if hint else ''}: ").strip().lstrip("$") or hint
    try:
        return float(value), "entered by the user"
    except ValueError:
        raise SystemExit(f"[cost] not a price: '{value}'") from None


def confirm_implausible(price: float, why: str) -> None:
    print(f"[cost] This price does not look right: {why}.")
    if os.environ.get("OA_PRICE_CONFIRMED") == "1":
        print("[cost] OA_PRICE_CONFIRMED=1: using it anyway.")
        return
    if not sys.stdin.isatty():
        raise SystemExit("[cost] stopped. Check the price, or set OA_PRICE_CONFIRMED=1 if it is right.")
    again = input(f"Type the price again to confirm ${price:.2f}/h, or press Enter to stop: ").strip().lstrip("$")
    if not again or abs(float(again) - price) > 1e-6:
        raise SystemExit("[cost] stopped; nothing was installed.")


def set_env_price(price: float) -> None:
    """Keep .env in step (older tools read OA_GPU_USD_PER_HOUR); the ledger is the source of truth."""
    path = env_file()
    example = ROOT / ".env.example"
    source = path if path.exists() else (example if example.exists() and path == ROOT / ".env" else None)
    lines = source.read_text().splitlines() if source else []
    lines = [line for line in lines if not line.startswith("OA_GPU_USD_PER_HOUR=")] + [f"OA_GPU_USD_PER_HOUR={price}"]
    if not any(line.startswith("OA_BILLABLE=") for line in lines):
        lines.append("OA_BILLABLE=1")
    path.write_text("\n".join(lines) + "\n")


# ---- the plan in dollars ------------------------------------------------------------------------------------------
def make_plan(price: float, spent: float, stop: float) -> dict:
    """Share out the money left after the fixed stages (called at the end of the dry run)."""
    left = stop - spent
    fixed = PLAN["fixed_hours_after"]["plan"] * price
    pool = max(0.0, (left - fixed) * (1 - PLAN["reserve"]))
    return {"usd_per_hour": price, "spent_usd": round(spent, 2), "stop_usd": stop, "left_usd": round(left, 2),
            "fixed_usd": round(fixed, 2), "training_usd": round(pool, 2),
            "allocated_usd": {k: round(pool * v, 2) for k, v in PLAN["shares"].items()}, "made_at": time.time()}


def stage_dollars(plan: dict, stage: str, price: float, spent_now: float, stop: float) -> float:
    """What a training stage may spend now: its share, capped by the money really left after the stages to come."""
    order = PLAN["order"]
    later = sum(plan["allocated_usd"][k] for k in order[order.index(stage) + 1:])
    room = (stop - spent_now) - PLAN["fixed_hours_after"][stage] * price - later
    return max(0.0, min(plan["allocated_usd"][stage], room))


def budget_summary(meter: Meter) -> list[str]:
    stop = float(os.environ.get("OA_BUDGET_STOP_USD") or 450)
    price = meter.price or 0.0
    spent = meter.spent()
    planned = out_dir() / "step_plan.json"
    plan = (json.loads(planned.read_text()).get("budget") if planned.exists() else None) or None
    dry_run_done = (out_dir() / "stages" / "dry_run.done").exists()
    setup = 0.0 if dry_run_done else PLAN["setup_hours"] * price
    if plan is None:
        plan = make_plan(price, spent + setup, stop)
    start = min((s["start"] for s in meter.data["segments"]), default=None)
    since = time.strftime("%Y-%m-%d %H:%M", time.localtime(start)) if start else "unknown"
    lines = [f"price      ${price:.2f}/h  ({(meter.data.get('price') or {}).get('source', 'not set')})",
             f"spent      ${spent:,.2f} of the ${stop:.0f} stop (this machine since {since}, all restarts)",
             f"to come    {'' if dry_run_done else f'setup + dry run ~${setup:,.0f}, '}evaluations, export, uploads "
             f"~${plan['fixed_usd']:,.0f} (estimates at this price)",
             "training   " + f"${plan['training_usd']:,.0f}: " + ", ".join(
                 f"{LABELS[k]} ${v:,.0f}" + (f" (~{v / price:.1f} h)" if price else "") for k, v in plan["allocated_usd"].items())
             + " (each stage stops at its full epochs if that is cheaper)"]
    if price and plan["training_usd"] < PLAN["min_training_usd"]:
        lines.append("WARNING: almost nothing is left for training at this price; the dry run will stop and say why.")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m oa_rlaif.cost")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("price", help="find and record the price of this machine")
    p.add_argument("--platform", default="lambda", choices=["lambda", "coreweave", "vast"])
    p.add_argument("--yes", action="store_true", help="do not ask to continue")
    sub.add_parser("status", help="price, spend and the plan in dollars")
    args = parser.parse_args(argv)
    meter = Meter()
    if args.cmd == "status":
        meter.observe()
        print("\n".join(budget_summary(meter)))
        return 0
    machine_gpus = gpus()
    found = discover(args.platform, meter, machine_gpus)
    price, source, from_api = found if found else (*ask(args.platform, machine_gpus), False)
    if price <= 0:
        raise SystemExit("[cost] the price must be above zero")
    ok, why = plausible(price, machine_gpus)
    if not ok and not from_api:
        confirm_implausible(price, why)
    elif not ok:
        print(f"[cost] note: the provider's price is outside the usual range ({why}); using the provider's price")
    first_time = meter.price is None
    meter.set_price(price, source, machine_gpus)
    set_env_price(price)
    print(f"[cost] {why}")
    print("\n".join(budget_summary(meter)))
    if first_time and not args.yes and os.environ.get("OA_YES") != "1" and sys.stdin.isatty():
        if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("[cost] stopped; nothing was installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
