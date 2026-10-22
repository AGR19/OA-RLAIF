"""The hourly price of THIS machine for the full evaluation's money guard (the training run's price logic, without its
training plan): the Vast.ai API, else OA_GPU_USD_PER_HOUR in the shell, else asked and checked against the GPUs.

    python -m full_eval.price [--platform vast] [--yes]
"""

from __future__ import annotations

import argparse
import os
import sys

from oa_rlaif.cost import Meter, ask, confirm_implausible, discover, gpus, plausible, set_env_price


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="full_eval.price")
    parser.add_argument("--platform", default="vast", choices=["vast", "lambda", "coreweave"])
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    meter, machine = Meter(), gpus()
    found = discover(args.platform, meter, machine)
    price, source, from_api = found if found else (*ask(args.platform, machine), False)
    ok, why = plausible(price, machine)
    if not ok and not from_api:
        confirm_implausible(price, why)
    meter.set_price(price, source, machine)
    set_env_price(price)
    cap = float(os.environ.get("FULL_EVAL_BUDGET_USD", "100"))
    print(f"[price] ${price:.2f}/h ({source}); {len(machine)} GPU(s): {', '.join(sorted({n for n, _ in machine})) or '?'}")
    print(f"[price] cap ${cap:.0f} = {cap / price:.1f} h of this machine, setup and downloads included "
          f"(spent so far ${meter.spent():.2f})")
    if not args.yes and sys.stdin.isatty() and os.environ.get("OA_YES") != "1":
        if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("[price] stopped; nothing was installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
