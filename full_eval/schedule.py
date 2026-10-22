"""What runs, in which order, and what the money allows.

A method is a model, how it is asked, and thinking off or on:
  grpo, sft, all_merged, sdft, base   one decision per class (the earlier evaluation's prompt, full_eval/infer.py)
  all_merged-agent                    All merged + mini-SWE-agent: the om_agent/ harness of the local console, one
                                      agent episode per class (full_eval/agent_run.py)
  <method>-think                      the same with thinking on (the models were trained with thinking off)

A unit is (method, task): one method aligning every class of one task. A unit either runs to the end or does not
start, so every alignment that is reported is complete. Units run in this priority order and each one starts only if
its projected cost fits the money left (FULL_EVAL_BUDGET_USD, default $100, minus a reserve for scoring and the
upload):

  1. thinking off, all six methods: every task except Bio-ML (small tasks first, then SNOMED CT -> ICD-11)
  2. the same with thinking on
  3. Bio-ML (three whole-ontology pairs): thinking off, then on

The projection is decisions / measured decisions-per-second of that kind of method (one decision or agent, thinking
off or on) + load time, times this machine's price. Spend is the machine's billed time since it started
(oa_rlaif.cost.Meter), so setup and downloads count too.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from oa_rlaif.paths import data_dir, out_dir, relative_to_raw

from .tasks import GIANT, Task, tasks

MODELS = ["grpo", "sft", "all_merged", "sdft", "base"]  # run order of the one-decision methods
AGENT = "all_merged-agent"  # All merged + mini-SWE-agent (om_agent/)
METHODS_OFF = MODELS + [AGENT]
METHODS = METHODS_OFF + [m + "-think" for m in METHODS_OFF]
REPORT_ORDER = ["base", "sft", "sdft", "grpo", "all_merged", AGENT]  # row order of the tables (thinking off; then on)
LABELS = {"base": "Gemma 4 E4B (base)", "sft": "+ SFT", "sdft": "+ OPSD", "grpo": "+ OPSD → GRPO (RLAIF)",
          "all_merged": "All merged (OPSD → GRPO + SFT)", AGENT: "All merged + mini-SWE-agent"}
SMALL_ORDER = ["anatomy__mouse-human", "dh", "dh_multiling", "biodiv", "complex_om__cree-swo", "cyber__iotsec-uco",
               "icd__who-icd10-to-icd11", "multifarm", "biodiv2026"]
GIANT_ORDER = ["icd__snomed-to-icd11", "bioml__NCIT-DOID", "bioml__SNOMED-FMA", "bioml__SNOMED-NCIT"]
# decisions/s on all GPUs until measured, per kind of method (FULL_EVAL_RATE_<KIND>, e.g. FULL_EVAL_RATE_AGENT_THINK)
DEFAULT_RATES = {"decide": 120.0, "decide-think": 20.0, "agent": 25.0, "agent-think": 8.0}
DEFAULT_LOAD_S = float(os.environ.get("FULL_EVAL_LOAD_S", "150"))  # model load + warm-up, until measured


@dataclass(frozen=True)
class Method:
    tag: str
    model: str     # the weights (full_eval.models)
    think: bool
    agent: bool

    @property
    def kind(self) -> str:
        return ("agent" if self.agent else "decide") + ("-think" if self.think else "")

    @property
    def base(self) -> str:
        """The tag without thinking: the row of the tables."""
        return self.tag.removesuffix("-think")


def method(tag: str) -> Method:
    think = tag.endswith("-think")
    base = tag.removesuffix("-think")
    agent = base.endswith("-agent")
    return Method(tag, base.removesuffix("-agent"), think, agent)


def label(tag: str) -> str:
    m = method(tag)
    return LABELS.get(m.base, m.base) + (", thinking on" if m.think else "")


def root() -> Path:
    return out_dir() / "full_eval"


def task_order(slugs: list[str] | None = None) -> list[Task]:
    """Tasks in the order their episodes are needed (small ones first, then the giant ones)."""
    by_slug = {t.slug: t for t in tasks()}
    order = [s for s in SMALL_ORDER + GIANT_ORDER if s in by_slug] + [s for s in by_slug if s not in SMALL_ORDER + GIANT_ORDER]
    return [by_slug[s] for s in order if not slugs or s in slugs]


def selected_methods() -> list[str]:
    """FULL_EVAL_METHODS (comma-separated tags) and FULL_EVAL_THINKING (off, on or both; default both) narrow the run."""
    chosen = [m.strip() for m in os.environ.get("FULL_EVAL_METHODS", os.environ.get("FULL_EVAL_MODELS", "")).split(",")
              if m.strip()]
    thinking = os.environ.get("FULL_EVAL_THINKING", "both")
    keep = {"off": (False,), "on": (True,)}.get(thinking, (False, True))
    return [m for m in METHODS if (not chosen or m in chosen) and method(m).think in keep]


def selected_tasks() -> list[str] | None:
    chosen = [s.strip() for s in os.environ.get("FULL_EVAL_TASKS", "").split(",") if s.strip()]
    return chosen or None


def units(methods: list[str] | None = None, slugs: list[str] | None = None) -> list[tuple[str, str]]:
    """(method, task) in priority order: thinking off on every task but Bio-ML (small tasks, then SNOMED CT ->
    ICD-11), the same with thinking on, then Bio-ML off and on."""
    methods = methods or selected_methods()
    present = [t.slug for t in task_order(slugs or selected_tasks())]
    small = [s for s in present if s not in GIANT]
    snomed = [s for s in present if s == "icd__snomed-to-icd11"]
    bioml = [s for s in present if s in GIANT and s != "icd__snomed-to-icd11"]
    order = []
    for think in (False, True):
        group = [m for m in methods if method(m).think == think]
        order += [(m, s) for m in group for s in small]
        order += [(m, s) for m in group for s in snomed]
    for think in (False, True):
        order += [(m, s) for m in methods if method(m).think == think for s in bioml]
    return order


# ---- sizes and speed ------------------------------------------------------------------------------------------------
def decisions(slug: str) -> tuple[int, bool]:
    """(decisions, exact): from the built episodes when ready, else an estimate from the ontology profiles."""
    manifest = root() / "episodes" / slug / "manifest.json"
    if (manifest.parent / "READY").exists():
        return json.loads(manifest.read_text())["episodes"], True
    estimate = {"icd__snomed-to-icd11": 141_182, "icd__who-icd10-to-icd11": 12_597, "biodiv2026": 124_563}.get(slug)
    if estimate is None:
        profiles = json.loads((data_dir() / "ontology_profiles.json").read_text())
        task = next(t for t in tasks() if t.slug == slug)
        estimate = sum(profiles.get(relative_to_raw(s.source), {}).get("classes", 0)
                       + profiles.get(relative_to_raw(s.source), {}).get("properties", 0) for s in task.specs)
    return estimate, False


@dataclass
class Speed:
    """Measured throughput of finished units (decisions per second on all GPUs) per kind of method, and load times."""
    path: Path = field(default_factory=lambda: root() / "speed.json")

    def _read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {"units": [], "loads": []}

    def record_unit(self, unit: str, n: int, seconds: float) -> None:
        data = self._read()
        data["units"].append({"unit": unit, "kind": method(unit.split("/", 1)[0]).kind, "decisions": n,
                              "seconds": round(seconds, 1), "at": time.time()})
        self.path.write_text(json.dumps(data, indent=1))

    def record_load(self, seconds: float) -> None:
        data = self._read()
        data["loads"].append(round(seconds, 1))
        self.path.write_text(json.dumps(data, indent=1))

    def rate(self, kind: str = "decide") -> tuple[float, str]:
        done = [u for u in self._read()["units"] if u.get("kind", "decide") == kind
                and u["decisions"] >= (500 if kind.startswith("agent") else 2000) and u["seconds"] > 0]
        if not done:
            default = float(os.environ.get(f"FULL_EVAL_RATE_{kind.upper().replace('-', '_')}", DEFAULT_RATES[kind]))
            return default, f"assumed {default:g}/s until a unit of this kind finishes"
        total = sum(u["decisions"] for u in done) / sum(u["seconds"] for u in done)
        return total, f"measured on {len(done)} unit(s)"

    def load_seconds(self) -> float:
        loads = self._read()["loads"]
        return statistics.median(loads) if loads else DEFAULT_LOAD_S


# ---- money -----------------------------------------------------------------------------------------------------------
class Budget:
    def __init__(self, cap: float | None = None):
        from oa_rlaif.cost import Meter

        self.meter = Meter()
        self.cap = float(cap if cap is not None else os.environ.get("FULL_EVAL_BUDGET_USD", "100"))
        self.reserve = float(os.environ.get("FULL_EVAL_RESERVE_USD", "3"))
        self.fake = os.environ.get("FULL_EVAL_FAKE") == "lexical"  # Mac rehearsal: $1/h from the rehearsal's start
        self.started = time.time()

    @property
    def price(self) -> float:
        if self.fake:
            return 1.0
        price = self.meter.price
        if not price:
            raise SystemExit("no price recorded for this machine: run scripts/price.sh vast first")
        return price

    def spent(self) -> float:
        if self.fake:
            return (time.time() - self.started) / 3600
        self.meter.observe()
        return self.meter.spent()

    def left(self) -> float:
        return self.cap - self.reserve - self.spent()

    def dollars(self, seconds: float) -> float:
        return self.price * seconds / 3600


def project(unit_list: list[tuple[str, str]], budget: Budget, speed: Speed, loaded: str | None = None) -> list[dict]:
    """Greedy plan in priority order: each unit with its projected seconds and dollars, and whether it fits."""
    left = budget.left()
    current = loaded
    plan = []
    for tag, slug in unit_list:
        n, exact = decisions(slug)
        rate, basis = speed.rate(method(tag).kind)
        seconds = n / rate + (speed.load_seconds() if tag != current else 0)
        cost = budget.dollars(seconds)
        fits = cost <= left
        if fits:
            left -= cost
            current = tag
        plan.append({"model": tag, "task": slug, "decisions": n, "exact": exact, "seconds": round(seconds),
                     "usd": round(cost, 2), "fits": fits, "rate": round(rate, 1), "rate_basis": basis})
    return plan


def unit_done(model: str, slug: str) -> bool:
    return (root() / "results" / model / slug / "scores.json").exists()


def replies_complete(model: str, slug: str) -> bool:
    folder = root() / "episodes" / slug
    if not (folder / "READY").exists():
        return False
    shards = json.loads((folder / "manifest.json").read_text())["shards"]
    replies = root() / "replies" / model / slug
    return all((replies / name.replace(".jsonl.gz", ".replies.jsonl.gz")).exists() for name in shards)
