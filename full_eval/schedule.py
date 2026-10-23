"""What runs, in which order, and what the money allows.

A method is a model, how it is asked, and thinking off or on:
  grpo, sft, all_merged, sdft, base   one decision per class (the earlier evaluation's prompt, full_eval/infer.py)
  all_merged-agent                    All merged + mini-SWE-agent: the om_agent/ harness of the local console, one
                                      agent episode per class (full_eval/agent_run.py)
  <method>-think                      the same with thinking on (the models were trained with thinking off)

The default run (DEFAULT_METHODS) is the five models with thinking off and All merged + mini-SWE-agent with thinking
on; FULL_EVAL_METHODS (tags, or 'all' for the 12) and FULL_EVAL_THINKING (off, on, both) change it.

A unit is (method, task): one method aligning the classes of one task (FULL_EVAL_SCOPE, full_eval/tasks.py). A unit
either runs to the end or does not start, so every reported unit is complete. Units run in this priority order and
each one starts only if its projected cost fits the money left (FULL_EVAL_BUDGET_USD, default $425, minus a reserve
for scoring and the upload):

  1. the thinking-off methods, one after the other, each on every task (small tasks, then SNOMED CT -> ICD-11, then
     Bio-ML), so each model loads once
  2. the thinking-on methods in the same order

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
DEFAULT_METHODS = MODELS + [AGENT + "-think"]  # five models, thinking off; the agent, thinking on
REPORT_ORDER = ["base", "sft", "sdft", "grpo", "all_merged", AGENT]  # row order of the tables (thinking off; then on)
LABELS = {"base": "Gemma 4 E4B (base)", "sft": "+ SFT", "sdft": "+ OPSD", "grpo": "+ OPSD → GRPO (RLAIF)",
          "all_merged": "All merged (OPSD → GRPO + SFT)", AGENT: "All merged + mini-SWE-agent",
          # dm_harness/ (branch dm_harness): the decision model alone, and the decision model controlling the agent
          "decider": "pplx-decider-v1.2 alone",
          "jah-agent": "JevAgentHarness: pplx-decider routes All merged + mini-SWE-agent, with the NONE/invalid guard",
          "jah-fit-agent": "JevAgentHarness with thresholds fitted on a calibration sample (routing, guard, cascade)"}
SMALL_ORDER = ["anatomy__mouse-human", "dh", "dh_multiling", "biodiv", "complex_om__cree-swo", "cyber__iotsec-uco",
               "icd__who-icd10-to-icd11", "multifarm", "biodiv2026"]
GIANT_ORDER = ["icd__snomed-to-icd11", "bioml__NCIT-DOID", "bioml__SNOMED-FMA", "bioml__SNOMED-NCIT"]
# decisions/s on all GPUs until measured, per kind of method (FULL_EVAL_RATE_<KIND>, e.g. FULL_EVAL_RATE_AGENT_THINK)
DEFAULT_RATES = {"decide": 120.0, "decide-think": 20.0, "agent": 25.0, "agent-think": 8.0}
DEFAULT_LOAD_S = float(os.environ.get("FULL_EVAL_LOAD_S", "150"))  # model load + warm-up, until measured
DEFAULT_BUDGET_USD = "425"  # the default run on 4 x B300: ~15 h, ~$363 at $24/h (README: The money)


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
    """DEFAULT_METHODS, or FULL_EVAL_METHODS (comma-separated tags, or 'all' for the 12); FULL_EVAL_THINKING (off, on or
    both; default both) narrows either."""
    chosen = [m.strip() for m in os.environ.get("FULL_EVAL_METHODS", os.environ.get("FULL_EVAL_MODELS", "")).split(",")
              if m.strip()]
    chosen = METHODS if chosen == ["all"] else chosen or DEFAULT_METHODS
    thinking = os.environ.get("FULL_EVAL_THINKING", "both")
    keep = {"off": (False,), "on": (True,)}.get(thinking, (False, True))
    return [m for m in METHODS if m in chosen and method(m).think in keep]


def selected_tasks() -> list[str] | None:
    chosen = [s.strip() for s in os.environ.get("FULL_EVAL_TASKS", "").split(",") if s.strip()]
    return chosen or None


def units(methods: list[str] | None = None, slugs: list[str] | None = None) -> list[tuple[str, str]]:
    """(method, task) in priority order: the thinking-off methods, each on every task (small tasks, then SNOMED CT ->
    ICD-11, then Bio-ML), then the thinking-on methods in the same order."""
    methods = methods or selected_methods()
    present = [t.slug for t in task_order(slugs or selected_tasks())]
    small = [s for s in present if s not in GIANT]
    snomed = [s for s in present if s == "icd__snomed-to-icd11"]
    bioml = [s for s in present if s in GIANT and s != "icd__snomed-to-icd11"]
    order = []
    for think in (False, True):
        order += [(m, s) for m in methods if method(m).think == think for s in small + snomed + bioml]
    return order


# ---- sizes and speed ------------------------------------------------------------------------------------------------
# Classes each task runs, per FULL_EVAL_SCOPE, until its episodes are built (then the manifest's exact count is used).
# reference: counted from the raw data with full_eval.build_episodes.sources_to_align on 2026-10-22; SNOMED CT ->
# ICD-11 is the complete map of ICD's scope (RF2 US 2026-03) and ICD-10 -> ICD-11 every WHO code.
ESTIMATES = {
    "reference": {
        "anatomy__mouse-human": 1_497, "dh": 157, "dh_multiling": 98, "biodiv": 828, "complex_om__cree-swo": 15,
        "cyber__iotsec-uco": 35, "icd__who-icd10-to-icd11": 12_597, "multifarm": 35_037, "biodiv2026": 80_825,
        "icd__snomed-to-icd11": 141_182, "bioml__NCIT-DOID": 3_792, "bioml__SNOMED-FMA": 3_776,
        "bioml__SNOMED-NCIT": 20_122},
    "all": {"icd__snomed-to-icd11": 141_182, "icd__who-icd10-to-icd11": 12_597, "biodiv2026": 124_563},
}


def decisions(slug: str) -> tuple[int, bool]:
    """(decisions, exact): from the built episodes when ready, else ESTIMATES, else the ontology profiles' size."""
    from .tasks import eval_scope

    manifest = root() / "episodes" / slug / "manifest.json"
    if (manifest.parent / "READY").exists():
        return json.loads(manifest.read_text())["episodes"], True
    estimate = ESTIMATES[eval_scope()].get(slug)
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
        self.cap = float(cap if cap is not None else os.environ.get("FULL_EVAL_BUDGET_USD", DEFAULT_BUDGET_USD))
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


# results/<task>/<method folder>/: one folder per dataset task, one per method inside it, numbered in table order
FOLDERS = {"base": "1-base", "sft": "2-sft", "sdft": "3-opsd", "grpo": "4-opsd-grpo", "all_merged": "5-all-merged",
           AGENT: "6-all-merged-agent", "decider": "7-pplx-decider", "jah-agent": "8-jah-all-merged-agent",
           "jah-fit-agent": "9-jah-all-merged-agent-fitted"}


def folder_name(tag: str) -> str:
    """The method's results folder: e.g. 4-opsd-grpo, 6-all-merged-agent-thinking (thinking on)."""
    m = method(tag)
    return FOLDERS[m.base] + ("-thinking" if m.think else "")


def results_dir(model: str, slug: str) -> Path:
    """results/<task>/<method folder>/, e.g. results/anatomy__mouse-human/4-opsd-grpo/."""
    return root() / "results" / slug / folder_name(model)


def bar(done: int, total: int, width: int = 24) -> str:
    """A text progress bar: [#########---------------]  37.5%."""
    share = min(1.0, done / total) if total else 0.0
    filled = int(round(share * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {share * 100:5.1f}%"


def clock(seconds: float | None) -> str:
    """Seconds as h:mm:ss (or '?')."""
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "?"
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def progress_text(label: str, done: int, total: int, rate: float, unit: str = "decisions") -> str:
    """One progress line: bar, counts, speed and the time left at that speed."""
    left = (total - done) / rate if rate > 0 else None
    return f"{label} {bar(done, total)} {done:,}/{total:,} {unit} · {rate:.1f}/s · ETA {clock(left)}"


def unit_done(model: str, slug: str) -> bool:
    return (results_dir(model, slug) / "scores.json").exists()


def replies_complete(model: str, slug: str) -> bool:
    folder = root() / "episodes" / slug
    if not (folder / "READY").exists():
        return False
    shards = json.loads((folder / "manifest.json").read_text())["shards"]
    replies = root() / "replies" / model / slug
    return all((replies / name.replace(".jsonl.gz", ".replies.jsonl.gz")).exists() for name in shards)
