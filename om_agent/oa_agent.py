#!/usr/bin/env python3
"""oa_agent.py: a Claude-Code-style console for the OA-RLAIF mini-SWE-agent ontology matcher (runs locally).

    ./oa_agent.sh

On launch it shows the model (name, where it is loaded from, format, thinking) and the agent's skills, then asks for
the two ontologies (type a path or drag the file into the window; a file from elsewhere is copied into ontologies/)
and for the output folder. Nothing runs until you say so: it asks before it starts, and the agent asks before every
command it runs (answer 2 once to allow the rest of the session). For each class you see the agent turn by turn: the
exploration steps (om profile, om show, om candidates), the model's thinking, its THOUGHT and command, and the
decision. A reference alignment is optional: with one, every save scores the result with the Top-k evaluation
(k = 1, 3, 5): Precision@k, Recall@k and F-measure@k as in DRAL-OA's clf.py (precision_at_k / recall_at_k: the source
classes that have a partner in the reference; the model's own choice first, then its other candidates), in results.md
and topk.svg. The agent never sees the
reference. /score (the om-score skill) scores this run or any run folders, side by side. The output folder holds alignment.rdf, decisions.tsv (every
decision with labels), decisions.jsonl, results.md (with a reference) and tasks/ (each class's full trajectory).
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import shlex
import shutil
import statistics
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_matching_ai as tm  # noqa: E402  (sets the OA-RLAIF import path)

tm.own_folders()  # everything the console writes stays in this folder; before any OA-RLAIF code reads the paths

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console(highlight=False)
SKILLS = [
    ("om profile", "both ontologies' modelling assumptions (hierarchy, part-of, conventions)"),
    ("om show <IRI>", "the source class card: name, synonyms, parents, children, definition"),
    ("om candidates <IRI> --k N", "the numbered candidate target classes C1..CN with their cards"),
    ("om submit C#[,C#] | NONE --relation R --confidence X", "record the decision; ends the class"),
    ("om-score <run folder(s)> [--reference FILE]", "Top-k evaluation (k = 1, 3, 5): Precision@k, Recall@k, "
                                                     "F-measure@k and the chart; /score runs it"),
]
HELP = [("/run [N|all]", "align the next N classes not yet decided (default 1); Ctrl-C stops after a class"),
        ("/align <label or IRI>", "run one class and show it (not recorded)"),
        ("/status", "progress, counts, speed, time left and the expected finish"),
        ("/save", "write alignment.rdf, decisions.tsv, summary.json and, with a reference, the Top-k scores (also after every /run)"),
        ("/score [run folders]", "the om-score skill: Top-k evaluation (k = 1, 3, 5) for this run, or for the given run folders side by side"),
        ("/open", "choose other ontologies or another output folder"),
        ("/thinking on|off", "the model's thinking (on now; the model was trained with it off)"),
        ("/permissions", "switch between asking before every command and allowing all"),
        ("/model", "the model and where it is loaded from"),
        ("/skills", "the tools the agent can use"),
        ("/compact", "switch between the full transcript and one line per class"),
        ("/quit", "leave (decisions are kept)")]


class Stop(Exception):
    """The user declined a command: stop the run."""


# ---- time estimates ---------------------------------------------------------------------------------------------------
SPEED_FILE = HERE / "cache" / "speed.json"  # seconds per class from earlier runs on this machine, by thinking on/off
DEFAULT_SECONDS = 8.0  # rough seconds per class on this Mac (M2 Pro, model loaded) from the first test runs


def saved_speeds() -> dict:
    try:
        return json.loads(SPEED_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_speed(think: bool, seconds: float) -> None:
    speeds = saved_speeds()
    key = "thinking on" if think else "thinking off"
    speeds[key] = (speeds.get(key, []) + [round(seconds, 2)])[-500:]
    SPEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPEED_FILE.write_text(json.dumps(speeds))


def duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


def ask(prompt: str) -> str:
    try:
        return console.input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]bye[/dim]")
        sys.exit(0)


# ---- model information ------------------------------------------------------------------------------------------------
def ollama(path: str, body: dict | None = None):
    base = tm.OLLAMA.removesuffix("/v1")
    request = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request, timeout=10).read())


def model_info(name: str) -> dict:
    info = {"name": name, "server": tm.OLLAMA.removesuffix("/v1")}
    try:
        info["version"] = ollama("/api/version")["version"]
        shown = ollama("/api/show", {"model": name})
        details = shown.get("details", {})
        info.update(parameters=details.get("parameter_size"), quantization=details.get("quantization_level"),
                    settings=shown.get("parameters", "").strip())
        info["context"] = next((v for k, v in (shown.get("model_info") or {}).items() if k.endswith("context_length")), None)
        found = re.search(r"num_ctx\s+(\d+)", info["settings"] or "")
        info["num_ctx"] = int(found.group(1)) if found else None
    except Exception as error:  # noqa: BLE001
        info["error"] = f"{type(error).__name__}: {error}"
    weights = HERE / "model" / "all_merged" / "model.safetensors"
    info["weights"] = f"{weights.parent} ({weights.stat().st_size / 1e9:.1f} GB bf16 safetensors)" if weights.exists() else "missing"
    info["store"] = str(HERE / "ollama-models")
    return info


def banner(info: dict, think: bool, auto: bool) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()
    grid.add_row("Model", f"[bold]{info['name']}[/bold]  ·  Gemma 4 E4B + OPSD + GRPO (RLAIF) + SFT, all merged")
    if "error" in info:
        grid.add_row("Server", f"[red]{info['server']} not ready: {info['error']}[/red]  (start with ./oa_agent.sh)")
    else:
        grid.add_row("Served by", f"Ollama {info['version']} at {info['server']} (local only)")
        grid.add_row("Loaded from", f"{info['weights']}\nimported into {info['store']} as "
                                    f"{info.get('quantization') or '?'}, {info.get('parameters') or '?'} parameters")
        window = f"{info['num_ctx']:,}" if info.get("num_ctx") else "?"
        grid.add_row("Format", f"Gemma 4 chat, temperature 0, {window}-token window per call, stop <turn|>")
    grid.add_row("Thinking", "[bold]on[/bold] (the model was trained and evaluated with thinking off; /thinking off "
                             "matches training)" if think else "off, as in training (/thinking on to turn it on)")
    grid.add_row("Agent", "mini-SWE-agent 2.4.6, local environment (no Docker), step limit 6 model calls per class")
    grid.add_row("Permissions", "allow all commands (/permissions to ask again)" if auto else
                 "ask before every command (answer 2 to allow the rest of the session)")
    console.print(Panel(grid, title="[bold] ✻ OA-RLAIF ontology-matching agent [/bold]", border_style="magenta",
                        padding=(1, 2)))
    skills()


def skills() -> None:
    table = Table(title="Skills: tools in the agent's shell", title_justify="left", show_header=False, box=None,
                  padding=(0, 2))
    table.add_column(style="green", no_wrap=True)
    table.add_column(style="dim")
    for command, what in SKILLS:
        table.add_row(command, what)
    console.print(table)
    console.print()


# ---- live transcript ----------------------------------------------------------------------------------------------------
def observation_text(content: str) -> str:
    found = re.search(r"<output>\n?(.*?)\n?</output>", content, re.S)
    return (found.group(1) if found else content).strip()


class Transcript:
    """Renders an agent's messages as they arrive, Claude-Code style."""

    def __init__(self, model_name: str, compact: bool, think: bool, window: int | None):
        self.model, self.compact, self.think, self.window = model_name, compact, think, window
        self.last_command = ""

    def __call__(self, message: dict) -> None:
        if self.compact:
            return
        role, content = message.get("role"), str(message.get("content") or "")
        extra = message.get("extra") or {}
        if role == "system":
            return
        if role == "assistant":
            command = ((extra.get("actions") or [{}])[0]).get("command", "")
            self.last_command = command
            if extra.get("scripted"):
                console.print(f"  [cyan]⏺[/cyan] {command} [dim](exploration step)[/dim]")
                return
            self.thinking(str(message.get("reasoning_content") or ""), content)
            thought = content.split("```", 1)[0].strip().removeprefix("THOUGHT:").strip()
            usage = ((extra.get("response") or {}).get("usage")) or {}
            tokens = (f"  [dim]({usage['prompt_tokens']:,} tokens in → {usage['completion_tokens']:,} out)[/dim]"
                      if usage.get("prompt_tokens") else "")
            console.print(f"  [magenta]✻ {self.model}[/magenta]  {thought}{tokens}")
            if self.window and usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0) > 0.9 * self.window:
                console.print(f"    [yellow]⚠ near the {self.window:,}-token window: raise num_ctx in the Modelfile[/yellow]")
            if command:
                console.print(f"  [green]⏺[/green] {command.replace('echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && ', '')}")
            return
        if role == "user" and "<returncode>" in content:
            console.print("    [dim]⎿ " + self.summarize(observation_text(content)) + "[/dim]")
        elif role == "user" and content.startswith("Format error"):
            response = (extra.get("response") or {}) if isinstance(extra.get("response"), dict) else {}
            reasoning = (((response.get("choices") or [{}])[0].get("message") or {}).get("reasoning_content")) or ""
            self.thinking(reasoning, "")
            console.print(f"    [yellow]⚠ {content.splitlines()[0][:140]}[/yellow]")
        elif role == "user":
            console.print(f"  [dim]task: {content.strip()[:160]}[/dim]")

    def thinking(self, reasoning: str, content: str) -> None:
        if not self.think:
            return
        reasoning, content = reasoning.strip(), content.strip()
        if not reasoning:
            console.print("  [dim]✻ Thinking: none (answered directly)[/dim]")
            return
        if content and content in reasoning:  # the answer was written inside the thinking channel
            reasoning = reasoning[: reasoning.rfind(content)].strip()
            console.print("  [dim]✻ The model wrote its answer inside the thinking channel (taken from there)"
                          + ("; no other thinking" if not reasoning else "") + "[/dim]")
            if not reasoning:
                return
        console.print("  [dim]✻ Thinking…[/dim]")
        for line in reasoning.splitlines():
            console.print(f"    [dim italic]│ {line}[/dim italic]", soft_wrap=True)

    def summarize(self, text: str) -> str:
        if self.last_command.startswith("om candidates"):
            names = re.findall(r"\[C(\d+)\]\nIRI: \S+\nname: ([^\n]+)", text)
            shown = " · ".join(f"C{i} {n}" for i, n in names[:4])
            return f"{len(names)} candidates: {shown}" + (f" · (+{len(names) - 4} more)" if len(names) > 4 else "")
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in text or text.startswith("source_iri"):
            return "submitted"
        lines = [line for line in text.splitlines() if line.strip()]
        head = " | ".join(lines[:2])[:170]
        return head + (f"  (+{len(lines) - 2} lines)" if len(lines) > 2 else "")


# ---- choosing the ontologies and the output folder ----------------------------------------------------------------------
def as_path(text: str) -> Path | None:
    """A typed or dragged-in path: quotes, backslash-escaped spaces, ~ and file:// URLs are all fine."""
    if not text:
        return None
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = [text]
    raw = " ".join(parts) if parts else text
    return tm.path_of(raw).expanduser().resolve() if raw.startswith("file:") else Path(raw).expanduser().resolve()


def upload(path: Path) -> Path:
    """A file from outside this folder is copied into ontologies/ (never over a different file of the same name)."""
    if HERE in path.parents:
        return path
    folder = HERE / "ontologies"
    folder.mkdir(exist_ok=True)
    dest, n = folder / path.name, 2
    while dest.exists() and not filecmp.cmp(dest, path, shallow=False):
        dest, n = folder / f"{path.stem}-{n}{path.suffix}", n + 1
    if not dest.exists():
        shutil.copy2(path, dest)
        console.print(f"  [green]uploaded[/green] → {dest}")
    else:
        console.print(f"  [dim]already in {dest.parent} (same file)[/dim]")
    return dest


def choose_ontology(role: str, default: Path) -> Path:
    while True:
        text = ask(f"[bold]{role} ontology[/bold] [dim](type a path or drag the file here; Enter = {default.name})[/dim]\n❯ ")
        path = as_path(text) or default
        if not path.is_file():
            console.print(f"  [red]no file at {path}[/red]")
            continue
        if path.suffix.lower() not in (".owl", ".rdf", ".ttl", ".xml", ".n3", ".nt", ".owx", ".ofn", ".omn"):
            if ask(f"  [yellow]{path.suffix or 'no extension'} is not a usual ontology format; use it anyway? [y/N][/yellow] ").lower() != "y":
                continue
        return upload(path)


def choose_reference(source_profile, target_profile) -> Path | None:
    """The reference alignment (optional): used only to score the result, never shown to the agent."""
    from oa_rlaif.ontology.alignment_io import orient, read_alignment

    while True:
        text = ask("[bold]Reference alignment[/bold] (optional) [dim](type a path or drag the file here; Enter = skip)[/dim]\n❯ ")
        if not text:
            console.print("  [dim]no reference: the result is not scored[/dim]")
            return None
        path = as_path(text)
        if path is None or not path.is_file():
            console.print(f"  [red]no file at {path}[/red]")
            continue
        try:
            cells = [a for a in read_alignment(path) if a.atomic]
        except Exception as error:  # noqa: BLE001
            console.print(f"  [red]not an alignment file ({type(error).__name__}: {error})[/red]")
            continue
        resolved = sum(orient(a, source_profile, target_profile).status != "unresolved_endpoint" for a in cells)
        if not resolved:
            console.print(f"  [red]none of its {len(cells):,} correspondences links these two ontologies[/red]")
            continue
        console.print(f"  [green]reference[/green]: {len(cells):,} correspondences, {resolved:,} between these ontologies")
        return upload(path)


def choose_output(default: Path, source: Path, target: Path) -> Path:
    while True:
        text = ask(f"[bold]Output folder[/bold] [dim](Enter = {default})[/dim]\n❯ ")
        out = as_path(text) or default
        if out.exists() and not out.is_dir():
            console.print(f"  [red]{out} is a file, not a folder[/red]")
            continue
        run_file = out / "run.json"
        if run_file.exists():
            previous = json.loads(run_file.read_text())
            if (previous.get("source"), previous.get("target")) != (str(source), str(target)):
                console.print(f"  [red]{out} holds a run of other ontologies ({Path(previous['source']).name} → "
                              f"{Path(previous['target']).name}); choose another folder (nothing is deleted)[/red]")
                continue
        decided = 0
        if (out / "decisions.jsonl").exists():
            decided = len({json.loads(line)["source"] for line in (out / "decisions.jsonl").read_text().splitlines()
                           if json.loads(line)["decision"] in ("match", "NONE")})
        if decided and ask(f"  {decided:,} classes are already decided there. Continue that run? [Y/n] ").lower() == "n":
            console.print("  [dim]choose another folder (nothing is deleted)[/dim]")
            continue
        out.mkdir(parents=True, exist_ok=True)
        if not run_file.exists():
            run_file.write_text(json.dumps({"source": str(source), "target": str(target)}, indent=1))
        return out


# ---- the console --------------------------------------------------------------------------------------------------------
class AgentConsole:
    def __init__(self, args):
        self.args = args
        self.think, self.auto, self.compact = not args.no_think, False, args.compact
        self.info = model_info(args.model)
        self.window = self.info.get("num_ctx")
        self.episodes: list[dict] = []
        self.waited = 0.0

    # -- launch: ontologies, output folder ------------------------------------------------------------------------------
    def open(self) -> None:
        while True:
            source = choose_ontology("Source", HERE / "ontologies" / "mouse.owl")
            target = choose_ontology("Target", HERE / "ontologies" / "human.owl")
            pair_id = ("anatomy/mouse-human" if (source.stem, target.stem) == ("mouse", "human")
                       else f"{source.stem}-{target.stem}")
            try:
                with console.status("Reading both ontologies and preparing one task per class (candidates, cards, "
                                    "profiles)…"):
                    episodes, self.source, self.target = tm.build_episodes(source, target, self.args.k, pair_id)
            except Exception as error:  # noqa: BLE001
                console.print(f"  [red]could not read the ontologies: {type(error).__name__}: {error}[/red]")
                continue
            if not episodes:
                console.print("  [red]no classes found in the source ontology[/red]")
                continue
            break
        self.episodes, self.source_path, self.target_path = episodes, source, target
        self.reference = choose_reference(self.source, self.target)
        self.out = choose_output(HERE / "output" / f"{source.stem}-{target.stem}", source, target)
        run = json.loads((self.out / "run.json").read_text())
        if self.reference and run.get("reference") != str(self.reference):
            run["reference"] = str(self.reference)
            (self.out / "run.json").write_text(json.dumps(run, indent=1))
        self.log = self.out / "decisions.jsonl"
        self.records = {}
        if self.log.exists():
            for line in self.log.read_text().splitlines():
                record = json.loads(line)
                if record["decision"] in ("match", "NONE"):
                    self.records[record["source"]] = record
        self.failed: set[str] = set()
        self.times: list[float] = []   # every class this session (the session's total)
        self.speeds: list[float] = []  # the same without the session's first class (model load): for estimates
        self.matcher = tm.Matcher(self.args.model, self.out, 1, "local", think=self.think)
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", no_wrap=True)
        grid.add_column()
        grid.add_row("Source", f"{source}  ({len(episodes):,} classes to align)")
        grid.add_row("Target", f"{target}  ({len(tm.own_classes(self.target)):,} classes)")
        grid.add_row("Per class", f"{self.args.k} candidates; matches with confidence ≥ {self.args.threshold} go into "
                                  "alignment.rdf (1:1)")
        grid.add_row("Reference", f"{self.reference}  (scoring only: Top-k evaluation, k = 1, 3, 5)"
                     if self.reference else "none (the result is not scored)")
        grid.add_row("Output", f"{self.out}")
        grid.add_row("Progress", f"{len(self.records):,} of {len(episodes):,} classes decided")
        left = len(episodes) - len(self.records)
        if left:
            grid.add_row("Estimate", self.eta(left, detail=True))
        console.print(Panel(grid, title="[bold]Task[/bold]", border_style="cyan", padding=(0, 2)))

    # -- time estimates -------------------------------------------------------------------------------------------------
    def per_class(self) -> tuple[float, str]:
        """Seconds per class (the median, so the first class, which also loads the model, does not skew it) and where
        the figure comes from: this session, else earlier runs on this machine, else the first tests."""
        if len(self.speeds) >= 3:
            return statistics.median(self.speeds[-50:]), f"median of the last {min(len(self.speeds), 50)} classes"
        saved = saved_speeds().get("thinking on" if self.think else "thinking off", [])
        if len(saved) >= 3:
            return statistics.median(saved[-200:]), f"median of {min(len(saved), 200)} earlier classes on this Mac"
        return DEFAULT_SECONDS, "rough figure from the first tests on this Mac; refined as classes finish"

    def eta(self, left: int, detail: bool = False, short: bool = False) -> str:
        per, basis = self.per_class()
        total = per * left
        finish = time.strftime("%a %H:%M", time.localtime(time.time() + total))
        if short:
            return f"about {duration(total)}, done around {finish}"
        text = f"about {duration(total)} for {left:,} classes ({per:.1f} s per class), done around {finish}"
        if not detail:
            return text
        notes = [basis]
        try:
            if not any(m.get("name", "").startswith(self.args.model) for m in ollama("/api/ps").get("models", [])):
                notes.append("plus about 30-50 s to load the model for the first class")
        except Exception:  # noqa: BLE001
            pass
        if not self.auto:
            notes.append("not counting the time you take to answer the prompts (answer 2 once to let it run alone)")
        return text + "\n" + "; ".join(notes)

    # -- permissions ----------------------------------------------------------------------------------------------------
    def approve(self, command: str) -> bool:
        if self.auto:
            return True
        asked = time.time()
        try:
            return self._ask_approval()
        finally:
            self.waited += time.time() - asked  # time spent answering is not counted in the speed

    def _ask_approval(self) -> bool:
        while True:
            answer = ask("    [bold]Run this command?[/bold]  [bold]1[/bold] yes  ·  [bold]2[/bold] yes, and don't ask "
                         "again this session  ·  [bold]3[/bold] no, stop  [dim](Enter = 1)[/dim] ❯ ").lower()
            if answer in ("", "1", "y", "yes"):
                return True
            if answer in ("2", "a"):
                self.auto = True
                console.print("    [dim]allowing all commands for the rest of the session (/permissions to ask again)[/dim]")
                return True
            if answer in ("3", "n", "no"):
                return False

    # -- running classes ------------------------------------------------------------------------------------------------
    def run_one(self, episode: dict, index: int, record_it: bool) -> dict:
        src = episode["source_iri"]
        console.print(f"\n[bold]● Class {index:,}/{len(self.episodes):,}[/bold]  {src}  "
                      f"[bold]“{tm.label(self.source, src)}”[/bold]")
        started, self.waited = time.time(), 0.0
        result = self.matcher.run_episode(episode, approve=self.approve,
                                          on_message=Transcript(self.args.model, self.compact, self.think, self.window))
        seconds = time.time() - started - self.waited
        if result["decision"] == "declined":
            console.print("  [yellow]declined; this class is not recorded[/yellow]")
            raise Stop
        if self.times:  # the session's first class also loads the model: not used for the speed
            self.speeds.append(seconds)
            save_speed(self.think, seconds)
        self.times.append(seconds)
        result["seconds"] = round(seconds, 1)
        if result["decision"] == "match":
            names = ", ".join(f"{t} “{tm.label(self.target, t)}”" for t in result["targets"])
            line = f"  [bold green]match[/bold green] → {names} ({result['relation']}, {result['confidence']})"
        elif result["decision"] == "NONE":
            line = "  [bold yellow]NONE[/bold yellow] (no candidate corresponds)"
        else:
            line = f"  [bold red]no submission[/bold red] ({result['exit']}; tried again on the next /run)"
        console.print(f"{line}  ·  {result['model_calls']} model call(s) · {seconds:.1f} s")
        if record_it:
            with self.log.open("a") as f:
                f.write(json.dumps(result) + "\n")
            if result["decision"] in ("match", "NONE"):
                self.records[src] = result
                self.failed.discard(src)
            else:
                self.failed.add(src)
            self.status(short=True)
        return result

    def run(self, how_many: str) -> None:
        todo = [e for e in self.episodes if e["source_iri"] not in self.records]
        if not todo:
            console.print("[green]every class is decided[/green]")
            self.save()
            return
        count = len(todo) if how_many == "all" else int(how_many or 1)
        console.print(f"[dim]aligning {min(count, len(todo)):,} of the {len(todo):,} classes not yet decided: "
                      f"{self.eta(min(count, len(todo)))}; Ctrl-C stops after the current class[/dim]")
        try:
            for episode in todo[:count]:
                self.run_one(episode, self.episodes.index(episode) + 1, record_it=True)
        except KeyboardInterrupt:
            console.print("\n[yellow]stopped; the current class was not recorded[/yellow]")
        except Stop:
            console.print("[yellow]run stopped[/yellow]")
        self.save()

    def align(self, query: str) -> None:
        q = query.strip().lower()
        hits = [e for e in self.episodes if e["source_iri"].lower() == q or e["source_iri"].lower().endswith("#" + q)]
        hits = hits or [e for e in self.episodes if tm.label(self.source, e["source_iri"]).lower() == q]
        hits = hits or [e for e in self.episodes if q in tm.label(self.source, e["source_iri"]).lower()]
        if not hits:
            console.print(f"[red]no source class matches '{query}'[/red]")
            return
        if len(hits) > 1:
            console.print(f"[dim]{len(hits)} classes match; running the first. Others: " +
                          ", ".join(tm.label(self.source, e['source_iri']) for e in hits[1:6]) + "[/dim]")
        try:
            self.run_one(hits[0], self.episodes.index(hits[0]) + 1, record_it=False)
        except (Stop, KeyboardInterrupt):
            console.print("[yellow]stopped[/yellow]")

    def status(self, short: bool = False) -> None:
        records = list(self.records.values())
        counts = {d: sum(r["decision"] == d for r in records) for d in ("match", "NONE")}
        left = len(self.episodes) - len(records)
        failed = f" · no submission {len(self.failed)}" if self.failed else ""
        head = f"[{len(records):,}/{len(self.episodes):,}] match {counts['match']:,} · NONE {counts['NONE']:,}{failed}"
        session = f" · this session {duration(sum(self.times))}" if self.times else ""
        if not left:
            console.print(f"  [dim]{head}{session} · every class decided[/dim]" if short else f"{head}{session} · every class decided")
        elif short:
            per, _ = self.per_class()
            finish = time.strftime("%a %H:%M", time.localtime(time.time() + per * left))
            console.print(f"  [dim]{head}{session} · about {duration(per * left)} left ({per:.1f} s/class) · "
                          f"done ≈ {finish}[/dim]")
        else:
            console.print(f"{head}{session}\n{self.eta(left, detail=True)}")

    def save(self) -> None:
        records = [self.records[e["source_iri"]] for e in self.episodes if e["source_iri"] in self.records]
        if not records:
            console.print("[dim]no decisions yet[/dim]")
            return
        summary = tm.write_outputs(self.out, records, len(self.episodes), self.source_path.as_uri(),
                                   self.target_path.as_uri(), self.source, self.target, self.reference, self.args.model,
                                   self.args.k, self.args.threshold, sum(r.get("seconds", 0) for r in records),
                                   candidates={e["source_iri"]: [c["iri"] for c in e["candidates"]] for e in self.episodes})
        console.print(f"[dim]saved in {self.out}: alignment.rdf ({summary['mappings']:,} mappings), decisions.tsv "
                      f"({len(records):,} decided classes, with labels), decisions.jsonl, summary.json, tasks/"
                      + (", results.md, topk.svg" if self.reference else "") + "[/dim]")
        if self.reference:
            self.scores(summary)

    def scores(self, summary: dict) -> None:
        t = summary["scores"]
        table = Table(title=f"{t['method']}: {t['classes_scored']:,} of {t['reference_classes']:,} "
                            "reference classes scored", title_justify="left")
        for column in ("Top-k", "Precision", "Recall", "F-measure", "Classes with a hit"):
            table.add_column(column, justify="left" if column == "Top-k" else "right")
        for k, v in t["k"].items():
            table.add_row(f"Top-{k}", f"{v['Precision']:.3f}", f"{v['Recall']:.3f}", f"{v['F-measure']:.3f}",
                          f"{v['classes_with_a_hit']:,}")
        console.print(table)

    def score_skill(self, rest: str) -> None:
        """/score: run the om-score skill, shown as an agent tool call."""
        import shlex
        import subprocess

        runs = shlex.split(rest) or [str(self.out)]
        args = [*runs] + (["--reference", str(self.reference)] if self.reference and not rest else [])
        if not rest and not self.reference:
            console.print("[yellow]no reference for this run: /score <run folders> --reference <file>[/yellow]")
            return
        console.print(f"  [green]⏺[/green] om-score {' '.join(shlex.quote(a) for a in args)}")
        subprocess.run([sys.executable, str(HERE / "om_score.py"), *args], cwd=HERE, check=False)

    def model(self) -> None:
        info = model_info(self.args.model)
        for key in ("name", "server", "version", "weights", "store", "quantization", "parameters", "context", "settings",
                    "error"):
            if info.get(key):
                value = " · ".join(" ".join(line.split()) for line in str(info[key]).splitlines())
                console.print(f"  [bold cyan]{key:<12}[/bold cyan] {value}")
        console.print(f"  [bold cyan]{'thinking':<12}[/bold cyan] {'on' if self.think else 'off'}")

    def set_thinking(self, value: str) -> None:
        if value not in ("on", "off"):
            console.print(f"thinking is {'on' if self.think else 'off'} (/thinking on or /thinking off)")
            return
        self.think = value == "on"
        self.matcher = tm.Matcher(self.args.model, self.out, 1, "local", think=self.think)
        console.print(f"[dim]thinking {value} for the next classes (each decision records whether thinking was on)[/dim]")

    # -- the prompt -----------------------------------------------------------------------------------------------------
    def loop(self) -> None:
        banner(self.info, self.think, self.auto)
        if "error" in self.info:
            return
        self.open()
        left = len(self.episodes) - len(self.records)
        if left and ask(f"\n[bold]Align all {left:,} remaining classes now[/bold] ({self.eta(left, short=True)})? The agent asks "
                        "before each command. [y/N] ").lower() in ("y", "yes"):
            self.run("all")
        else:
            console.print("[dim]/run all aligns every class, /run 5 the next five, /help lists the commands[/dim]")
        while True:
            line = ask("[bold magenta]❯[/bold magenta] ")
            if not line:
                continue
            command, _, rest = line.partition(" ")
            if command in ("/quit", "/exit"):
                break
            elif command == "/run":
                if rest.strip() and rest.strip() != "all" and not rest.strip().isdigit():
                    console.print("[red]/run N or /run all[/red]")
                else:
                    self.run(rest.strip() or "1")
            elif command == "/align" and rest:
                self.align(rest)
            elif command == "/status":
                self.status()
            elif command in ("/save", "/results"):
                self.save()
            elif command == "/score":
                self.score_skill(rest.strip())
            elif command == "/open":
                self.open()
            elif command == "/thinking":
                self.set_thinking(rest.strip())
            elif command == "/permissions":
                self.auto = not self.auto
                console.print("[dim]" + ("allowing all commands" if self.auto else "asking before every command") + "[/dim]")
            elif command == "/model":
                self.model()
            elif command == "/skills":
                skills()
            elif command == "/compact":
                self.compact = not self.compact
                console.print(f"[dim]transcript: {'one line per class' if self.compact else 'full'}[/dim]")
            else:
                for name, what in HELP:
                    console.print(f"  [bold]{name:<24}[/bold] [dim]{what}[/dim]")
        console.print(f"[dim]bye; decisions are in {self.log}[/dim]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Claude-Code-style console for the OA-RLAIF ontology-matching agent")
    parser.add_argument("--model", default=tm.DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=40, help="candidates per class (training used 40)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-think", action="store_true", help="start with thinking off (as in training)")
    parser.add_argument("--compact", action="store_true", help="one line per class instead of the full transcript")
    AgentConsole(parser.parse_args()).loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
