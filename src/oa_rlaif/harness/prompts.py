"""One prompt/answer format for SDFT, GRPO, evaluation and the mini-SWE-agent harness.

Decision episodes: candidates are numbered C1..Ck. The model answers with a short THOUGHT and one bash block in
mini-SWE-agent v2 format that submits by index:

    THOUGHT: <evidence>
    ```mswea_bash_command
    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit C7 --relation = --confidence 0.92
    ```

Submitting an index (or NONE) instead of an IRI makes invented targets impossible to emit undetected, keeps
completions short, and is exactly what `om submit` executes inside the Docker harness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DECISION_SYSTEM = (
    "You are OA-RLAIF, an ontology alignment agent. Decide which candidate target entity corresponds to the "
    "source entity. Weigh lexical evidence (names, synonyms), structural evidence (parents, children) and "
    "semantic evidence (definitions, codes). Respect each ontology's modelling assumptions (its profile): "
    "a thesaurus 'broader' term is not a subclass, part-of is not is-a, residual 'other/unspecified' categories "
    "only match residual categories, and different semantic categories never match. "
    "If no candidate corresponds, submit NONE. If the source corresponds to several candidates equally well "
    "(for example two synonymous target classes), list them as C#,C# (at most 3); otherwise submit one.\n"
    "Reply with one line starting 'THOUGHT:' that names your evidence, then exactly one bash code block:\n"
    "```mswea_bash_command\n"
    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit <C# | C#,C# | NONE> --relation <=|<|>|classified_under> "
    "--confidence <0..1>\n```"
)

KNOWLEDGE_SYSTEM = (
    "You are OA-RLAIF, an expert in ontologies and ontology matching. Answer from your own knowledge, "
    "concisely and precisely. Do not invent identifiers you are unsure of."
)

RELATION_TEXT = {"=": "equivalent to", "classified_under": "classified under", "<": "narrower than", ">": "broader than",
                 "related": "most closely related to (equivalent, broader or narrower)"}


def decision_user(episode: dict) -> str:
    task = {"classified_under": "classified under", "related": RELATION_TEXT["related"]}.get(episode.get("relation"), "equivalent to")
    lines = [f"Task: find the target entity that the source entity is {task}.",
             f"Alignment: {episode['pair_id']}", ""]
    if episode.get("source_profile") or episode.get("target_profile"):
        lines += ["ONTOLOGY PROFILES (modelling assumptions)",
                  "Source: " + (episode.get("source_profile") or "unknown"),
                  "Target: " + (episode.get("target_profile") or "unknown"), ""]
    lines += ["SOURCE", episode["source_card"], "", "CANDIDATES"]
    for i, cand in enumerate(episode["candidates"], 1):
        lines.append(f"[C{i}]\n{cand['card']}")
    return "\n".join(lines)


def decision_messages(episode: dict) -> list[dict]:
    return [{"role": "system", "content": DECISION_SYSTEM}, {"role": "user", "content": decision_user(episode)}]


def _fields(card_text: str) -> dict[str, str]:
    out = {}
    for line in card_text.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            out[key] = value
    return out


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower().replace("grey", "gray")) if len(w) > 2}


def evidence_thought(episode: dict) -> str:
    """Evidence-based reasoning derived from the cards (used in the teacher's demonstration)."""
    source = _fields(episode["source_card"])
    names = source.get("name", "") + " " + source.get("synonyms", "")
    index = {c["iri"]: i for i, c in enumerate(episode["candidates"], 1)}
    gold = next((g for g in episode.get("gold", []) if g in index), None)
    if gold is None:
        best = max(episode["candidates"], key=lambda c: len(_tokens(names) & _tokens(_fields(c["card"]).get("name", ""))),
                   default=None)
        hint = f"; the closest, C{index[best['iri']]} '{_fields(best['card']).get('name', '')}', denotes a different concept" if best else ""
        return f"no candidate's name, synonyms or parents correspond to '{source.get('name', '')}'{hint}."
    target = _fields(next(c["card"] for c in episode["candidates"] if c["iri"] == gold))
    shared = sorted(_tokens(names) & _tokens(target.get("name", "") + " " + target.get("synonyms", "")))
    parts = [f"C{index[gold]} '{target.get('name', '')}' matches source '{source.get('name', '')}'"]
    if shared:
        parts.append("lexically (shared terms: " + ", ".join(shared[:5]) + ")")
    parents = _tokens(source.get("parents", "")) & _tokens(target.get("parents", ""))
    if parents:
        parts.append("and structurally (parents agree: " + ", ".join(sorted(parents)[:3]) + ")")
    if target.get("definition") and not shared:
        parts.append("and its definition describes the same concept")
    return " ".join(parts) + "."


def gold_answer(episode: dict, thought: str = "") -> str:
    """Reference completion (used as the teacher's privileged demonstration, never shown to the student)."""
    index = {c["iri"]: i for i, c in enumerate(episode["candidates"], 1)}
    present = sorted(index[g] for g in episode.get("gold", []) if g in index)[:3]
    choice = ",".join(f"C{i}" for i in present) or "NONE"
    relation = episode.get("relation", "=")
    relation = "classified_under" if relation == "classified_under" else ("=" if relation not in ("<", ">") else relation)
    thought = thought or evidence_thought(episode)
    return (f"THOUGHT: {thought}\n```mswea_bash_command\n"
            f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit {choice} --relation {relation} --confidence 0.95\n```")


def privileged_context(episode: dict) -> str:
    """OPSD privileged context: what the frozen teacher sees besides the prompt, i.e. the ground-truth answer."""
    index = {c["iri"]: i for i, c in enumerate(episode["candidates"], 1)}
    golds = [f"C{index[g]}" for g in episode.get("gold", []) if g in index]
    if golds:
        why = f"Reference alignment: the correct answer is {', '.join(golds)} ({RELATION_TEXT.get(episode.get('relation'), 'equivalent to')})."
    else:
        why = "Reference alignment: none of the candidates is correct; the correct answer is NONE."
    return why + "\nExample of a correct reply:\n" + gold_answer(episode)


SUBMIT = re.compile(
    r"```mswea_bash_command\s*\n\s*echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\s*&&\s*om submit\s+(C\d+(?:,C\d+){0,2}|NONE)"
    r"(?:\s+--relation\s+(\S+))?(?:\s+--confidence\s+([0-9]*\.?[0-9]+))?\s*\n```", re.S)


@dataclass(frozen=True)
class Decision:
    valid: bool
    choice: str | None = None  # IRI, "NONE" or None when invalid
    index: int | None = None
    relation: str = ""
    confidence: float | None = None
    error: str = ""
    choices: tuple = ()  # every submitted IRI (1-3); `choice` is the first one


def parse_decision(completion: str, episode: dict) -> Decision:
    """Strict parser: exactly one well-formed submit block, index within range, confidence within [0, 1]."""
    if not isinstance(completion, str) or not completion.strip().startswith("THOUGHT:"):
        return Decision(False, error="missing THOUGHT line")
    blocks = completion.count("```mswea_bash_command")
    if blocks != 1:
        return Decision(False, error=f"expected one bash block, found {blocks}")
    match = SUBMIT.search(completion)
    if not match:
        return Decision(False, error="no valid om submit command")
    label, relation, confidence = match.group(1), match.group(2) or "=", match.group(3)
    conf = float(confidence) if confidence is not None else None
    if conf is not None and not 0.0 <= conf <= 1.0:
        return Decision(False, error="confidence outside [0,1]")
    if label == "NONE":
        return Decision(True, "NONE", None, relation, conf, choices=("NONE",))
    indices = [int(part[1:]) for part in label.split(",")]
    if len(set(indices)) != len(indices):
        return Decision(False, error="the same candidate is listed twice")
    for index in indices:
        if not 1 <= index <= len(episode["candidates"]):
            return Decision(False, error=f"candidate C{index} does not exist")
    iris = tuple(episode["candidates"][i - 1]["iri"] for i in indices)
    return Decision(True, iris[0], indices[0], relation, conf, choices=iris)
