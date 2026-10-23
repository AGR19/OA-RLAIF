"""The candidate question pplx-decider answers for one class (arm A, and the harness's NONE/invalid guard), in the
System-One wire format that JevAgentHarness and pplx_decider share.

    {"state": "<task + source card>",
     "questions": {"match": {"type": "choice", "instructions": "...",
                             "criteria": {"C1": "<name | synonyms | parents>", ..., "NONE": "..."}}}}

The options are the episode's candidates under the same names the agent sees (C1, C2, ... in the shown order), plus
NONE. The answer's probabilities give the pick (the most likely option) and the ranking of the candidates (Top-k ranks
2-5). pplx-decider places the context in the state and the options in the criteria (its model card: "Put the context
the questions need in the state").
"""

from __future__ import annotations

from typing import Any

QUESTION_ID = "match"
NONE = "NONE"
CARD_CHARS = 400  # one option's description: the candidate card without its IRI, cut to this length
STATE_CHARS = 4000

INSTRUCTIONS = {
    "=": "Which candidate target entity is the same concept as the source entity? Answer NONE if no candidate is.",
    "classified_under": ("Which candidate target category classifies the source concept (the category it falls "
                         "under)? Answer NONE if no candidate does."),
}
NONE_TEXT = {
    "=": "None of the candidates is the same concept as the source entity",
    "classified_under": "None of the candidates classifies the source concept",
}


def card_text(card: str, limit: int = CARD_CHARS) -> str:
    """A card without its IRI line, its fields joined on one line: 'name: heart | synonyms: ... | parents: ...'."""
    lines = [line.strip() for line in (card or "").splitlines() if line.strip() and not line.startswith("IRI:")]
    text = " | ".join(lines)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def option_names(episode: dict) -> list[str]:
    return [f"C{i}" for i in range(1, len(episode["candidates"]) + 1)]


def request(episode: dict) -> dict[str, Any]:
    """The System-One request for one class's candidate choice."""
    relation = "classified_under" if episode.get("relation") == "classified_under" else "="
    task = ("Align a source entity to the target ontology: find the target category it is classified under."
            if relation == "classified_under" else
            "Align a source entity to the target ontology: find the target entity that is the same concept.")
    state = f"TASK: {task}\nPAIR: {episode['pair_id']}\n\nSOURCE ENTITY:\n{episode['source_card']}"
    criteria = {name: card_text(c["card"]) for name, c in zip(option_names(episode), episode["candidates"])}
    criteria[NONE] = NONE_TEXT[relation]
    return {"state": state[:STATE_CHARS],
            "questions": {QUESTION_ID: {"type": "choice", "instructions": INSTRUCTIONS[relation], "criteria": criteria}}}


def read(episode: dict, response: dict[str, Any]) -> dict[str, Any]:
    """The answer to :func:`request`: pick (a candidate IRI or NONE), its probability, the probabilities of every
    option, and the candidates ranked by probability (NONE left out)."""
    answer = (response.get("answers") or {}).get(QUESTION_ID) or {}
    probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
    names = option_names(episode)
    iri = {name: c["iri"] for name, c in zip(names, episode["candidates"])}
    best = max(probs, key=probs.get) if probs else NONE
    ranked = sorted((n for n in names if n in probs), key=lambda n: -probs[n])
    return {"pick": iri.get(best, NONE), "pick_option": best, "p": probs.get(best, 0.0),
            "best_candidate": ranked[0] if ranked else None,
            "p_best_candidate": probs.get(ranked[0], 0.0) if ranked else 0.0,
            "ranking": [iri[n] for n in ranked], "probabilities": {k: round(v, 4) for k, v in probs.items()}}


def reply_row(episode: dict, answer: dict[str, Any], seconds: float, usage: dict | None = None) -> dict[str, Any]:
    """Arm A's reply row, in the agent-row layout full_eval.score reads ('turns' marks an agent row): the decision
    model's pick is the submission, and its probabilities order the other candidates."""
    match = answer["pick"] != NONE
    return {"id": episode["id"], "decision": "match" if match else "NONE",
            "targets": [answer["pick"]] if match else [],
            "relation": episode.get("relation", "="), "confidence": round(answer["p"], 4), "exit": "Submitted",
            "model_calls": 0, "extra_commands": 0, "format_errors": 0, "turns": [],
            "prompt_tokens": int((usage or {}).get("input_tokens") or 0), "completion_tokens": 0,
            "thinking_chars": 0, "answer_in_thinking": 0, "seconds": round(seconds, 3),
            "ranking": answer["ranking"], "decider": {"probabilities": answer["probabilities"]}}
