"""Thinking on / off for Gemma 4: switching it, and separating the thinking text from the answer.

Gemma 4's chat template turns thinking on with `<|think|>` in the system turn (chat_template_kwargs
enable_thinking=True); the model then writes `<|channel>thought ... <channel|>` before its answer. Both markers are
special tokens, so a thinking run keeps special tokens in its output (skip_special_tokens=False) and split() separates
the two parts (Ollama returns the thinking apart, in its reasoning field: pass it as `reasoning`). The models were
trained with thinking off: with it on, they often write their usual answer inside the thinking text and never close
it; the answer is then taken from there (answer_in_thinking=True). The full evaluation (full_eval/infer.py) and the
mini-SWE-agent harness (om_agent/) use this one function.
"""

from __future__ import annotations

OPEN, CLOSE, END = "<|channel>", "<channel|>", "<turn|>"
ANSWER_MARK = "```mswea_bash_command"


def split(text: str, reasoning: str = "") -> dict:
    """{"answer", "thinking", "answer_in_thinking"} from one raw completion (special tokens kept), plus the thinking
    a server returned apart from it (`reasoning`)."""
    text = text or ""
    thinking, answer = "", text
    if OPEN in text:
        before, _, rest = text.partition(OPEN)
        if CLOSE in rest:
            thinking, _, after = rest.partition(CLOSE)
            answer = before + after
        else:  # never closed: everything after the marker is thinking
            thinking, answer = rest, before
        thinking = thinking.removeprefix("thought").lstrip("\n")
    answer = answer.replace(END, "").replace(OPEN, "").replace(CLOSE, "").strip()
    thinking = "\n".join(x for x in ((reasoning or "").strip(), thinking.replace(END, "").strip()) if x)
    inside = ANSWER_MARK not in answer and ANSWER_MARK in thinking
    if inside:  # the answer was written inside the thinking text: take it from there, from its THOUGHT line on
        start = thinking.rfind("THOUGHT:", 0, thinking.find(ANSWER_MARK))
        answer = thinking[start:] if start >= 0 else thinking
    return {"answer": answer, "thinking": thinking, "answer_in_thinking": inside}


def settings(think: bool) -> dict:
    """Generation settings for a vLLM chat call (offline LLM.chat or the OpenAI server's extra_body)."""
    return {"chat_template_kwargs": {"enable_thinking": think}, "skip_special_tokens": not think}
