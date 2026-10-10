"""Reward math and format gate. Eq. 1 is checked against hand-computed softmax values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oa_rlaif.harness.prompts import gold_answer, parse_decision, privileged_context
from oa_rlaif.reward.combined import make_reward_functions
from oa_rlaif.reward.dral_reward import DralScorer
from oa_rlaif.reward.task_reward import task_reward


@pytest.fixture
def scorer(tmp_path):
    # source s; candidates a (close), b (parent-like, medium), c (far). Unit vectors make cosines exact.
    iris = np.array(["s", "a", "b", "c"])
    z = np.array([[1, 0], [1, 0], [0.6, 0.8], [0, 1]], dtype=np.float32)
    t = np.array([[1, 0], [1, 0], [0.6, 0.8], [-1, 0]], dtype=np.float32)
    path = tmp_path / "p.npz"
    np.savez(path, iris=iris, z=z, t=t)
    return DralScorer([path], temperature=0.1)


def episode(kind="gold", gold=("a",)):
    return {"id": "e1", "pair_id": "test/pair", "kind": kind, "source_iri": "s", "relation": "=",
            "source_card": "IRI: s\nname: S", "gold": list(gold),
            "candidates": [{"iri": x, "card": f"IRI: {x}\nname: {x.upper()}"} for x in ("a", "b", "c")]}


def reply(choice):
    return ("THOUGHT: lexical match of names\n```mswea_bash_command\n"
            f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit {choice} --relation = --confidence 0.9\n```")


def test_eq1_matches_hand_computed_softmax(scorer):
    sims = np.array([1.0, 0.6, -0.5])  # 0.5*cos_z + 0.5*cos_t for a, b, c
    expected = np.exp(sims / 0.1) / np.exp(sims / 0.1).sum()
    got = scorer.probabilities("s", ["a", "b", "c"])
    assert np.allclose(got, expected, atol=1e-6)
    assert scorer.reward("s", ["a", "b", "c"], "a") == pytest.approx(expected[0], abs=1e-6)
    assert scorer.reward("s", ["a", "b", "c"], "NONE") == pytest.approx(1 - expected.max(), abs=1e-6)
    assert scorer.reward("unknown", ["a"], "a") is None


@pytest.mark.parametrize("choice,gold_expected", [("C1", 1.0), ("C2", 0.0), ("C3", 0.0), ("NONE", 0.0)])
def test_task_reward_cases(scorer, choice, gold_expected):
    result = task_reward(reply(choice), episode(), scorer)
    assert result["valid"] and result["gold"] == gold_expected
    probs = scorer.probabilities("s", ["a", "b", "c"])
    dral = {"C1": probs[0], "C2": probs[1], "C3": probs[2], "NONE": 1 - probs.max()}[choice]
    assert result["task"] == pytest.approx(0.5 * gold_expected + 0.5 * dral, abs=1e-6)


def test_near_miss_beats_far_miss(scorer):
    near = task_reward(reply("C2"), episode(), scorer)["task"]
    far = task_reward(reply("C3"), episode(), scorer)["task"]
    assert near > far


def test_none_is_correct_when_gold_absent(scorer):
    result = task_reward(reply("NONE"), episode(gold=()), scorer)
    assert result["gold"] == 1.0


def test_silver_and_unlabeled_scaling(scorer):
    full = task_reward(reply("C1"), episode(), scorer)["task"]
    assert task_reward(reply("C1"), episode("silver"), scorer)["task"] == pytest.approx(0.5 * full)
    unlabeled = task_reward(reply("C1"), episode("unlabeled", gold=()), scorer)
    assert unlabeled["gold"] is None
    assert unlabeled["task"] == pytest.approx(0.3 * scorer.reward("s", ["a", "b", "c"], "a"))


@pytest.mark.parametrize("bad", [
    "no thought\n```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit C1\n```",
    reply("C9"),  # index out of range (hallucinated candidate)
    reply("C1") + "\n" + reply("C2").split("\n", 1)[1],  # two bash blocks
    "THOUGHT: x\n```mswea_bash_command\nom submit C1\n```",  # missing completion echo
    "THOUGHT: x\n```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit C1 --confidence 1.7\n```",
])
def test_invalid_replies_get_zero(scorer, bad):
    result = task_reward(bad, episode(), scorer)
    assert not result["valid"] and result["task"] == 0.0


def test_gold_answer_round_trips_and_privileged_context_names_answer():
    ep = episode(gold=("b",))
    decision = parse_decision(gold_answer(ep), ep)
    assert decision.valid and decision.choice == "b" and decision.index == 2
    assert "C2" in privileged_context(ep)
    none_ep = episode(gold=())
    assert parse_decision(gold_answer(none_ep), none_ep).choice == "NONE"


def test_combined_weights_and_invalid_gate(scorer):
    (task_fn, judge_fn), weights = make_reward_functions(scorer=scorer, judge=None)
    assert weights == [0.8, 0.2]
    import json
    eps = [json.dumps(episode())] * 2
    task = task_fn(None, [reply("C1"), "garbage"], eps)
    judge = judge_fn(None, [reply("C1"), "garbage"], eps)
    assert task[1] == 0.0 and judge[1] == 0.0 and task[0] > 0.5 and judge[0] == 0.5
    assert math.isclose(0.8 * task[1] + 0.2 * judge[1], 0.0)
