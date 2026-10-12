"""Multi-target submissions: parser, reward (F1 against gold), om CLI and equivalence extraction."""

import json

from oa_rlaif.harness import om_cli
from oa_rlaif.harness.prompts import gold_answer, parse_decision
from oa_rlaif.ontology.alignment_io import select_equivalences
from oa_rlaif.ontology.models import Alignment
from oa_rlaif.reward.task_reward import correct, task_reward

EP = {"id": "m1", "source_iri": "s", "kind": "gold", "relation": "=", "source_card": "IRI: s\nname: heart",
      "candidates": [{"iri": "t1", "card": "IRI: t1\nname: heart"}, {"iri": "t2", "card": "IRI: t2\nname: lung"}, {"iri": "t3", "card": "IRI: t3\nname: cardiac organ"}],
      "gold": ["t1", "t3"]}


def reply(choice):
    return f"THOUGHT: x\n```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit {choice} --relation = --confidence 0.9\n```"


def test_parser_accepts_up_to_three_distinct_targets():
    d = parse_decision(reply("C1,C3"), EP)
    assert d.valid and d.choices == ("t1", "t3") and d.choice == "t1"
    assert not parse_decision(reply("C1,C1"), EP).valid
    assert not parse_decision(reply("C1,C4"), EP).valid
    assert not parse_decision(reply("C1,C2,C3,C1"), EP).valid


def test_reward_is_f1_so_extra_picks_do_not_pay():
    assert correct(("t1", "t3"), EP) == 1.0
    assert round(correct(("t1",), EP), 3) == 0.667          # one of two gold
    single = {**EP, "gold": ["t1"]}
    assert correct(("t1",), single) == 1.0
    assert round(correct(("t1", "t2"), single), 3) == 0.667  # padding with a wrong target is penalised
    assert correct(("NONE",), {**EP, "gold": []}) == 1.0
    assert task_reward(reply("C1,C3"), EP)["gold"] == 1.0


def test_gold_answer_lists_all_present_gold():
    assert "om submit C1,C3 " in gold_answer(EP)


def test_om_cli_prints_one_line_per_target(tmp_path, monkeypatch, capsys):
    task = tmp_path / "episode.json"
    task.write_text(json.dumps(EP))
    monkeypatch.setenv("OM_TASK", str(task))
    assert om_cli.main(["submit", "C1,C3", "--relation", "=", "--confidence", "0.8"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith("source_iri") and [l.split("\t")[1] for l in out[1:]] == ["t1", "t3"]
    assert om_cli.main(["submit", "C1,C1"]) == 2


def test_select_equivalences_keeps_multi_target_sources_but_not_shared_targets():
    picks = [Alignment("s1", "t1", "=", 0.9), Alignment("s1", "t2", "=", 0.8), Alignment("s2", "t1", "=", 0.7)]
    kept = {(a.source_iri, a.target_iri) for a in select_equivalences(picks)}
    assert kept == {("s1", "t1"), ("s1", "t2")}


def test_more_than_three_gold_targets():
    """om submit names at most 3 targets: 3 correct picks out of 6 gold is a fully correct answer everywhere."""
    from oa_rlaif.harness.prompts import fully_correct

    ep = {**EP, "candidates": [{"iri": f"t{i}", "card": f"IRI: t{i}"} for i in range(1, 9)],
          "gold": [f"t{i}" for i in range(1, 7)]}
    assert correct(("t1", "t2", "t3"), ep) == 1.0                      # was 0.667 (recall over all 6)
    assert round(correct(("t1", "t2"), ep), 3) == 0.8                   # 2 of the 3 allowed
    assert round(correct(("t1", "t2", "t7"), ep), 3) == 0.667           # a wrong pick costs precision
    assert fully_correct(("t4", "t5", "t6"), ep["gold"]) and not fully_correct(("t1", "t2"), ep["gold"])
    assert not fully_correct(("t1", "t2", "t7"), ep["gold"])
    assert fully_correct(("NONE",), []) and not fully_correct(("NONE",), ["t1"])
    assert fully_correct(("t1", "t3"), ["t1", "t3"]) and not fully_correct(("t1",), ["t1", "t3"])
    assert task_reward(gold_answer(ep), ep)["gold"] == 1.0              # the worked answer gets the full gold term
