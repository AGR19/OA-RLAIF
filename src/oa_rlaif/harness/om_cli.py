"""`om`: the ontology-matching tool the agent calls inside the Docker sandbox (stdlib only).

    om profile                           print the modelling profiles of the source and target ontologies
    om show <IRI>                        print the source entity card
    om candidates <IRI> [--k N]          print numbered candidate cards C1..Ck
    om submit <C#|C#,C#|NONE> [--relation R] [--confidence X]   (several targets: comma-separated, at most 3)
                                         validate and print the decision as TSV (the mini-SWE-agent submission
                                         when chained after `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT &&`)

The task is read from $OM_TASK (default /task/episode.json): one decision episode as written by the data build.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

RELATIONS = {"=", "<", ">", "classified_under"}


def load() -> dict:
    with open(os.environ.get("OM_TASK", "/task/episode.json"), encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="om")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("profile")
    show = sub.add_parser("show")
    show.add_argument("iri")
    cands = sub.add_parser("candidates")
    cands.add_argument("iri")
    cands.add_argument("--k", type=int, default=20)
    submit = sub.add_parser("submit")
    submit.add_argument("choice")
    submit.add_argument("--relation", default="=")
    submit.add_argument("--confidence", type=float, default=1.0)
    args = parser.parse_args(argv)
    episode = load()

    if args.command == "profile":
        print("Source ontology: " + (episode.get("source_profile") or "unknown"))
        print("Target ontology: " + (episode.get("target_profile") or "unknown"))
        return 0
    if args.command in ("show", "candidates") and args.iri != episode["source_iri"]:
        print(f"error: unknown source entity {args.iri}; this task is about {episode['source_iri']}", file=sys.stderr)
        return 2
    if args.command == "show":
        print(episode["source_card"])
        return 0
    if args.command == "candidates":
        for i, cand in enumerate(episode["candidates"][: args.k], 1):
            print(f"[C{i}]\n{cand['card']}")
        return 0

    choice = args.choice.strip().upper()
    if args.relation not in RELATIONS:
        print(f"error: relation must be one of {sorted(RELATIONS)}", file=sys.stderr)
        return 2
    if not 0.0 <= args.confidence <= 1.0:
        print("error: confidence must be within [0, 1]", file=sys.stderr)
        return 2
    labels = ["NONE"] if choice == "NONE" else choice.split(",")
    if len(labels) > 3 or len(set(labels)) != len(labels):
        print("error: list at most 3 distinct candidates, e.g. C2 or C2,C5", file=sys.stderr)
        return 2
    targets = []
    for label in labels:
        if label == "NONE" and len(labels) == 1:
            targets.append("NONE")
        elif label.startswith("C") and label[1:].isdigit() and 1 <= int(label[1:]) <= len(episode["candidates"]):
            targets.append(episode["candidates"][int(label[1:]) - 1]["iri"])
        else:
            print(f"error: {label} is not a listed candidate (C1..C{len(episode['candidates'])}) or NONE", file=sys.stderr)
            return 2
    lines = [f"{episode['source_iri']}\t{t}\t{args.relation}\t{args.confidence:.4f}" for t in targets]
    print("source_iri\ttarget_iri\trelation\tconfidence")
    print("\n".join(lines))
    line = "\n".join(lines)
    decision = os.path.join(os.path.dirname(os.environ.get("OM_TASK", "/task/episode.json")), "decision.tsv")
    try:
        with open(decision, "w", encoding="utf-8") as f:
            f.write("source_iri\ttarget_iri\trelation\tconfidence\n" + line + "\n")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
