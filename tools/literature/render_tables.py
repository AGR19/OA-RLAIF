"""Render verified.json into the ranked Markdown tables inside docs/LITERATURE_REVIEW.md.

Only rows produced by verify_dois.py are rendered. The narrative around the tables is written by hand;
this script replaces the text between <!-- TABLE:<cat> --> and <!-- /TABLE:<cat> --> markers.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVIEW = ROOT / "docs" / "LITERATURE_REVIEW.md"


def doi_link(row: dict) -> str:
    if row.get("venue_doi"):
        return f"[{row['venue_doi']}](https://doi.org/{row['venue_doi']})"
    if row.get("doi_verified"):
        return f"[{row['doi_verified']}](https://doi.org/{row['doi_verified']})"
    if row.get("url"):
        return f"no DOI · [link]({row['url']})"
    return "—"


def table(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: (r.get("citations") is None, -(r.get("citations") or 0)))
    lines = [
        "| # | Paper | Venue | Year | Citations | DOI |",
        "|---|---|---|---|---:|---|",
    ]
    for rank, row in enumerate(rows, 1):
        found, given = row.get("found_title") or "", row["title"]
        title = (given if not found or (given.lower().startswith(found.lower()) and len(given) > len(found)) else found)
        title = title.replace("|", "\\|")
        cites = "n/a" if row.get("citations") is None else f"{row['citations']:,}"
        year = row.get("year") or next((w for w in reversed(str(row.get("venue", "")).split()) if w.isdigit() and len(w) == 4), "")
        lines.append(f"| {rank} | {title} | {row['venue']} | {year} | {cites} | {doi_link(row)} |")
    return "\n".join(lines)


def main() -> int:
    data = json.loads((HERE / "verified.json").read_text())
    rows = [r for r in data["papers"] if r["status"] in ("verified", "no_doi")]
    dropped = [r for r in data["papers"] if r["status"] == "dropped"]
    text = REVIEW.read_text()
    for cat in ("rl", "opd", "om", "closest", "verify", "survey"):
        block = table([r for r in rows if r["cat"] == cat])
        pattern = re.compile(rf"(<!-- TABLE:{cat} -->)(.*?)(<!-- /TABLE:{cat} -->)", re.S)
        if not pattern.search(text):
            print(f"marker for {cat} missing", file=sys.stderr)
            return 1
        text = pattern.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(3)}", text)
    drop_lines = "\n".join(f"- `{r['id']}` — {r['title']}: {r['reason']}" for r in dropped) or "- none"
    text = re.sub(r"(<!-- DROPPED -->)(.*?)(<!-- /DROPPED -->)", lambda m: f"{m.group(1)}\n{drop_lines}\n{m.group(3)}", text, flags=re.S)
    text = re.sub(r"Snapshot date: [0-9-]+|Snapshot date: _pending_", f"Snapshot date: {data['snapshot']}", text)
    REVIEW.write_text(text)
    print(f"rendered {len(rows)} rows, {len(dropped)} dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
