"""Baseline metric rows and knowledge passages from the non-alignment files in the raw folder.

Baselines: every metrics file (DH performance CSVs and cubes, results.md tables, Bio-ML leaderboard / results
JSON / results HTML, Complex-OM GED-and-class-eval TSVs) becomes rows for the evaluation comparison tables.
Knowledge passages: the datasource, both DRAL-OA papers, benchmark READMEs/docs/pages and the transcribed
screenshot are cut into ~1.5k-character passages; A4B later writes questions about each (SDFT knowledge).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

from lxml import html as lxml_html

from ..paths import ROOT, relative_to_raw
from .splits import stable_fraction

# ---------------------------------------------------------------- baselines


def _csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        yield from csv.DictReader(f, delimiter=delimiter)


def _markdown_tables(text: str):
    header = None
    for line in text.splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = cells
            elif set("".join(cells)) <= set("-: "):
                continue
            else:
                yield dict(zip(header, cells))
        else:
            header = None


def _json_records(value, prefix=""):
    if isinstance(value, list):
        for i, item in enumerate(value):
            yield from _json_records(item, f"{prefix}[{i}]")
    elif isinstance(value, dict):
        scalars = {k: v for k, v in value.items() if not isinstance(v, (dict, list))}
        if scalars:
            yield {"path": prefix, **scalars}
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                yield from _json_records(v, f"{prefix}.{k}")


def _html_tables(path: Path):
    tree = lxml_html.fromstring(path.read_bytes())
    for table in tree.iter("table"):
        rows = [[" ".join(c.text_content().split()) for c in tr.iter("td", "th")] for tr in table.iter("tr")]
        rows = [r for r in rows if any(r)]
        if len(rows) > 1:
            for row in rows[1:]:
                yield dict(zip(rows[0], row))


def baseline_rows(path: Path) -> list[dict]:
    name = path.name.lower()
    if name.endswith((".csv", ".tsv")):
        records = list(_csv_rows(path))
    elif name.endswith(".md"):
        records = list(_markdown_tables(path.read_text(errors="ignore")))
    elif name.endswith(".json"):
        records = list(_json_records(json.loads(path.read_text())))
    elif name.endswith((".html", ".htm")):
        records = list(_html_tables(path))
    else:
        records = []
    source = relative_to_raw(path)
    return [{"source": source, "row": {k: v for k, v in r.items() if k}} for r in records]


# ---------------------------------------------------------------- knowledge passages

TRANSCRIPTIONS = {"Screenshot 2026-03-17 at 4.51.58 AM.png": "configs/knowledge/snomed_icd10cm_map_screenshot.md"}
# Binary files are never read as text: without a transcription they yield no passages.
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".xlsx", ".xls", ".zip", ".gz", ".docx", ".pptx"}


def _file_key(name: str) -> str:
    """macOS screenshot names contain U+202F (narrow no-break space) before AM/PM; compare on plain spaces."""
    return " ".join(name.replace("\u202f", " ").replace("\xa0", " ").split())


def _pdf_pages(path: Path, first: int = 1, last: int | None = None) -> list[tuple[int, str]]:
    args = ["pdftotext", "-enc", "UTF-8", "-f", str(first)] + (["-l", str(last)] if last else [])
    text = subprocess.run(args + [str(path), "-"], capture_output=True, text=True, check=True).stdout
    return [(first + i, page) for i, page in enumerate(text.split("\f")) if page.strip()]


def _clean(text: str) -> str:
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"---\n.*?\n---\n", " ", text, count=1, flags=re.S)  # astro/markdown front matter
    text = re.sub(r"<[^>]+>|\{[^{}\n]{0,200}\}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


# Textbook (Euzenat & Shvaiko, Ontology Matching, 2007; PDF page numbers): every unit from the Introduction
# (PDF p9) to the end of the References (p321) is used: Introduction, Parts I-V (Chapters 1-11), Appendices A-C and
# References. Dropped: front matter (p1-8: cover, copyright, contents) and the Index (p322-332, term/page lists).
TEXTBOOK_PAGES = (9, 321)
TEXTBOOK_UNITS = 16  # Introduction + 11 chapters + 3 appendices + References
_UNIT_NUMBER = re.compile(r"^(\d{1,2}|[A-C])$")


def textbook_unit(page_text: str, expected: list[str]) -> str | None:
    """Unit label if this page starts the next expected unit, else None.

    Units must appear in book order (Introduction, 1..11, A..C, References). This rejects running headers,
    which print a page number followed by "<chapter> <title>" on even pages.
    """
    lines = [" ".join(x.split()) for x in page_text.splitlines() if x.strip()][:3]
    if not lines or not expected:
        return None
    nxt = expected[0]
    if nxt in ("Introduction", "References"):
        return nxt if lines[0] == nxt else None
    if lines[0] == nxt and len(lines) > 1 and not lines[1][:1].isdigit():
        kind = "Chapter" if nxt.isdigit() else "Appendix"
        return f"{kind} {nxt}: {lines[1]}"
    return None


def document_text(path: Path) -> list[tuple[str, str]]:
    """(section label, text) blocks for one source file. Labels "unit|page" start a new passage per unit."""
    transcriptions = {_file_key(k): v for k, v in TRANSCRIPTIONS.items()}
    if _file_key(path.name) in transcriptions:
        return [("transcription", (ROOT / transcriptions[_file_key(path.name)]).read_text())]
    suffix = path.suffix.lower()
    if suffix in BINARY_SUFFIXES:
        return []
    if suffix == ".pdf":
        pages = _pdf_pages(path)
        if path.name.startswith("Ontology Matching Textbook"):
            first, last = TEXTBOOK_PAGES
            expected = ["Introduction", *map(str, range(1, 12)), "A", "B", "C", "References"]
            out, unit = [], "Front matter"
            for n, t in pages:
                if not first <= n <= last:
                    continue
                found = textbook_unit(t, expected)
                if found:
                    unit = found
                    expected.pop(0)
                head = " ".join(t.split())[:40]
                if head.startswith("Part ") and len(t.strip()) < 120:  # Part divider pages carry only a title
                    continue
                out.append((f"{unit}|p{n}", t))
            if expected:
                raise ValueError(f"textbook units not found: {expected}")
            return out
        return [(f"p{n}", t) for n, t in pages]
    if suffix in (".html", ".htm"):
        tree = lxml_html.fromstring(path.read_bytes())
        for bad in tree.xpath("//script|//style|//nav|//footer"):
            bad.getparent().remove(bad)
        return [("page", " ".join(tree.text_content().split()))]
    return [("file", _clean(path.read_text(errors="ignore")))]


def passages(path: Path, size: int = 1500) -> list[dict]:
    """Passages of about `size` characters that never cross a unit (chapter) boundary.

    Every passage is training material. The closed-book test holds out questions, not passages (datasource.py),
    so no part of a source is withheld from training.
    """
    source = relative_to_raw(path)
    out, buffer, section, unit = [], "", "", ""

    def flush():
        nonlocal buffer
        text = buffer.strip()
        if len(text) >= 200:
            pid = hashlib.sha1(f"{source}|{len(out)}|{text[:80]}".encode()).hexdigest()[:16]
            out.append({"id": pid, "source": source, "unit": unit or section, "section": section, "text": text,
                        "split": "train"})
        buffer = ""

    for label, text in document_text(path):
        label_unit = label.split("|")[0] if "|" in label else ""
        if label_unit != unit and buffer:
            flush()
        if label_unit != unit or not section:
            section = label
        unit = label_unit
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = " ".join(paragraph.split())
            if not paragraph:
                continue
            if len(buffer) + len(paragraph) > size and buffer:
                flush()
                section = label
            buffer += paragraph + "\n\n"
    flush()
    return out
