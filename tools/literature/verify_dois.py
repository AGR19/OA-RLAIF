"""Verify literature entries: resolve DOIs, match titles, fetch citation counts and BibTeX.

Stdlib only so it runs anywhere. For each entry in papers.json:
  1. pick a DOI (given, arXiv DataCite DOI 10.48550/arXiv.<id>, or Crossref title search),
  2. resolve it at doi.org with CSL-JSON content negotiation and check the title matches,
  3. fetch BibTeX from doi.org,
  4. fetch citation counts (and any venue DOI) from the Semantic Scholar batch API.
Entries that cannot be verified are reported as dropped, never guessed.

Usage: python tools/literature/verify_dois.py [--papers PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = "oa-rlaif-literature-check/1.0 (research; no-reply)"
try:  # python.org macOS builds ship without a CA bundle; certifi supplies one when present
    import ssl

    import certifi

    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where())))
    )
except ImportError:
    pass
S2_FIELDS = "title,year,venue,citationCount,externalIds,publicationVenue"


def http(url: str, *, accept: str = "application/json", data: bytes | None = None, retries: int = 5) -> bytes:
    headers = {"Accept": accept, "User-Agent": UA}
    if data is not None:
        headers["Content-Type"] = "application/json"
    delay = 2.0
    for attempt in range(retries):
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def norm(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\$\\neq\$|\\neq|!=", " not ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = text.lower().replace("‐", "-")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def title_ok(expected: str, found: str, loose: bool) -> tuple[bool, float]:
    a, b = norm(expected), norm(found)
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if loose:
        return (a in b or b in a or ratio >= 0.6), ratio
    return (ratio >= 0.85 or (len(a) > 20 and (a in b or b in a))), ratio


def csl(doi: str) -> dict:
    raw = http(f"https://doi.org/{urllib.parse.quote(doi, safe='/')}", accept="application/vnd.citationstyles.csl+json")
    return json.loads(raw)


def bibtex(doi: str) -> str:
    raw = http(f"https://doi.org/{urllib.parse.quote(doi, safe='/')}", accept="application/x-bibtex; charset=utf-8")
    return raw.decode("utf-8", "replace").strip()


def crossref_search(title: str) -> list[dict]:
    query = urllib.parse.urlencode({"query.bibliographic": title, "rows": 5, "select": "DOI,title,container-title,issued,type"})
    payload = json.loads(http(f"https://api.crossref.org/works?{query}"))
    return payload.get("message", {}).get("items", [])


def first_title(record: dict) -> str:
    title = record.get("title", "")
    return title[0] if isinstance(title, list) and title else (title or "")


def year_of(record: dict) -> int | None:
    for key in ("issued", "published", "published-print", "published-online", "created"):
        parts = (record.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def semantic_scholar(ids: list[str]) -> list[dict | None]:
    out: list[dict | None] = []
    for start in range(0, len(ids), 400):
        chunk = ids[start:start + 400]
        body = json.dumps({"ids": chunk}).encode()
        raw = http(f"https://api.semanticscholar.org/graph/v1/paper/batch?fields={S2_FIELDS}", data=body, retries=8)
        out.extend(json.loads(raw))
        time.sleep(1.5)
    return out


def verify(entry: dict, log) -> dict:
    result = dict(entry)
    result.update(status="dropped", reason="", doi_verified=None, found_title=None, year=None, container=None)
    if entry.get("no_doi"):
        result.update(status="no_doi", reason=entry.get("note") or "grey literature or CEUR proceedings without DOI; URL recorded")
        return result
    candidates: list[str] = []
    if entry.get("doi"):
        candidates.append(entry["doi"])
    if entry.get("arxiv"):
        candidates.append(f"10.48550/arXiv.{entry['arxiv']}")
    if not candidates:
        result["reason"] = "no DOI found by Crossref title search"
        try:
            for item in crossref_search(entry["title"]):
                ok, _ = title_ok(entry["title"], first_title(item), entry.get("title_match") == "loose")
                if ok:
                    candidates.append(item["DOI"])
                    break
        except Exception as error:  # network or parse failure is reported, not hidden
            result["reason"] = f"crossref search failed: {error}"
    for doi in candidates:
        try:
            record = csl(doi)
        except Exception as error:
            result["reason"] = f"doi.org failed for {doi}: {error}"
            continue
        found = record.get("title") or ""
        found = found[0] if isinstance(found, list) else found
        ok, ratio = title_ok(entry["title"], found, entry.get("title_match") == "loose")
        if not ok:
            result["reason"] = f"title mismatch for {doi} (ratio {ratio:.2f}): {found!r}"
            continue
        container = record.get("container-title") or record.get("publisher") or ""
        container = container[0] if isinstance(container, list) and container else container
        result.update(status="verified", doi_verified=doi, found_title=found, year=year_of(record), container=container, reason="")
        try:
            result["bibtex"] = bibtex(doi)
        except Exception as error:
            result["bibtex"] = ""
            result["reason"] = f"bibtex fetch failed: {error}"
        break
    log(f"{result['status']:>9}  {entry['id']:<28} {result.get('doi_verified') or result['reason'][:70]}")
    time.sleep(0.25)
    return result


def s2_id(result: dict) -> str | None:
    if result.get("arxiv"):
        return f"ARXIV:{result['arxiv']}"
    if result.get("doi_verified"):
        return f"DOI:{result['doi_verified']}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=Path, default=HERE / "papers.json")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    entries = json.loads(args.papers.read_text())
    snapshot = dt.date.today().isoformat()
    cache_path = args.out / "verify_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    results = []
    for entry in entries:
        key = json.dumps(entry, sort_keys=True)
        if key in cache and cache[key]["status"] in ("verified", "no_doi"):
            results.append(cache[key])
            print(f"   cached  {entry['id']}", flush=True)
            continue
        result = verify(entry, lambda line: print(line, flush=True))
        cache[key] = result
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))
        results.append(result)

    ids = [s2_id(r) for r in results]
    lookup = [i for i in ids if i]
    print(f"semantic scholar batch: {len(lookup)} ids", flush=True)
    try:
        s2 = semantic_scholar(lookup)
    except Exception as error:
        print(f"semantic scholar failed: {error}", file=sys.stderr)
        s2 = [None] * len(lookup)
    s2_map = dict(zip(lookup, s2))
    for result, key in zip(results, ids):
        paper = s2_map.get(key) if key else None
        result["citations"] = paper.get("citationCount") if paper else None
        result["s2_venue"] = (paper.get("venue") if paper else None) or None
        venue_doi = ((paper or {}).get("externalIds") or {}).get("DOI")
        if venue_doi and (result.get("doi_verified") or "").lower().startswith("10.48550/") and not venue_doi.lower().startswith("10.48550/"):
            try:
                record = csl(venue_doi)
                found = record.get("title") or ""
                found = found[0] if isinstance(found, list) else found
                if title_ok(result["title"], found, False)[0]:
                    result["venue_doi"] = venue_doi
            except Exception:
                pass
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verified.json").write_text(json.dumps({"snapshot": snapshot, "papers": results}, indent=2, ensure_ascii=False))
    bib = [r["bibtex"] for r in results if r.get("bibtex")]
    (args.out / "references.bib").write_text("\n\n".join(bib) + "\n")
    counts = {s: sum(r["status"] == s for r in results) for s in ("verified", "no_doi", "dropped")}
    print(json.dumps({"snapshot": snapshot, **counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
