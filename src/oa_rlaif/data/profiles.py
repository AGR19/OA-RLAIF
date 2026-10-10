"""Parse every ontology once into a TSV profile cache (data/profiles/<key>/), in parallel.

Large files (> BIG_BYTES) run in a single-worker pool so peak memory stays bounded on a 16 GB laptop,
while small files run concurrently. The cache key is the provenance path plus size and mtime.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..ontology.parser import PARSER_VERSION, OntologyParser
from ..paths import data_dir, relative_to_raw
from ..progress import stage

BIG_BYTES = 150_000_000


def cache_key(path: Path) -> str:
    stat = path.stat()
    raw = f"{relative_to_raw(path)}|{stat.st_size}|{stat.st_mtime_ns}|parser{PARSER_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cache_dir(path: Path) -> Path:
    return data_dir() / "profiles" / cache_key(path)


def is_cached(path: Path) -> bool:
    return (cache_dir(path) / "profile.yaml").exists()


def _parse_one(path_str: str) -> tuple[str, int, str]:
    path = Path(path_str)
    target = cache_dir(path)
    try:
        profile = OntologyParser().parse(path, relative_to_raw(path))
        OntologyParser().save(profile, target)
        return path_str, profile.metadata.get("classes", 0), ""
    except Exception as error:  # recorded and reported; a failed parse never silently disappears
        target.mkdir(parents=True, exist_ok=True)
        (target / "error.txt").write_text(f"{type(error).__name__}: {error}")
        return path_str, 0, f"{type(error).__name__}: {error}"


def build_profiles(files: list[Path], workers: int | None = None) -> dict[str, str]:
    """Parse all uncached files. Returns {path: error} for files that failed to parse."""
    todo = [f for f in files if not is_cached(f)]
    small = [f for f in todo if f.stat().st_size <= BIG_BYTES]
    big = sorted((f for f in todo if f.stat().st_size > BIG_BYTES), key=lambda f: f.stat().st_size)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    errors: dict[str, str] = {}
    with stage("parse_ontologies", total=len(todo)) as bar:
        with ProcessPoolExecutor(max_workers=1) as big_pool, ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [big_pool.submit(_parse_one, str(f)) for f in big]
            futures += [pool.submit(_parse_one, str(f)) for f in small]
            for future in as_completed(futures):
                path, _, error = future.result()
                if error:
                    errors[path] = error
                bar.advance()
    return errors


def load_profile(path: Path):
    return OntologyParser().load(cache_dir(path))


def main() -> int:
    import json
    import time

    from .families import discover, ontology_files

    files = ontology_files(discover())
    started = time.time()
    errors = build_profiles(files)
    print(json.dumps({"files": len(files), "errors": errors, "minutes": round((time.time() - started) / 60, 1)}, indent=1))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
