"""Small, atomic file helpers (vendored from the parent repo's src/utils/io.py without its path guards)."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml


def _atomic(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_suffix(path.suffix + ".tmp")


@contextmanager
def atomic_open(path):
    """Write to <path>.tmp and rename on success, so a crash or a second writer never leaves a damaged file."""
    temporary = _atomic(path)
    with temporary.open("w", encoding="utf-8") as f:
        yield f
    temporary.replace(path)


@contextmanager
def exclusive_lock(path, what: str):
    """Fail fast if another process holds the lock (for example, two data builds at the same time)."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another {what} is running ({path}); wait for it to finish") from None
        yield


def read_yaml(path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def write_yaml(path, value) -> None:
    temporary = _atomic(path)
    temporary.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
    temporary.replace(path)


def write_tsv(path, columns, rows: Iterable) -> int:
    temporary = _atomic(path)
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(k, "") for k in columns] if isinstance(row, dict) else row)
            count += 1
    temporary.replace(path)
    return count


def read_tsv(path) -> Iterator[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f, delimiter="\t")


def write_jsonl(path, records: Iterable[dict]) -> int:
    temporary = _atomic(path)
    count = 0
    with temporary.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def read_jsonl(path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
