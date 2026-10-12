"""JSON files that several processes update at once: one training process per GPU, plus the dashboard.

update() holds an exclusive lock (fcntl) for the whole read-modify-write, so no process loses another's change;
write_atomic() writes to a temporary file whose name is unique to the writer and then renames it, so a reader
never sees half a file and two writers never collide on one temporary name. (A shared temporary name once
crashed a 3-GPU dry run on Vast.ai.) Standard library only: the cost calculator uses it before anything is
installed.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path


def read(path, default=None):
    """The file's JSON, or `default` when it does not exist yet."""
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write_atomic(path, data, indent: int = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(data, indent=indent))
    os.replace(temporary, path)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def update(path, change: Callable[[dict], object], default: Callable[[], dict] = dict) -> dict:
    """Locked read-modify-write: `change` edits the current content (or `default()`) in place; whatever it returns
    is ignored (so `lambda d: d.setdefault(...)` cannot replace the file by accident). Returns what was written."""
    with locked(path):
        data = read(path)
        data = default() if data is None else data
        change(data)
        write_atomic(path, data)
        return data


def is_main_process() -> bool:
    """Process 0 of a multi-GPU launch (accelerate/torchrun set RANK and LOCAL_RANK), or a single process."""
    return os.environ.get("RANK", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0"
