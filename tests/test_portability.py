"""Portability gates: no machine-specific paths, no secrets, no licensed data tracked in git."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".tex", ".txt", ".cfg", ".json", ".example"}
SKIP_DIRS = {".venv", "data", "outputs", ".git", "build", "__pycache__", ".pytest_cache", "third_party"}
# Documents that intentionally describe where the Mac copy lives may mention ~ (home-relative), never /Users.
ABSOLUTE = re.compile(r"(/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+/|C:\\\\Users\\\\)")
SECRET = re.compile(r"(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})")


def _files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not (set(path.relative_to(ROOT).parts) & SKIP_DIRS):
            if path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", ".env.example"}:
                yield path


def test_no_absolute_user_paths():
    offenders = []
    for path in _files():
        if path.name == "verified.json" or path.name == "verify_log.txt":
            continue
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if ABSOLUTE.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, "machine-specific paths found:\n" + "\n".join(offenders)


def test_no_secrets():
    offenders = [str(p.relative_to(ROOT)) for p in _files() if SECRET.search(p.read_text(errors="ignore"))]
    assert not offenders, f"possible secrets in: {offenders}"


def test_gitignore_blocks_licensed_and_generated_data():
    ignore = (ROOT / ".gitignore").read_text()
    for pattern in ["data/", "outputs/", ".env", ".venv/", "*.gguf", "*.safetensors"]:
        assert pattern in ignore, pattern


def test_git_index_has_no_licensed_files():
    if not (ROOT / ".git").exists():
        return
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    banned = re.compile(r"(\.owl$|\.ttl$|\.rdf$|\.pdf$|sct2_|der2_|\.sqlite$|\.gguf$|\.safetensors$|^data/|^outputs/)")
    allowed = re.compile(r"^docs/figures/oa_rlaif_[a-z_]+\.pdf$")  # our own method figures, built from docs/figures/*.tex
    hits = [f for f in tracked.splitlines() if banned.search(f) and not allowed.match(f)]
    assert not hits, f"licensed or generated files are tracked: {hits[:20]}"
