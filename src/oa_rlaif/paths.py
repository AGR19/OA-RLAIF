"""Portable locations. Everything is relative to the RLAIF root or set by environment variables.

Nothing here may contain a machine-specific absolute path; tests/test_no_absolute_paths.py enforces it.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _dir(variable: str, default: str) -> Path:
    value = os.environ.get(variable) or default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def raw_dir() -> Path:
    """The supplied `Raw Ontology Folders` directory."""
    return _dir("OA_RAW_DIR", "../OA Finetune Small Model/datasets/Raw Ontology Folders")


def bioml_dir() -> Path:
    """Bio-ML 2026 download (refs, candidate lists, NCIT/DOID/FMA) fetched by scripts/fetch_bioml.py."""
    return _dir("OA_BIOML_DIR", "../data/external/bio-ml")


def papers_dir() -> Path:
    """Extra datasource papers (the user's `Matching Surveys` folder): knowledge passages and cited literature."""
    return _dir("OA_PAPERS_DIR", "data/external/matching_surveys")


def data_dir() -> Path:
    return _dir("OA_DATA_DIR", "data")


def out_dir() -> Path:
    return _dir("OA_OUT_DIR", "outputs")


def config_dir() -> Path:
    return ROOT / "configs"


def relative_to_raw(path: Path) -> str:
    """Provenance is stored relative to the raw folder so manifests port across machines."""
    try:
        return Path(path).resolve().relative_to(raw_dir()).as_posix()
    except ValueError:
        try:
            return "bio-ml/" + Path(path).resolve().relative_to(bioml_dir()).as_posix()
        except ValueError:
            try:
                return "matching_surveys/" + Path(path).resolve().relative_to(papers_dir()).as_posix()
            except ValueError:
                try:
                    return Path(path).resolve().relative_to(data_dir()).as_posix()  # e.g. external/cyber/uco.owl
                except ValueError:
                    return Path(path).name
