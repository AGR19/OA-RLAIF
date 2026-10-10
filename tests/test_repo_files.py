"""Every source, test, script and config file must be committable: an ignore rule must never hide code from GitHub.

(An unanchored `data/` rule once hid the whole src/oa_rlaif/data package; a fresh clone then failed with
"No module named 'oa_rlaif.data'".)
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not shutil.which("git") or not (ROOT / ".git").exists(), reason="not a git checkout")
def test_no_code_is_git_ignored():
    files = [p for d in ("src", "tests", "scripts", "configs", "docker") for p in (ROOT / d).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and ".egg-info" not in str(p)
             and p.suffix in {".py", ".sh", ".yaml", ".yml", ".toml", ".om", ""} and p.name != ".DS_Store"]
    assert files
    result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT, input="\n".join(str(f.relative_to(ROOT)) for f in files),
                            capture_output=True, text=True)
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert not ignored, f"git ignores these code files: {ignored}"
