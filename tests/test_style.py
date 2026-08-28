"""Style rules for anything published with this project.

Em dashes are not used in documentation, comments, docstrings or help text.
"""

from pathlib import Path

import pytest

# Spelled as an escape so that this file does not trip its own rule.
EM_DASH = "\u2014"

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ("*.py", "*.md", "*.cff", "*.toml", "*.html", "*.css", "*.js")
SKIP_DIRS = {".git", ".venv", "build", "dist", "renders", "__pycache__"}


def _published_files():
    for pattern in PATTERNS:
        for path in ROOT.rglob(pattern):
            if not SKIP_DIRS.intersection(path.parts):
                yield path


@pytest.mark.parametrize("path", sorted(_published_files(), key=str), ids=str)
def test_no_em_dashes(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    offenders = [(n, line) for n, line in enumerate(lines, 1) if EM_DASH in line]
    assert not offenders, "em dash in " + "; ".join(
        f"{path.relative_to(ROOT)}:{n}: {line.strip()[:70]}" for n, line in offenders
    )
