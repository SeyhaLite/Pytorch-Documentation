#!/usr/bin/env python3
"""Guard against example files that no page references.

Every ``.py`` file in ``docs/source/examples/`` must be embedded in at least one
Markdown page via ``literalinclude``.  An orphan file means either a broken
include path or dead code that readers will never see, so the Read the Docs
``pre_build`` job fails on it.

Usage
-----
    python tests/check_examples_referenced.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "docs" / "source" / "examples"
DOCS_DIR = REPO_ROOT / "docs" / "source"

# literalinclude examples/xyz_1.py  |  :filename: examples/xyz_1.py
_INCLUDE_RE = re.compile(r"examples/([A-Za-z0-9_\-.]+\.py)")


def main() -> int:
    if not EXAMPLES_DIR.is_dir():
        print(f"missing {EXAMPLES_DIR}", file=sys.stderr)
        return 2

    examples = {p.name for p in sorted(EXAMPLES_DIR.glob("*.py")) if p.name != "__init__.py"}
    referenced: set[str] = set()

    for page in DOCS_DIR.glob("*.md"):
        referenced.update(_INCLUDE_RE.findall(page.read_text(encoding="utf-8")))
    for page in DOCS_DIR.rglob("*.rst"):
        referenced.update(_INCLUDE_RE.findall(page.read_text(encoding="utf-8")))

    orphans = sorted(examples - referenced)
    stale = sorted(referenced - examples)

    print(f"examples on disk : {len(examples)}")
    print(f"examples in docs : {len(referenced & examples)}")

    bad = 0
    if orphans:
        bad = 1
        print("\nORPHAN examples (on disk but never included in a page):", file=sys.stderr)
        for name in orphans:
            print(f"  - examples/{name}", file=sys.stderr)
    if stale:
        bad = 1
        print("\nSTALE references (a page includes a file that does not exist):", file=sys.stderr)
        for name in stale:
            print(f"  - examples/{name}", file=sys.stderr)

    if not bad:
        print("OK: every example is referenced, every reference exists.")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
