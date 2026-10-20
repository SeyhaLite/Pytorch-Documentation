#!/usr/bin/env python3
"""Resolve the Sphinx HTML output directory, locally and on Read the Docs.

Both post-build checks need to read the *built* site, and the two environments
put it in different places:

    local            docs/_build/html
    Read the Docs    $READTHEDOCS_OUTPUT/html

Hard-coding the local path is what broke the first RTD build: the checks ran in
post_build, found no docs/_build, and failed the whole build even though Sphinx
itself had succeeded.

Shared by tests/check_locale.py and tests/check_ui_strings.py so the two checks
can never disagree about where the output lives.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL_BUILD = REPO / "docs" / "_build" / "html"

BUILD_CMD = "sphinx-build -b html docs/source docs/_build/html -q"


def build_dir() -> Path | None:
    """Return the HTML output directory, or None when nothing has been built.

    Read the Docs is detected by the READTHEDOCS_OUTPUT environment variable,
    which points at the directory the built site is collected from.
    """
    rtd = os.environ.get("READTHEDOCS_OUTPUT")
    if rtd:
        # On Read the Docs this is the ONLY correct location. Deliberately no
        # fallback to docs/_build: silently reading a stale local build would
        # let a broken RTD build pass its own post-build checks.
        candidate = Path(rtd) / "html"
        return candidate if candidate.is_dir() else None

    return LOCAL_BUILD if LOCAL_BUILD.is_dir() else None


def require_build_dir(check_name: str) -> Path:
    """Like build_dir(), but exits 2 with an actionable message."""
    found = build_dir()
    if found is not None:
        return found

    where = os.environ.get("READTHEDOCS_OUTPUT")
    print(f"{check_name}: cannot find the built HTML site.", file=sys.stderr)
    if where:
        print(f"  READTHEDOCS_OUTPUT is set to {where} but {where}/html does not "
              f"exist, so the Sphinx build has not produced output yet.",
              file=sys.stderr)
        print("  These checks must run in post_build, after Sphinx.", file=sys.stderr)
    else:
        print(f"  build it first:\n    {BUILD_CMD}", file=sys.stderr)
    raise SystemExit(2)
