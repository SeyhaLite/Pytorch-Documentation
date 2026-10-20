#!/usr/bin/env python3
"""Guard the files that must be served from the site root.

`google59471f16170f14f0.html` is the Google Search Console ownership token for
https://pytorch-documentation.readthedocs.io/ . Google re-checks it
periodically, so if it disappears the property can be silently unverified even
though verification succeeded once. Nothing in a normal build would notice:
Sphinx exits 0, the pages render fine, and the only symptom is Search Console
dropping the site weeks later.

Sphinx only copies a file to the output ROOT if its directory is listed in
`html_extra_path`; anything under `_static/` lands in a subdirectory and would
404 at the expected URL. So this checks all three links in the chain:

  1. the file exists in docs/source/extra/
  2. conf.py actually registers html_extra_path for that directory
  3. after a build, the file is at the output root, byte-identical

Usage
-----
    sphinx-build -b html docs/source docs/_build/html -q
    python tests/check_extra_files.py            # exit 1 on any problem
"""

from __future__ import annotations

import sys
from pathlib import Path

from _buildpath import require_build_dir

REPO = Path(__file__).resolve().parent.parent
EXTRA_SRC = REPO / "docs" / "source" / "extra"
CONF = REPO / "docs" / "source" / "conf.py"

# name -> why it must stay
ROOT_FILES = {
    "google59471f16170f14f0.html": "Google Search Console ownership token",
}


def main() -> int:
    problems: list[str] = []

    # 1. present in source
    for name, why in ROOT_FILES.items():
        src = EXTRA_SRC / name
        if not src.is_file():
            problems.append(
                f"missing docs/source/extra/{name} ({why}).\n"
                f"    Google re-checks this file periodically; without it the property\n"
                f"    can be silently unverified. Restore it before deploying."
            )

    # 2. conf.py registers html_extra_path (otherwise nothing reaches the root)
    conf_text = CONF.read_text(encoding="utf-8") if CONF.is_file() else ""
    if "html_extra_path" not in conf_text:
        problems.append(
            "conf.py does not set html_extra_path, so docs/source/extra/ is never\n"
            "    copied and every root file will 404."
        )
    elif "extra" not in conf_text.split("html_extra_path", 1)[1][:200]:
        problems.append("conf.py sets html_extra_path but it does not point at extra/")

    # 3. present at the output root, byte-identical
    build = require_build_dir("check_extra_files")
    for name, why in ROOT_FILES.items():
        src, out = EXTRA_SRC / name, build / name
        if not out.is_file():
            problems.append(
                f"{name} is not at the build output root ({out}).\n"
                f"    It would 404 at https://pytorch-documentation.readthedocs.io/{name}"
            )
        elif src.is_file() and src.read_bytes() != out.read_bytes():
            problems.append(
                f"{name} at the output root differs from docs/source/extra/{name}; "
                f"the copy is not verbatim."
            )
        # A copy under _static/ would be served at the wrong URL.
        misplaced = build / "_static" / name
        if misplaced.exists():
            problems.append(
                f"{name} was also copied into _static/, where it would be served\n"
                f"    at /_static/{name} instead of /{name}."
            )

    print(f"root files checked : {len(ROOT_FILES)}")
    print(f"build output       : {build}")
    for name in ROOT_FILES:
        state = "OK" if (build / name).is_file() else "MISSING"
        print(f"  {state:<8} /{name}   ({ROOT_FILES[name]})")

    print()
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1
    print(f"OK: all {len(ROOT_FILES)} root file(s) are served from the site root, verbatim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
