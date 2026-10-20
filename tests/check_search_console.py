#!/usr/bin/env python3
"""Guard the Google Search Console verification artefacts.

Google offers two ways to prove ownership of
https://pytorch-documentation.readthedocs.io/ , and this project ships both:

  * **HTML file** -- `docs/source/extra/google59471f16170f14f0.html`, copied to
    the build output root by `html_extra_path`.
  * **Meta tag** -- `<meta name="google-site-verification" ...>` in `<head>`,
    emitted on every page by `_templates/layout.html` from
    `google_site_verification` in conf.py.

Why both, and why the meta tag is the one that actually works
-------------------------------------------------------------
Read the Docs mounts a build under `/<language>/<version>/`, and this project
sets `language = "km"`. So the site lives at `/km/latest/`, and the file Google
asks for at the domain root 404s no matter what `html_extra_path` contains:

    https://pytorch-documentation.readthedocs.io/google59471f16170f14f0.html   404
    https://pytorch-documentation.readthedocs.io/km/latest/google59471f16170f14f0.html   200

The meta tag survives this because the root URL redirects to `/km/latest/`,
whose index.html carries the tag. The file is kept for the case where the
property is registered as a URL-prefix at `/km/latest/`.

Why a test
----------
Google re-checks periodically. If either artefact disappears the property can be
silently un-verified weeks later, and nothing in a normal build notices: Sphinx
still exits 0 and every page still renders.

Usage
-----
    sphinx-build -b html docs/source docs/_build/html -q
    python tests/check_search_console.py         # exit 1 on any problem
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

from _buildpath import REPO, require_build_dir

EXTRA_SRC = REPO / "docs" / "source" / "extra"
CONF = REPO / "docs" / "source" / "conf.py"
LAYOUT = REPO / "docs" / "source" / "_templates" / "layout.html"

# Files that html_extra_path must place at the output root.
ROOT_FILES = {
    "google59471f16170f14f0.html": "Google Search Console HTML-file token",
}

# The meta tag Google issued. Must appear verbatim.
META_NAME = "google-site-verification"
META_CONTENT = "L2d5Vp4txgR8OhRYpJvFenCBS0BqDfbfMYt0SG7rGHw"
META_TAG = f'<meta name="{META_NAME}" content="{META_CONTENT}" />'


def check_source(problems: list[str]) -> None:
    """Everything that must be true before a build can possibly be right."""
    for name, why in ROOT_FILES.items():
        if not (EXTRA_SRC / name).is_file():
            problems.append(
                f"missing docs/source/extra/{name} ({why}).\n"
                f"    Google re-checks periodically; without it the property can be\n"
                f"    silently un-verified."
            )

    conf_text = CONF.read_text(encoding="utf-8") if CONF.is_file() else ""
    if "html_extra_path" not in conf_text:
        problems.append(
            "conf.py does not set html_extra_path, so docs/source/extra/ is never\n"
            "    copied and the root file will 404."
        )
    if META_CONTENT not in conf_text:
        problems.append(
            f"conf.py does not define google_site_verification = {META_CONTENT!r}.\n"
            "    Without it layout.html emits no meta tag and domain verification fails."
        )
    if LAYOUT.is_file():
        layout = LAYOUT.read_text(encoding="utf-8")
        if META_NAME not in layout:
            problems.append(
                "_templates/layout.html no longer emits the google-site-verification\n"
                "    meta tag."
            )
    else:
        problems.append(
            "_templates/layout.html is missing, so no page carries the meta tag."
        )


def check_build(build: Path, problems: list[str]) -> None:
    """Everything that must be true of the built output."""
    for name, why in ROOT_FILES.items():
        src, out = EXTRA_SRC / name, build / name
        if not out.is_file():
            problems.append(
                f"{name} is not at the build output root.\n"
                f"    On Read the Docs it is served at /km/latest/{name}, never at /{name}."
            )
        elif src.is_file() and src.read_bytes() != out.read_bytes():
            problems.append(f"{name} at the output root is not byte-identical to its source.")
        if (build / "_static" / name).exists():
            problems.append(
                f"{name} was also copied into _static/, where it would be served\n"
                f"    at /_static/{name} instead of /{name}."
            )

    # The meta tag must be inside <head>, exactly once, on every page. Google
    # requires it before the first <body>, and a duplicated tag is ambiguous.
    pages = [p for p in sorted(glob.glob(str(build / "*.html")))
             if Path(p).name not in ROOT_FILES]
    if not pages:
        problems.append("no built pages found")
        return

    good = 0
    for path in pages:
        html = Path(path).read_text(encoding="utf-8")
        name = Path(path).name
        head_end = html.find("</head>")
        body_start = html.find("<body")
        head = html[:head_end] if head_end >= 0 else ""
        count = html.count(META_NAME)

        if META_TAG not in head:
            problems.append(f"{name}: exact meta tag not found inside <head>")
        elif body_start >= 0 and html.find(META_TAG) > body_start:
            problems.append(f"{name}: meta tag appears after <body>")
        elif count != 1:
            problems.append(f"{name}: meta tag appears {count} times, expected exactly 1")
        else:
            good += 1

    print(f"meta tag verified in <head> of {good}/{len(pages)} page(s)")


def main() -> int:
    problems: list[str] = []
    check_source(problems)
    build = require_build_dir("check_search_console")
    check_build(build, problems)

    print(f"build output      : {build}")
    for name, why in ROOT_FILES.items():
        state = "OK" if (build / name).is_file() else "MISSING"
        print(f"  {state:<8} /{name}   ({why})")
    print(f"  {'OK' if META_CONTENT in (build / 'index.html').read_text(encoding='utf-8') else 'MISSING':<8} "
          f"<meta name=\"{META_NAME}\">   (on every page)")
    print("\nNote: on Read the Docs these are served under /km/latest/, not at the")
    print("      domain root, because the build is mounted at /<language>/<version>/.")

    print()
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1
    print("OK: both Search Console verification methods are in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
