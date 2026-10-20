#!/usr/bin/env python3
"""Fail if any theme/Sphinx UI string still renders in English.

The site is Khmer.  Sphinx resolves interface strings from a compiled catalog,
so a msgid that is missing, misspelled, or dropped when the catalog is
regenerated silently falls back to English -- the build still succeeds and the
page still looks fine at a glance.  That happened once already with
"Please activate JavaScript to enable the search functionality.", so the check
is now permanent.

The scan deliberately strips <script> blocks, highlighted code and <code>
spans first: those legitimately contain English identifiers and API names,
which must never be translated.

Usage
-----
    sphinx-build -b html docs/source docs/_build/html -q
    python tests/check_ui_strings.py            # exit 1 on any English UI string

On Read the Docs this runs in post_build, where the output lives under
$READTHEDOCS_OUTPUT/html; _buildpath handles both locations.
"""

from __future__ import annotations

import collections
import glob
import re
import sys
from pathlib import Path

from _buildpath import require_build_dir

# Interface strings emitted by sphinx_rtd_theme and Sphinx itself.
UI_STRINGS = [
    "Please activate JavaScript to enable the search functionality.",
    "Search Results",
    "Preparing search",
    "Searching",
    "Search docs",
    "Edit on GitHub",
    "Edit on GitLab",
    "Edit on Bitbucket",
    "View page source",
    "Navigation menu",
    "Mobile navigation menu",
    "Page navigation",
    "Sequential page navigation",
    "Last updated on",
    "General Index",
    "Search Page",
    "Full index on one page",
    "This Page",
    "Next topic",
    "Previous topic",
    "Expand sidebar",
    "Collapse sidebar",
    "Quick search",
    "About these documents",
    "Search within",
    "Your search did not match",
    "Permalink to this",
    "Hide Search Matches",
    "Table of Contents",
    ">Next<",
    ">Previous<",
    ">Index<",
    ">Search<",
    ">Footer<",
]

# Khmer strings that MUST appear, proving the catalog is actually loaded.
REQUIRED_KHMER = [
    "ស្វែងរកឯកសារ",                       # search field label
    "ពណ៌បង្ហាញ",                            # theme control caption
    "ម៉ឺនុយរុករក",                         # Navigation menu
    "កែនៅលើ GitHub",                    # Edit on GitHub
    "ធ្វើបច្ចុប្បន្នភាពចុងក្រោយ",           # Last updated on
    "រក្សាសិទ្ធិ",                            # Copyright
    "សូមបើក JavaScript",                 # noscript fallback (search page)
]


def strip_non_ui(html: str) -> str:
    """Remove script, highlighted code and inline code before scanning."""
    s = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    s = re.sub(r'<div class="highlight">.*?</div>\s*</div>', "", s, flags=re.S)
    s = re.sub(r"<pre[^>]*>.*?</pre>", "", s, flags=re.S)
    s = re.sub(r"<code[^>]*>.*?</code>", "", s, flags=re.S)
    return s


def main() -> int:
    build = require_build_dir("check_ui_strings")
    pages = sorted(glob.glob(str(build / "*.html")))
    if not pages:
        print(f"no built pages under {build}", file=sys.stderr)
        return 2

    english = collections.Counter()
    where: dict[str, set[str]] = collections.defaultdict(set)
    khmer_found = collections.Counter()

    for path in pages:
        body = strip_non_ui(Path(path).read_text(encoding="utf-8"))
        name = Path(path).name
        for needle in UI_STRINGS:
            n = body.count(needle)
            if n:
                english[needle] += n
                where[needle].add(name)
        for needle in REQUIRED_KHMER:
            if needle in body:
                khmer_found[needle] += 1

    print(f"scanned {len(pages)} built page(s)")

    problems: list[str] = []

    if english:
        problems.append("English UI strings still rendered:")
        for needle, n in english.most_common():
            problems.append(f"    {n:>3}x {needle!r} in {sorted(where[needle])[:2]}")

    missing = [k for k in REQUIRED_KHMER if khmer_found[k] == 0]
    if missing:
        problems.append("Khmer strings absent from every page (catalog not loaded?):")
        problems.extend(f"    {m!r}" for m in missing)

    print(f"\nKhmer strings confirmed present:")
    for k in REQUIRED_KHMER:
        print(f"  {'OK  ' if khmer_found[k] else 'MISS'} {k}  ({khmer_found[k]} page(s))")

    print()
    if problems:
        print("\n".join(problems))
        print("\nFAIL: untranslated UI strings. "
              "Edit tools/build_locale.py, then run:\n"
              "  python tools/build_locale.py && "
              "sphinx-build -b html docs/source docs/_build/html -q",
              file=sys.stderr)
        return 1
    print("OK: no English UI strings remain and the Khmer catalog is loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
