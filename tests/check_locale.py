#!/usr/bin/env python3
"""Guard the Khmer UI catalogs against drift.

Two things can silently break the Khmer interface:

1.  A Sphinx upgrade changes a msgid in `searchtools.js`, so the entry in
    `tools/build_locale.py` no longer matches and the string renders in English.
2.  Someone edits `build_locale.py` but forgets to regenerate the `.mo` and the
    JS catalog, so the committed artefacts disagree with the source of truth.

This script catches both.  It parses the *built* `searchtools.js` (so it checks
against the Sphinx version actually in use) and compares it with the committed
`km-search-i18n.js`.

Usage
-----
    sphinx-build -b html docs/source docs/_build/html -q     # build first
    python tests/check_locale.py                             # exit 1 on drift

On Read the Docs this runs in post_build, where the output lives under
$READTHEDOCS_OUTPUT/html; _buildpath handles both locations.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from _buildpath import REPO, require_build_dir
JS_CATALOG = REPO / "docs" / "source" / "_static" / "js" / "km-search-i18n.js"
MO_FILE = REPO / "docs" / "source" / "locale" / "km" / "LC_MESSAGES" / "sphinx.mo"
GENERATOR = REPO / "tools" / "build_locale.py"

# Strings Sphinx looks up in JS. Some are read with `_()`, one with an explicit
# `Documentation.gettext(...)` call spanning several lines, so both are matched.
# One precise pattern: the call, optional whitespace (the no-results message is
# wrapped over several lines), a single double-quoted literal, then the closing
# paren. Anything looser starts matching across unrelated statements.
LOOKUP_RE = re.compile(
    r"""(?:\b_\(|Documentation\.gettext\()\s*"((?:[^"\\]|\\.)*)"\s*\)""",
    re.S,
)


def js_lookup_strings(path: Path) -> set[str]:
    src = path.read_text(encoding="utf-8")
    return {m.group(1) for m in LOOKUP_RE.finditer(src) if m.group(1).strip()}


def catalog_messages(path: Path) -> dict[str, str]:
    """Pull the `messages` object back out of the generated JS catalog."""
    src = path.read_text(encoding="utf-8")
    i = src.index("var CATALOG = ") + len("var CATALOG = ")
    depth = 0
    for j, ch in enumerate(src[i:], start=i):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(src[i:j + 1])["messages"]
    raise ValueError("could not locate the CATALOG object")


def main() -> int:
    problems: list[str] = []

    built_searchtools = require_build_dir("check_locale") / "_static" / "searchtools.js"
    if not built_searchtools.exists():
        print(f"missing {built_searchtools}", file=sys.stderr)
        return 2

    for f in (JS_CATALOG, MO_FILE, GENERATOR):
        if not f.exists():
            problems.append(f"missing {f.relative_to(REPO)}")
    if problems:
        print("\n".join(problems))
        return 1

    wanted = js_lookup_strings(built_searchtools)
    have = catalog_messages(JS_CATALOG)

    print(f"searchtools.js looks up {len(wanted)} string(s)")
    print(f"km-search-i18n.js provides {len(have)}")

    missing = sorted(wanted - have.keys())
    if missing:
        problems.append("not translated (will render in English):")
        problems.extend(f"    {m!r}" for m in missing)

    unused = sorted(have.keys() - wanted)
    if unused:
        # Not fatal: a string may be looked up by the theme or a future Sphinx.
        print(f"\nnote: {len(unused)} catalog entry/entries not looked up by this "
              f"searchtools.js (harmless, kept for other templates):")
        for u in unused:
            print(f"    {u[:70]!r}")

    # Verify the committed .mo is newer than the generator, i.e. not stale.
    if MO_FILE.stat().st_mtime < GENERATOR.stat().st_mtime:
        problems.append(
            f"{MO_FILE.relative_to(REPO)} is older than {GENERATOR.relative_to(REPO)} "
            f"— regenerate with: python tools/build_locale.py"
        )
    if JS_CATALOG.stat().st_mtime < GENERATOR.stat().st_mtime:
        problems.append(
            f"{JS_CATALOG.relative_to(REPO)} is older than {GENERATOR.relative_to(REPO)} "
            f"— regenerate with: python tools/build_locale.py"
        )

    # Every Khmer msgstr must actually contain Khmer, otherwise the entry is a
    # no-op that looks translated.
    khmer_re = re.compile(r"[\u1780-\u17ff]")
    for msgid, msgstr in have.items():
        if not khmer_re.search(msgstr):
            problems.append(f"msgstr for {msgid[:40]!r} contains no Khmer text: {msgstr!r}")

    print()
    if problems:
        print("\n".join(problems))
        print("\nFAIL: Khmer catalog drift detected", file=sys.stderr)
        return 1
    print("OK: every JS-side search string is translated and the catalogs are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
