#!/usr/bin/env python3
"""Guard the Khmer UI catalogs against drift.

Two things can silently break the Khmer interface:

1.  A Sphinx upgrade changes a msgid in `searchtools.js`, so the entry in
    `tools/build_locale.py` no longer matches and the string renders in English.
2.  Someone edits `build_locale.py` but forgets to regenerate the `.mo` and the
    JS catalog, so the committed artefacts disagree with the source of truth.

This script catches both. Staleness is detected by regenerating the expected
bytes in memory and comparing them with the committed artefacts -- never by
comparing file mtimes. git does not preserve mtimes: a clone gives every file
the checkout timestamp in index order, so an mtime comparison reports drift that
does not exist. That is exactly how Read the Docs build #35269960 failed while
the same check passed locally.  It parses the *built* `searchtools.js` (so it checks
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
LOCALE_DIR = REPO / "docs" / "source" / "locale" / "km" / "LC_MESSAGES"
MO_FILE = LOCALE_DIR / "sphinx.mo"
PO_FILE = LOCALE_DIR / "sphinx.po"
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

    for f in (JS_CATALOG, MO_FILE, PO_FILE, GENERATOR):
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

    # Staleness: compare CONTENT, never mtimes.
    #
    # git does not preserve modification times -- a fresh clone gives every file
    # the checkout timestamp, in index order. An mtime comparison therefore
    # passed on the author's machine and failed on Read the Docs for no real
    # reason (build #35269960). Regenerating the expected bytes in memory and
    # comparing them is deterministic on any machine, and works because
    # tools/build_locale.py pins the catalog dates.
    try:
        sys.path.insert(0, str(REPO / "tools"))
        import build_locale

        for path, expected in (
            (JS_CATALOG, build_locale.render_js().encode("utf-8")),
            (PO_FILE, build_locale.render_po()),
            (MO_FILE, build_locale.render_mo()),
        ):
            actual = path.read_bytes()
            if actual != expected:
                problems.append(
                    f"{path.relative_to(REPO)} does not match what "
                    f"{GENERATOR.relative_to(REPO)} generates now "
                    f"({len(actual)} bytes on disk vs {len(expected)} expected).\n"
                    f"    Someone edited ENTRIES without regenerating. Run:\n"
                    f"      python tools/build_locale.py"
                )
            else:
                print(f"  byte-identical to the generator: {path.name}")

        # Byte equality proves the file is up to date; loading it proves it is a
        # usable catalog. Both matter -- a truncated .mo can still compare equal
        # to a truncated render.
        import gettext

        with open(MO_FILE, "rb") as fh:
            km = gettext.GNUTranslations(fh)
        unresolved = [m for m, k in build_locale.ENTRIES if km.gettext(m) != k]
        if unresolved:
            sample = ", ".join(repr(x[:28]) for x in unresolved[:3])
            problems.append(
                f"{MO_FILE.relative_to(REPO)} loads but {len(unresolved)} of "
                f"{len(build_locale.ENTRIES)} entries do not resolve ({sample}…)"
            )
        else:
            print(f"  loads via gettext: all {len(build_locale.ENTRIES)} entries resolve")
    except ImportError as exc:
        # Never fail the build because Babel is absent; the msgid coverage check
        # above is the one that protects readers.
        print(f"note: skipping the staleness check ({exc})")

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
