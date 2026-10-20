#!/usr/bin/env python3
"""Detect broken fenced-block nesting in every Markdown file of the repo.

Why this exists
---------------
CommonMark closes a fenced block at the first following line of backticks that
has NO info string and is AT LEAST as long as the opening fence.  Two
consequences break documents that mix prose with code:

  * A directive opened with ``` that wraps a ```python block ends at the code
    block's own closing fence.  Everything after it -- the rest of the
    admonition plus every following section -- leaks out as prose, and Sphinx
    fails with `Lexing literal_block ... resulted in an error`, so
    `sphinx-build -W` exits non-zero.

  * The leftover bare fence then unbalances the rest of the file, so the damage
    spreads well past the block that caused it.

Detection is reliable; automatic repair is NOT.  When a block is truncated the
tool cannot tell which of the two candidate bare fences the author meant as the
outer closer, and guessing rewrites the wrong line and silently corrupts the
file.  This tool therefore only reports, with enough context to fix it by hand
in a few seconds.  See CONTRIBUTING.md rule 6.

Usage
-----
    python tools/check_markdown.py             # report; exit 1 on any problem
    python tools/check_markdown.py --quiet     # only the names of bad files
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FENCE_RE = re.compile(r"^(?P<fence>`{3,})(?P<rest>.*)$")
DIRECTIVE_RE = re.compile(r"^\{(?P<name>[\w-]+)\}(?P<args>.*)$")

SKIP_DIRS = {".git", "_build", "node_modules", ".venv", "__pycache__", "dist", "build"}


def markdown_files() -> list[Path]:
    """Every Markdown file in the repository.

    docs/source/*.md is rendered by Sphinx/MyST; the root README.md and
    CONTRIBUTING.md are rendered by GitHub.  Both follow the same CommonMark
    fence rules, so both are checked.
    """
    return [
        path for path in sorted(REPO.rglob("*.md"))
        if not SKIP_DIRS.intersection(path.parts)
    ]


def closing_index(lines: list[str], start: int, n: int) -> int | None:
    """Index of the fence that closes a block opened at `start` with `n` backticks."""
    for j in range(start + 1, len(lines)):
        m = FENCE_RE.match(lines[j])
        if m and not m.group("rest").strip() and len(m.group("fence")) >= n:
            return j
    return None


def truncated_openers(lines: list[str], open_i: int, close_i: int) -> list[int]:
    """Opening fences inside the block that never close before `close_i`.

    A non-empty result means the outer fence was too short: a nested block's own
    closing fence was consumed as the outer closer.
    """
    bad: list[int] = []
    k = open_i + 1
    while k < close_i:
        m = FENCE_RE.match(lines[k])
        if m and m.group("rest").strip():
            n2 = len(m.group("fence"))
            closer = closing_index(lines, k, n2)
            if closer is None or closer >= close_i:
                bad.append(k)
                k += 1
                continue
            k = closer + 1
            continue
        k += 1
    return bad


def scan(path: Path) -> list[str]:
    """Return human-readable problems for one file (empty list == healthy)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    rel = path.relative_to(REPO).as_posix()
    pad = " " * len(rel)
    problems: list[str] = []

    i = 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        # Only lines with an info string open a block; a bare fence here is
        # either content of an already-consumed block or leftover damage that
        # the truncation report below already explains.
        if not m or not m.group("rest").strip():
            i += 1
            continue

        n = len(m.group("fence"))
        rest = m.group("rest").strip()
        dm = DIRECTIVE_RE.match(rest)
        label = dm.group("name") if dm else rest[:28]
        close = closing_index(lines, i, n)

        if close is None:
            problems.append(f"{rel}:{i + 1}: unterminated {n}-backtick fence ({label})")
            i += 1
            continue

        bad = truncated_openers(lines, i, close)
        if bad:
            inner = ", ".join(str(b + 1) for b in bad)
            problems.append(
                f"{rel}:{i + 1}: `{label}` opens with {n} backticks but the nested\n"
                f"{pad}   block(s) at line(s) {inner} never close inside it.\n"
                f"{pad}   CommonMark closed the outer fence at line {close + 1}, so everything\n"
                f"{pad}   after that leaks out as prose.\n"
                f"{pad}   Fix: use {n + 1} backticks for BOTH the opening fence and the\n"
                f"{pad}   intended closing fence; leave the nested fence(s) at {n}."
            )

        if dm and dm.group("name") == "admonition" and "`" in dm.group("args"):
            problems.append(
                f"{rel}:{i + 1}: admonition title contains a backtick "
                f"({dm.group('args').strip()[:36]!r}).\n"
                f"{pad}   MyST parses the title as inline markup and the fence state\n"
                f"{pad}   machine desynchronises. Write API names in titles without backticks."
            )

        i = close + 1

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="print only the names of files with problems")
    args = ap.parse_args()

    files = markdown_files()
    problems: list[str] = []
    bad_files: list[str] = []

    for path in files:
        found = scan(path)
        if found:
            bad_files.append(path.relative_to(REPO).as_posix())
            problems.extend(found)

    if args.quiet:
        for name in bad_files:
            print(name)
    else:
        for p in problems:
            print(p)
        print(f"\nscanned {len(files)} Markdown file(s): "
              f"{len(files) - len(bad_files)} clean, {len(bad_files)} with problems")

    if problems:
        print("\nreport-only tool: fix the fences by hand (CONTRIBUTING.md rule 6).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
