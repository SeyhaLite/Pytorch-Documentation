#!/usr/bin/env python3
"""Lint and repair MyST fence nesting in docs/source/*.md.

Why this exists
---------------
A CommonMark fence is terminated by the first closing fence of the SAME length.
An admonition opened with ``` that wraps a ```python code block therefore ends
at the code block's own fence: every following section collapses into one
literal_block, Pygments fails to lex the Khmer prose, and
`sphinx-build -W --keep-going` exits non-zero.

Two rules keep pages intact:

  Rule 1  An outer fence must be strictly longer than any fence it wraps.
          An admonition containing a code block uses ```` (four backticks).

  Rule 2  Admonition titles are plain text.  A backtick in a title is parsed as
          inline markup and desynchronises the fence state machine.

This tool validates both.  `--check` only reports; the default mode additionally
repairs Rule 1 by lengthening the outer delimiter (content lines are never
rewritten, so no text can be lost).

Usage
-----
    python tools/fix_myst_fences.py            # repair + report
    python tools/fix_myst_fences.py --check    # CI mode: exit 1 on any problem
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs" / "source"

FENCE_RE = re.compile(r"^(?P<fence>`{3,})(?P<rest>.*)$")
DIRECTIVE_RE = re.compile(r"^\{(?P<name>[\w-]+)\}(?P<args>.*)$")


def fence_lines(lines: list[str]) -> list[tuple[int, int, str]]:
    """(index, backtick_count, rest) for every fence line, in order."""
    out = []
    for i, line in enumerate(lines):
        m = FENCE_RE.match(line)
        if m:
            out.append((i, len(m.group("fence")), m.group("rest")))
    return out


def parse_stack(lines: list[str]) -> tuple[list[dict], list[str]]:
    """Pair fences with an explicit stack.

    A bare fence of length N closes the innermost open fence of length N.
    A fence with an info string / directive opens a new block.  Anything still
    open at EOF is reported as unterminated.
    """
    open_stack: list[dict] = []
    closed: list[dict] = []
    errors: list[str] = []

    for i, n, rest in fence_lines(lines):
        if rest.strip():                       # an opening fence
            dm = DIRECTIVE_RE.match(rest.strip())
            open_stack.append({
                "open": i, "n": n,
                "name": dm.group("name") if dm else None,
                "title": dm.group("args").strip() if dm else rest.strip(),
                "inner": [],
            })
            continue

        # A bare fence of length N closes the innermost open fence whose length
        # is >= N (CommonMark: a closing fence must be at least as long as the
        # opening one, so ``` terminates a ```` directive).
        for pos in range(len(open_stack) - 1, -1, -1):
            if open_stack[pos]["n"] >= n:
                block = open_stack.pop(pos)
                block["close"] = i
                # every block still open wrapped this one
                for outer in open_stack:
                    outer["inner"].append(i)
                closed.append(block)
                break
        else:
            errors.append(f"line {i + 1}: bare {n}-backtick fence closes nothing")

    for block in open_stack:
        errors.append(
            f"line {block['open'] + 1}: unterminated {block['n']}-backtick fence "
            f"({block['name'] or block['title'][:24]})"
        )
    return sorted(closed, key=lambda b: b["open"]), errors


def report(lines: list[str]) -> tuple[list[dict], list[str]]:
    """Blocks violating Rule 1 (need a longer outer fence) + all warnings."""
    blocks, errors = parse_stack(lines)
    needs = []
    for b in blocks:
        where = f"line {b['open'] + 1}"
        label = b["name"] or (b["title"][:24] if b["title"] else "code")
        if b["inner"] and b["n"] <= 3:
            needs.append(b)
        if b["name"] == "admonition" and "`" in b["title"]:
            errors.append(f"{where}: admonition title contains a backtick ({b['title'][:36]!r})")
        if not b["inner"] and b["n"] > 3 and b["name"] in (None, "admonition"):
            errors.append(f"{where}: {{{label}}} uses {b['n']} backticks but wraps no fence")
    return needs, errors


def repair(path: Path) -> tuple[list[str], list[str]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    log: list[str] = []

    for _ in range(8):  # lengthening one block can expose another
        needs, _ = report(lines)
        if not needs:
            break
        for b in needs:
            want = "`" * (b["n"] + 1)
            lines[b["open"]] = want + lines[b["open"]][b["n"]:]
            lines[b["close"]] = want
            log.append(f"line {b['open'] + 1}: {{{b['name'] or 'code'}}} -> {len(want)} backticks")
        path.write_text("\n".join(lines), encoding="utf-8")
        lines = path.read_text(encoding="utf-8").split("\n")

    _, warnings = report(lines)
    return log, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report only; exit 1 on problems")
    args = ap.parse_args()

    n_bad = n_repaired = 0
    for path in sorted(DOCS.glob("*.md")):
        lines = path.read_text(encoding="utf-8").split("\n")
        if args.check:
            needs, warnings = report(lines)
            msgs = [
                f"line {b['open'] + 1}: {{{b['name'] or 'code'}}} wraps {len(b['inner'])} "
                f"fence line(s) but uses {b['n']} backticks"
                for b in needs
            ]
            n_bad += len(msgs) + len(warnings)
        else:
            msgs, warnings = repair(path)
            n_repaired += len(msgs)
            n_bad += len(warnings)

        for msg in msgs:
            print(f"{'NEEDS  ' if args.check else 'FIXED  '} {path.name:<30} {msg}")
        for msg in warnings:
            print(f"WARN    {path.name:<30} {msg}")

    if args.check:
        print(f"\noutstanding problems: {n_bad}")
    else:
        print(f"\nrepaired: {n_repaired} · remaining warnings: {n_bad}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
