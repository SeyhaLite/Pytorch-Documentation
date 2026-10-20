#!/usr/bin/env python3
"""Verify that every documented colour pair reaches WCAG AA contrast (4.5:1).

The palette lives in docs/source/_static/css/custom.css as CSS custom
properties on `html[data-theme="dark"]` and `html[data-theme="light"]`.
This script parses those two blocks, resolves each documented text/background
pair, and fails (exit 1) if anything drops below the threshold.

Run it after touching any colour:

    python tools/check_contrast.py

What gets checked, in both themes:
  * page palette  — body text, headings, muted, links, buttons, marks, chips
  * chrome        — sidebar items, captions, table headers, theme control
  * code          — every Pygments token colour on the code-block background
  * admonitions   — title bar/label pairs AND body text on the tinted body

A value is either a `--kh-*` token name (resolved from the theme block) or a
literal hex colour.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CSS = Path(__file__).resolve().parent.parent / "docs" / "source" / "_static" / "css" / "custom.css"
MIN_RATIO = 4.5


def _linear(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def tokens(css: str, selector: str) -> dict[str, str]:
    i = css.index(selector)
    j = css.index("}", i)
    return dict(re.findall(r"(--kh-[\w-]+):\s*(#[0-9a-fA-F]{3,8})", css[i:j]))


def resolve(value: str, tk: dict[str, str]) -> str:
    if value.startswith("--kh-"):
        if value not in tk:
            raise KeyError(f"{value} is not defined in this theme block")
        return tk[value]
    return value


# --------------------------------------------------------------------------- #
# Pair tables
# --------------------------------------------------------------------------- #

PAGE = [
    ("body text on page", "--kh-text", "--kh-page-bg"),
    ("body text on card", "--kh-text", "--kh-card-bg"),
    ("body text on row-alt", "--kh-text", "--kh-row-alt-bg"),
    ("body text on row-hover", "--kh-text", "--kh-row-hover-bg"),
    ("headings on page", "--kh-heading", "--kh-page-bg"),
    ("headings on card", "--kh-heading", "--kh-card-bg"),
    ("muted on card", "--kh-muted", "--kh-card-bg"),
    ("muted on page", "--kh-muted", "--kh-page-bg"),
    ("link on page", "--kh-link", "--kh-page-bg"),
    ("link on card", "--kh-link", "--kh-card-bg"),
    ("visited link on page", "--kh-link-visited", "--kh-page-bg"),
    ("code text on code bg", "--kh-code-text", "--kh-code-bg"),
    ("button text on button", "--kh-btn-text", "--kh-btn-bg"),
    ("mark text on mark bg", "--kh-mark-text", "--kh-mark-bg"),
    ("chip / kbd / samp text", "--kh-chip-text", "--kh-chip-bg"),
]

CHROME_DARK = [
    ("sidebar nav item", "#c8cfd8", "--kh-sidebar-bg"),
    ("sidebar current branch", "#dfe5ec", "#232830"),
    ("sidebar nested l2/l3", "#bcc4ce", "--kh-sidebar-bg"),
    ("sidebar active page", "#ffffff", "#2f4460"),
    ("sidebar caption", "#93a0ad", "--kh-sidebar-bg"),
    ("sidebar header link", "#f2f5f9", "--kh-sidebar-header-bg"),
    ("sidebar version text", "#b6bfca", "--kh-sidebar-header-bg"),
    ("theme toggle label", "#aeb8c4", "--kh-sidebar-bg"),
    ("theme toggle hover", "#ffffff", "--kh-sidebar-bg"),
    ("search placeholder", "--kh-faint", "--kh-input-bg"),
    ("search input text", "--kh-text", "--kh-input-bg"),
    ("table header text", "--kh-heading", "#232830"),
    ("breadcrumb text", "--kh-muted", "--kh-content-bg"),
    ("footer text", "--kh-muted", "--kh-content-bg"),
]

CHROME_LIGHT = [
    ("sidebar nav item", "#3d4753", "--kh-sidebar-bg"),
    ("sidebar caption", "#57606a", "--kh-sidebar-bg"),
    ("sidebar header link", "#ffffff", "--kh-sidebar-header-bg"),
    ("theme toggle label", "#4a5563", "--kh-sidebar-bg"),
    ("search input text", "--kh-text", "--kh-input-bg"),
    ("table header text", "--kh-heading", "#e8ecf1"),
    ("breadcrumb text", "--kh-muted", "--kh-content-bg"),
    ("footer text", "--kh-muted", "--kh-content-bg"),
]

PYGMENTS_DARK = [
    ("comment", "#8b949e"), ("keyword", "#ff7b72"), ("string", "#a5d6ff"),
    ("number", "#79c0ff"), ("name", "#d9dde4"), ("builtin", "#ffa657"),
    ("function", "#d2a8ff"), ("class", "#f0c674"), ("variable", "#ffa198"),
    ("operator", "#ff7b72"), ("decorator", "#d2a8ff"), ("constant", "#79c0ff"),
]

PYGMENTS_LIGHT = [
    ("comment", "#57606a"), ("keyword", "#a626a4"), ("string", "#0a3069"),
    ("number", "#0550ae"), ("name", "#24292f"), ("builtin", "#953800"),
    ("function", "#8250df"), ("class", "#953800"), ("variable", "#953800"),
    ("operator", "#cf222e"), ("decorator", "#8250df"), ("tag", "#116329"),
]

# Sidebar header controls (search field + theme segmented control).
NAV_CONTROLS = [
    ("nav caption on header", "--kh-nav-caption", "--kh-sidebar-header-bg"),
    ("segment text on segment", "--kh-seg-text", "--kh-seg-bg"),
    ("segment active text", "--kh-seg-active-text", "--kh-seg-active-bg"),
    ("segment hover text", "#ffffff", "--kh-seg-hover-bg"),
    ("search input text", "--kh-text", "--kh-input-bg"),
    ("search placeholder", "--kh-faint", "--kh-input-bg"),
    ("header title link", "#f2f5f9", "--kh-sidebar-header-bg"),
    ("header version text", "#b6bfca", "--kh-sidebar-header-bg"),
]

ADMONITION_TITLE = [
    ("note title", "--kh-note-title-fg", "--kh-note-title-bg"),
    ("tip title", "--kh-tip-title-fg", "--kh-tip-title-bg"),
    ("warning title", "--kh-warning-title-fg", "--kh-warning-title-bg"),
    ("danger title", "--kh-danger-title-fg", "--kh-danger-title-bg"),
    ("seealso title", "--kh-seealso-title-fg", "--kh-seealso-title-bg"),
]

ADMONITION_BODY = [
    ("note body text", "--kh-text", "--kh-note-bg"),
    ("tip body text", "--kh-text", "--kh-tip-bg"),
    ("warning body text", "--kh-text", "--kh-warning-bg"),
    ("danger body text", "--kh-text", "--kh-danger-bg"),
    ("seealso body text", "--kh-text", "--kh-seealso-bg"),
    ("note body link", "--kh-link", "--kh-note-bg"),
    ("tip body link", "--kh-link", "--kh-tip-bg"),
    ("warning body muted", "--kh-muted", "--kh-warning-bg"),
    ("note body heading", "--kh-heading", "--kh-note-bg"),
]


def check(title: str, rows, tk: dict[str, str], quiet: bool = False) -> int:
    fails = 0
    if not quiet:
        print(f"\n=== {title} ===")
        print(f"{'pair':<26}{'fg':<10}{'bg':<10}{'ratio':>9}   AA")
    for label, fg, bg in rows:
        fgv, bgv = resolve(fg, tk), resolve(bg, tk)
        r = contrast(fgv, bgv)
        ok = r >= MIN_RATIO
        if not ok:
            fails += 1
            print(f"{title}: {label:<26}{fgv} on {bgv} = {r:.2f}:1  FAIL")
        elif not quiet:
            print(f"{label:<26}{fgv:<10}{bgv:<10}{r:>7.2f}:1   PASS")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="only print failures and the summary")
    args = ap.parse_args()

    css = CSS.read_text(encoding="utf-8")
    dark = tokens(css, 'html[data-theme="dark"] {')
    light = tokens(css, 'html[data-theme="light"] {')

    fails = 0
    fails += check("DARK — page palette", PAGE, dark, args.quiet)
    fails += check("DARK — chrome", CHROME_DARK, dark, args.quiet)
    fails += check("DARK — Pygments tokens", [(n, c, "--kh-code-bg") for n, c in PYGMENTS_DARK], dark, args.quiet)
    fails += check("DARK — nav controls", NAV_CONTROLS, dark, args.quiet)
    fails += check("DARK — admonition titles", ADMONITION_TITLE, dark, args.quiet)
    fails += check("DARK — admonition bodies", ADMONITION_BODY, dark, args.quiet)
    fails += check("LIGHT — page palette", PAGE, light, args.quiet)
    fails += check("LIGHT — chrome", CHROME_LIGHT, light, args.quiet)
    fails += check("LIGHT — Pygments tokens", [(n, c, "--kh-code-bg") for n, c in PYGMENTS_LIGHT], light, args.quiet)
    fails += check("LIGHT — nav controls", NAV_CONTROLS, light, args.quiet)
    fails += check("LIGHT — admonition titles", ADMONITION_TITLE, light, args.quiet)
    fails += check("LIGHT — admonition bodies", ADMONITION_BODY, light, args.quiet)

    total = 2 * (len(PAGE) + len(PYGMENTS_DARK) + len(ADMONITION_TITLE)
                 + len(ADMONITION_BODY) + len(NAV_CONTROLS)) + len(CHROME_DARK) + len(CHROME_LIGHT)
    print()
    if fails:
        print(f"FAIL: {fails} of {total} pair(s) below {MIN_RATIO}:1")
    else:
        print(f"PASS: all {total} text/background pairs reach >= {MIN_RATIO}:1 (WCAG AA)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
