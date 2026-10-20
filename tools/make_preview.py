#!/usr/bin/env python3
"""Build ONE self-contained offline preview of the documentation.

Why: the app file viewer renders HTML inside a sandboxed iframe with no network
access, so `<link rel=stylesheet>` and `<script src>` never load there.  This
script inlines every stylesheet and script into a single page so the palette,
the Khmer font stack, the dark/light control and the copy buttons can be
inspected visually without a server.

    python tools/make_preview.py                 # -> preview.html (repo root)
    python tools/make_preview.py -o /tmp/p.html  # custom output path

Prerequisite: build the site first.

    sphinx-build -b html docs/source docs/_build/html -q -W --keep-going

Notes
-----
* The page is built from `tensors.html`, which contains tables, inline chips,
  code blocks with Pygments highlighting, admonitions of several types and the
  full sidebar navigation — i.e. every element the styling brief covers.
* The Google Fonts `@import` is kept but will silently fail offline; the
  `font-family` stack then falls back to any locally installed Khmer font.
  A banner in the preview says so explicitly rather than hiding it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "docs" / "_build" / "html"

LINK_RE = re.compile(r'[ \t]*<link[^>]*rel="stylesheet"[^>]*>', re.I)
HREF_RE = re.compile(r'href="([^"]+)"', re.I)
SCRIPT_SRC_RE = re.compile(r'[ \t]*<script[^>]*\ssrc="([^"]+)"[^>]*>\s*</script>', re.I)
IMPORT_RE = re.compile(r'@import\s+url\([^)]*\)\s*;?')

BANNER = """
<div id="preview-banner" style="position:sticky;top:0;z-index:9999;margin:0;
     padding:10px 16px;font:14px/1.6 system-ui,'Kantumruy Pro','Noto Sans Khmer',
     'Khmer OS',sans-serif;background:#1f6feb;color:#fff;border-bottom:2px solid #0d419d;">
  <strong>Offline preview</strong> — CSS និង JS ទាំងអស់ត្រូវបាន inline។
  ប្រើប៊ូតុង <em>ពណ៌ងងឹត / ពណ៌ភ្លឺ</em> នៅ sidebar ដើម្បីប្ដូរ mode
  និងប៊ូតុង <em>ចម្លង</em> នៅជ្រុង code block នីមួយៗ។
  <span style="opacity:.85">ពុម្ពអក្សរ Khmer ផ្ទុកពី Google Fonts CDN —
  បើគ្មានអ៊ីនធឺណិត វានឹងធ្លាក់ទៅពុម្ពអក្សរខ្មែរក្នុងម៉ាស៊ីន។</span>
</div>
"""


def inline_css(html: str) -> tuple[str, int]:
    """Replace every stylesheet <link> with an inline <style> block."""
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        href = HREF_RE.search(match.group(0))
        if not href:
            return match.group(0)
        path = BUILD / href.group(1).split("?")[0]
        if not path.exists():
            return f"<!-- missing stylesheet: {href.group(1)} -->"
        css = path.read_text(encoding="utf-8")
        count += 1
        return (f'<style data-source="{href.group(1).split("?")[0]}">\n'
                f"{css}\n</style>")

    return LINK_RE.sub(repl, html), count


def inline_js(html: str) -> tuple[str, int]:
    """Replace every <script src> with an inline <script> block.

    The theme-toggle script is ALREADY inline in the built page (rendered by
    _templates/searchbox.html right after the control), so it is left alone —
    that adjacency is exactly what the styling brief requires.
    """
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        path = BUILD / match.group(1).split("?")[0]
        if not path.exists():
            return f"<!-- missing script: {match.group(1)} -->"
        js = path.read_text(encoding="utf-8").replace("</script>", "<\\/script>")
        count += 1
        return f'<script data-source="{match.group(1).split("?")[0]}">\n{js}\n</script>'

    return SCRIPT_SRC_RE.sub(repl, html), count


def force_dark_default(html: str) -> str:
    """Start the preview in dark mode even when the OS prefers light.

    The control's own script reads localStorage first, then prefers-color-scheme.
    Seeding a <script> in <head> before it runs gives a deterministic first
    paint without touching the control's markup or its inline script.
    """
    seed = (
        "<script>try{window.localStorage.setItem('kh-docs-theme','dark');}"
        "catch(e){}</script>\n"
    )
    return html.replace("<head>", "<head>\n" + seed, 1)


def strip_cdn_import(css_inlined_html: str) -> str:
    """Neutralise the Google Fonts @import so an offline viewer does not stall.

    The @import lives inside the inlined custom.css.  Replacing it with a
    comment keeps the font-family fallback stack intact while removing the
    network dependency; the banner explains the trade-off.
    """
    return IMPORT_RE.sub("/* @import removed for the offline preview */", css_inlined_html)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", default="tensors.html",
                    help="built page to use as the preview (default: tensors.html)")
    ap.add_argument("-o", "--output", default=str(REPO / "preview.html"),
                    help="output path (default: <repo>/preview.html)")
    ap.add_argument("--keep-cdn-fonts", action="store_true",
                    help="keep the Google Fonts @import (needs network)")
    args = ap.parse_args()

    src = BUILD / args.page
    if not src.exists():
        print(f"missing built page: {src}\n"
              f"run: sphinx-build -b html docs/source docs/_build/html -q -W --keep-going",
              file=sys.stderr)
        return 2

    html = src.read_text(encoding="utf-8")
    html, n_css = inline_css(html)
    html, n_js = inline_js(html)
    if not args.keep_cdn_fonts:
        html = strip_cdn_import(html)
    html = force_dark_default(html)
    html = html.replace("<body class=\"wy-body-for-nav\">",
                        "<body class=\"wy-body-for-nav\">\n" + BANNER, 1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    print(f"source page     : {args.page}")
    print(f"stylesheets     : {n_css} inlined")
    print(f"scripts         : {n_js} inlined (+1 already-inline theme control)")
    print(f"CDN @import     : {'kept' if args.keep_cdn_fonts else 'removed'}")
    print(f"output          : {out} ({size_kb:.0f} KB)")

    # Sanity: the page must still carry the elements the brief asks to inspect.
    probes = {
        "table": "<table" in html,
        "kbd chip": "<kbd" in html or "class=\"kbd\"" in html,
        "inline code": "<code" in html,
        "code block": "class=\"highlight\"" in html,
        "admonition": "admonition" in html,
        "sidebar nav": "wy-menu-vertical" in html,
        "theme control": "kh-theme-toggle" in html,
        "copy button js": "kh-copy-btn" in html,
        "dark palette": "--kh-page-bg: #14161a" in html,
        "light palette": "--kh-page-bg: #ffffff" in html,
        "khmer font stack": "Kantumruy Pro" in html,
        "khmer prose": "ការបង្កើត" in html or "ខ្មែរ" in html,
    }
    print("\nelements present in the preview:")
    bad = 0
    for name, ok in probes.items():
        print(f"  {'OK  ' if ok else 'MISS'} {name}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
