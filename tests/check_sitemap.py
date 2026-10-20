#!/usr/bin/env python3
"""Validate the generated sitemap.xml.

A malformed or stale sitemap is worse than none: Google reports it as an error
in Search Console, and URLs that 404 teach the crawler not to trust the file.
This checks the things that can silently go wrong.

  1. the file exists at the build output root and is well-formed XML
  2. the root element and namespace are the sitemaps.org 0.9 schema
  3. every <loc> is an absolute https URL under html_baseurl
  4. every <loc> has a matching page in the build output (no 404s submitted)
  5. every indexable built page is listed (nothing discoverable is omitted)
  6. no duplicates
  7. the pages that must NOT be listed are absent

Usage
-----
    sphinx-build -b html docs/source docs/_build/html -q
    python tests/check_sitemap.py                 # exit 1 on any problem
    python tests/check_sitemap.py --live          # also HEAD-check each URL
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from _buildpath import require_build_dir

REPO = Path(__file__).resolve().parent.parent
CONF = REPO / "docs" / "source" / "conf.py"

NS = "http://www.sitemaps.org/schemas/sitemap/0.9"

# Pages that must never appear in the sitemap.
MUST_NOT_LIST = {
    # Google's ownership token. It has no content; indexing it is pointless and
    # would put a verification artefact in search results.
    "google59471f16170f14f0.html": "Search Console ownership token",
    # Renders nothing until searchtools.js runs, so it is a thin duplicate.
    "search.html": "dynamic search page, no indexable content",
    # Auto-generated alphabetical index; low value, duplicates every heading.
    "genindex.html": "auto-generated index",
}


def expected_baseurl() -> str:
    """Read html_baseurl's fallback from conf.py.

    On Read the Docs html_baseurl comes from READTHEDOCS_CANONICAL_URL, but the
    extension falls back to DEFAULT_BASEURL locally, and that fallback is the
    production URL. Reading it from the extension keeps this test honest if the
    URL ever changes.
    """
    ext = REPO / "docs" / "source" / "_ext" / "khmer_sitemap.py"
    src = ext.read_text(encoding="utf-8")
    for line in src.split("\n"):
        if line.startswith("DEFAULT_BASEURL"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise AssertionError("DEFAULT_BASEURL not found in khmer_sitemap.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true",
                    help="HEAD-request every <loc> to confirm it resolves")
    args = ap.parse_args()

    build = require_build_dir("check_sitemap")
    problems: list[str] = []

    path = build / "sitemap.xml"
    if not path.is_file():
        print(f"FAIL: {path} does not exist.\n"
              f"      Is khmer_sitemap in the `extensions` list of conf.py?",
              file=sys.stderr)
        return 1

    # 1-2. well-formed XML with the right schema
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        print(f"FAIL: sitemap.xml is not well-formed XML: {exc}", file=sys.stderr)
        return 1

    if root.tag != f"{{{NS}}}urlset":
        problems.append(f"root element is {root.tag!r}, expected {{{NS}}}urlset")

    locs = [el.text.strip() for el in root.iter(f"{{{NS}}}loc") if el.text and el.text.strip()]
    if not locs:
        problems.append("sitemap contains no <loc> entries")

    base = expected_baseurl()
    built = {p.name for p in build.glob("*.html")}

    print(f"sitemap          : {path.relative_to(REPO)}")
    print(f"urls             : {len(locs)}")
    print(f"expected baseurl : {base}")

    # 3, 4, 6
    seen: set[str] = set()
    listed_names: set[str] = set()
    for loc in locs:
        if not loc.startswith("https://"):
            problems.append(f"not an absolute https URL: {loc}")
            continue
        if not loc.startswith(base):
            problems.append(f"URL is outside html_baseurl ({base}): {loc}")
        name = loc.rsplit("/", 1)[-1]
        listed_names.add(name)
        if loc in seen:
            problems.append(f"duplicate <loc>: {loc}")
        seen.add(loc)
        if name not in built:
            problems.append(f"submitted URL has no built page (would 404): {loc}")

    # 5. every indexable page is listed
    expected = built - set(MUST_NOT_LIST) - {"sitemap.xml"}
    omitted = sorted(expected - listed_names)
    if omitted:
        problems.append(f"indexable pages missing from the sitemap: {omitted}")

    # 7. exclusions hold
    for name, why in MUST_NOT_LIST.items():
        if name in listed_names:
            problems.append(f"{name} is listed but must not be ({why})")

    print(f"built pages      : {len(built)}")
    print(f"indexable        : {len(expected)}   (excluded: {', '.join(sorted(MUST_NOT_LIST))})")
    print(f"all listed resolve to a built page: {not any('would 404' in p for p in problems)}")

    if args.live and locs:
        print("\nlive HEAD checks:")
        bad = 0
        for loc in locs:
            try:
                with urlopen(Request(loc, method="HEAD",
                                     headers={"User-Agent": "sitemap-check/1.0"}),
                             timeout=20) as r:
                    status = r.status
            except HTTPError as e:
                status = e.code
            except Exception as e:
                status = f"ERR {type(e).__name__}"
            if status != 200:
                bad += 1
                problems.append(f"live URL returned {status}: {loc}")
        print(f"  {len(locs) - bad}/{len(locs)} returned HTTP 200")

    print()
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1
    print("OK: sitemap is well-formed, complete, and every URL resolves to a built page")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
