#!/usr/bin/env python3
"""Generate the Khmer UI catalogs for Sphinx, the RTD theme and searchtools.js.

There are two translation channels in a Sphinx HTML build, and they are loaded
completely differently:

1.  **Python side** — templates call `_('...')`, resolved from a compiled
    `.mo`.  This covers the sidebar, breadcrumbs, prev/next buttons, footer and
    the noscript warning.

2.  **JavaScript side** — `searchtools.js` calls `Documentation.gettext(...)`,
    which reads `Documentation.TRANSLATIONS`.  Sphinx populates that object from
    a per-locale JS catalog it generates only for locales it ships with.  `km`
    is not one of them, so `TRANSLATIONS` stays `{}` and the search UI would
    render in English.  This script therefore also emits a small JS file that
    calls `Documentation.addTranslations(...)`.

Both outputs are generated from the single `ENTRIES` table below, so they cannot
drift apart. Generation is deterministic -- the catalog dates are pinned -- so
re-running this script on unchanged input writes byte-identical files, which is
what lets tests/check_locale.py verify the committed artefacts by content
instead of by mtime (git does not preserve mtimes).  `tests/check_locale.py` verifies that every msgid searchtools.js
actually looks up is present.

Usage
-----
    python tools/build_locale.py

Regenerate after editing ENTRIES, then rebuild the site.  Both outputs are
committed: Read the Docs does not need Babel to build.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCALE_DIR = REPO / "docs" / "source" / "locale" / "km" / "LC_MESSAGES"
JS_OUT = REPO / "docs" / "source" / "_static" / "js" / "km-search-i18n.js"

# msgids consumed by searchtools.js at runtime. These MUST match the literal
# strings in Sphinx's searchtools.js -- note that the result-count message uses
# a plain "${resultCount}" placeholder that the script substitutes afterwards
# with String.replace, so it is part of the msgid.
JS_MSGIDS = {
    "Search Results",
    "Searching",
    "Preparing search...",
    "Search finished, found ${resultCount} page(s) matching the search query.",
    "Your search did not match any documents. Please make sure that all words "
    "are spelled correctly and that you've selected enough categories.",
    ", in ",
}

# (msgid, Khmer msgstr). API names, product names and brand names stay in English.
ENTRIES: list[tuple[str, str]] = [
    # --- sphinx_rtd_theme chrome -----------------------------------------
    ("Page navigation", "ការរុករកទំព័រ"),
    ("Sequential page navigation", "ការរុករកទំព័រតាមលំដាប់"),
    ("Previous", "មុន"),
    ("Next", "បន្ទាប់"),
    ("Footer", "បាតទំព័រ"),
    ("Edit on GitHub", "កែនៅលើ GitHub"),
    ("Edit on GitLab", "កែនៅលើ GitLab"),
    ("Edit on Bitbucket", "កែនៅលើ Bitbucket"),
    ("View page source", "មើលកូដប្រភព"),
    ("Navigation menu", "ម៉ឺនុយរុករក"),
    ("Mobile navigation menu", "ម៉ឺនុយរុករកសម្រាប់ទូរស័ព្ទ"),
    ("Logo", "រូបសញ្ញា"),
    ("Versions", "កំណែ"),
    ("Downloads", "ទាញយក"),
    ("On Read the Docs", "នៅលើ Read the Docs"),
    ("Project Home", "ទំព័រដើមគម្រោង"),
    ("Builds", "ការសាងសង់"),
    ("Build", "ការសាងសង់"),
    ("Revision", "កំណែបង្កើត"),
    ("Search docs", "ស្វែងរកឯកសារ"),
    ("Search within %(docstitle)s", "ស្វែងរកក្នុង %(docstitle)s"),
    ("About these documents", "អំពីឯកសារទាំងនេះ"),
    ("Index", "លិបិក្រម"),
    ("Search", "ស្វែងរក"),
    ("Copyright", "រក្សាសិទ្ធិ"),
    ("Last updated on %(last_updated)s.", "ធ្វើបច្ចុប្បន្នភាពចុងក្រោយ %(last_updated)s។"),
    ("&#169; Copyright %(copyright)s.", "&#169; រក្សាសិទ្ធិ %(copyright)s។"),
    ('&#169; <a href="%(path)s">Copyright</a> %(copyright)s.',
     '&#169; <a href="%(path)s">រក្សាសិទ្ធិ</a> %(copyright)s។'),
    ("Built with %(sphinx_web)s using a", "សាងសង់ដោយ %(sphinx_web)s ប្រើ"),
    ("theme", "theme"),
    ("provided by %(readthedocs_web)s", "ផ្ដល់ដោយ %(readthedocs_web)s"),

    # --- search UI ---------------------------------------------------------
    # The noscript fallback is rendered by the search.html template (Python
    # side); the rest are looked up by searchtools.js at runtime.
    ("Please activate JavaScript to enable the search functionality.",
     "សូមបើក JavaScript ដើម្បីអាចប្រើមុខងារស្វែងរក។"),
    ("Search Results", "លទ្ធផលស្វែងរក"),
    ("Searching", "កំពុងស្វែងរក"),
    ("Preparing search...", "កំពុងរ្កៀបចំការស្វែងរក…"),
    ("Search finished, found ${resultCount} page(s) matching the search query.",
     "ការស្វែងរកបានបញ្ចប់ រកឃើញ ${resultCount} ទំព័រដែលត្រូវនឹងពាក្យស្វែងរក។"),
    ("Your search did not match any documents. Please make sure that all words "
     "are spelled correctly and that you've selected enough categories.",
     "ការស្វែងរករបស់អ្នកមិនត្រូវនឹងឯកសារណាទេ។ សូមប្រាកដថាពាក្យទាំងអស់សរសេរត្រូវ "
     "ហើយអ្នកបានជ្រើសរើសប្រភេទគ្រាន់គ្រប់គ្រាន់។"),
    (", in ", " ក្នុង "),

    # --- Sphinx builtin ---------------------------------------------------
    ("General Index", "លិបិក្រមទូទៅ"),
    ("Search Page", "ទំព័រស្វែងរក"),
    ("Contents", "មាតិកា"),
    ("Full index on one page", "លិបិក្រមពេញលេញក្នុងទំព័រតែមួយ"),
    ("This Page", "ទំព័រនេះ"),
    ("Next topic", "ប្រធានបទបន្ទាប់"),
    ("Previous topic", "ប្រធានបទមុន"),
    ("Expand sidebar", "ពង្រីករបារចំហៀង"),
    ("Collapse sidebar", "បង្រួមរបារចំហៀង"),
    ("Quick search", "ស្វែងរករហ័ស"),
    ("Language", "ភាសា"),
    ("Permalink to this headline", "តំណអចិន្ត្រៃយ៍ទៅចំណងជើងនេះ"),
    ("Permalink to this definition", "តំណអចិន្ត្រៃយ៍ទៅនិយមន័យនេះ"),
    ("Hide Search Matches", "លាក់លទ្ធផលស្វែងរក"),
    ("Search mode:", "របៀបស្វែងរក៖"),
    ("Table of Contents", "មាតិកា"),
]


def render_js() -> str:
    """Return the exact bytes that km-search-i18n.js must contain.

    Pure function -- no filesystem access, no Babel.  tests/check_locale.py
    calls this to verify the committed artefact matches the ENTRIES table,
    which works on any machine (unlike comparing mtimes, which is meaningless
    after a git clone because git does not preserve them).
    """
    table = dict(ENTRIES)
    missing = sorted(JS_MSGIDS - table.keys())
    if missing:
        raise SystemExit(
            "JS_MSGIDS not covered by ENTRIES:\n  " + "\n  ".join(repr(m) for m in missing)
        )
    messages = {msgid: table[msgid] for msgid in sorted(JS_MSGIDS)}

    body = json.dumps({"locale": "km", "messages": messages,
                       "plural_expr": "n === 1 ? 0 : 1"},
                      ensure_ascii=False, indent=2)
    return JS_TEMPLATE.replace("__CATALOG_JSON__", body)


JS_TEMPLATE = """/* Khmer translations for the strings searchtools.js resolves at runtime.
 *
 * GENERATED FILE -- do not edit by hand.
 * Regenerate with:  python tools/build_locale.py
 *
 * Sphinx only emits a JS catalog for the locales it ships with. "km" is not one
 * of them, so Documentation.TRANSLATIONS would stay empty and the search UI
 * would fall back to English. This file fills it in.
 *
 * Source of truth: the ENTRIES table in tools/build_locale.py, which also
 * produces docs/source/locale/km/LC_MESSAGES/sphinx.mo.
 */
(function () {
  "use strict";
  var CATALOG = __CATALOG_JSON__;

  function apply() {
    if (typeof Documentation !== "undefined" && Documentation.addTranslations) {
      Documentation.addTranslations(CATALOG);
    }
  }

  /* doctools.js defines Documentation; searchtools.js only looks strings up
     when it renders results, which is strictly after DOMContentLoaded. */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply);
  } else {
    apply();
  }
})();
"""


# Pinned so that regenerating produces byte-identical files. Babel stamps
# POT-Creation-Date with "now" by default, which would make every run of this
# script show up as a diff in two committed artefacts -- and would make a
# content-based staleness check impossible.
FIXED_DATE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def build_catalog():
    """Return the Babel catalog. Pure: touches no files."""
    from babel.messages.catalog import Catalog

    catalog = Catalog(
        locale="km",
        project="ឯកសារ PyTorch ជាភាសាខ្មែរ",
        copyright_holder="PyTorch-Documentation (Khmer) contributors",
        creation_date=FIXED_DATE,
        revision_date=FIXED_DATE,
    )
    for msgid, msgstr in ENTRIES:
        catalog.add(msgid, string=msgstr, auto_comments=["UI string"])
    return catalog


def render_po() -> bytes:
    """Exact bytes of sphinx.po. Pure, so tests can compare against the file."""
    import io

    from babel.messages.pofile import write_po

    buf = io.BytesIO()
    write_po(buf, build_catalog(), width=78, sort_output=False)
    return buf.getvalue()


def render_mo() -> bytes:
    """Exact bytes of sphinx.mo. Pure, so tests can compare against the file."""
    import io

    from babel.messages.mofile import write_mo

    buf = io.BytesIO()
    write_mo(buf, build_catalog())
    return buf.getvalue()


def build_mo() -> Path:
    """Write sphinx.po and sphinx.mo, and return the path of the .mo."""
    LOCALE_DIR.mkdir(parents=True, exist_ok=True)
    (LOCALE_DIR / "sphinx.po").write_bytes(render_po())
    mo = LOCALE_DIR / "sphinx.mo"
    mo.write_bytes(render_mo())
    return mo


def build_js() -> Path:
    """Write the JS-side catalog for searchtools.js."""
    JS_OUT.parent.mkdir(parents=True, exist_ok=True)
    JS_OUT.write_text(render_js(), encoding="utf-8")
    return JS_OUT


def main() -> int:
    mo = build_mo()
    js = build_js()
    print(f"wrote {mo.relative_to(REPO)}  ({mo.stat().st_size:,} bytes, "
          f"{len(ENTRIES)} entries)")
    print(f"wrote {js.relative_to(REPO)}  ({js.stat().st_size:,} bytes, "
          f"{len(JS_MSGIDS)} JS strings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
