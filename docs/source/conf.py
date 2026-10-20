"""Sphinx configuration for the Khmer PyTorch documentation.

Build with:
    sphinx-build -b html docs/source docs/_build/html -W --keep-going
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "ឯកសារ PyTorch ជាភាសាខ្មែរ"
author = "PyTorch-Documentation (Khmer) contributors"
copyright = (
    f"{datetime.now(timezone.utc).year}, {author}. "
    "ខ្លឹមសារអត្ថបទ: CC BY 4.0 · កូដឧទាហរណ៍: BSD 3-Clause · "
    "ដកស្រង់ពី PyTorch Documentation (CC BY 4.0, © PyTorch Contributors)"
)
version = "1.0"
release = "1.0.0"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_ext"))

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "khmer_assets",   # local: copy button, i18n JS catalog, print CSS
    "khmer_sitemap",  # local: sitemap.xml in the output root
]
# NOTE: sphinx.ext.intersphinx is deliberately NOT enabled. Every cross
# reference in these pages uses MyST's {doc} role, which resolves against the
# local toctree -- no page uses :py:, :func:, :class: or :ref:. Enabling
# intersphinx would make the build fetch https://docs.python.org/3/objects.inv
# on every run, adding a network dependency (and, under -W, a network failure
# mode) for zero benefit.

templates_path = [str(HERE / "_templates")]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
]

# ---------------------------------------------------------------------------
# Source formats
# ---------------------------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

myst_enable_extensions = [
    "colon_fence",   # ::: fences, required for figure/admonition/literalinclude
    "deflist",
    "fieldlist",
    "tasklist",
    "attrs_inline",
    "attrs_block",
    "substitution",
]
myst_heading_anchors = 4
# Keep default CommonMark behaviour: a single newline is NOT a hard line break.
suppress_warnings = []

# ---------------------------------------------------------------------------
# HTML output / theme
# ---------------------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = "ឯកសារ PyTorch ជាភាសាខ្មែរ"
html_short_title = "PyTorch ខ្មែរ"
# Read the Docs sets READTHEDOCS_CANONICAL_URL to the canonical URL of the
# version being built, e.g. https://pytorch-documentation.readthedocs.io/km/latest/
# It already includes the language and version segments. khmer_sitemap uses it as
# the base for every <loc>, and falls back to the production URL on a local build
# so the generated sitemap still contains resolvable links.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

html_static_path = [str(HERE / "_static")]

# Files here are copied verbatim to the OUTPUT ROOT (not under _static/).
#
# google59471f16170f14f0.html is the Google Search Console ownership token and
# must stay reachable at
#     https://pytorch-documentation.readthedocs.io/google59471f16170f14f0.html
# Google re-checks it periodically, so removing it can silently drop the site
# from Search Console even after verification once succeeded. Do not delete it.
html_extra_path = [str(HERE / "extra")]
html_css_files = [
    # Custom palette is loaded after the theme CSS so it can win the cascade.
    "css/custom.css",
]
# NOTE: no html_js_files on purpose.
#   * the copy-button script is registered by _ext/khmer_assets.py with
#     loading_method="defer", so it runs after the markup is parsed;
#   * the dark/light control ships its <script> inline, immediately after the
#     element itself (see _templates/searchbox.html), so its behaviour never
#     depends on <head> script ordering.
# html_js_files entries are rendered as blocking tags inside <head> and their
# position relative to the body markup cannot be controlled.

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
    "includehidden": True,
    "style_nav_header_background": "var(--kh-sidebar-header-bg)",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # NOTE: sphinx_rtd_theme 3.x removed `display_version` and `github_url`.
    # The GitHub flyout link is configured through html_context below instead.
}

# Google Search Console ownership token for
#   https://pytorch-documentation.readthedocs.io/
# Rendered into <head> on every page by _templates/layout.html.
#
# This value is PUBLIC BY DESIGN: Google reads it from the served HTML, so it
# must be committed. It is not a credential and grants no access -- leaking it
# would only let someone else claim verification of a domain they already
# control. Do not "rotate" it as if it were a secret.
#
# Why a meta tag rather than the HTML-file method: Read the Docs mounts the
# build under /<language>/<version>/, so nothing Sphinx emits can ever be served
# at the domain root. The file Google asks for at
#   https://pytorch-documentation.readthedocs.io/google59471f16170f14f0.html
# 404s no matter what html_extra_path contains; the same file IS served at
#   .../km/latest/google59471f16170f14f0.html
# The file is kept because it works for a URL-prefix property registered at
# /km/latest/, but the meta tag is what verifies the bare domain.
google_site_verification = "L2d5Vp4txgR8OhRYpJvFenCBS0BqDfbfMYt0SG7rGHw"

html_context = {
    # Consumed by _templates/layout.html.
    "google_site_verification": google_site_verification,
    "display_github": True,
    "github_user": "SeyhaLite",
    "github_repo": "Pytorch-Documentation",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
    "language": "km",
    # The sidebar theme control is rendered only while this flag is true, so
    # deleting the override template also removes the control cleanly.
    "khmer_theme_toggle": True,
}

html_show_sourcelink = True
html_show_sphinx = False
html_show_copyright = True
html_copy_source = True
html_last_updated_fmt = "%Y-%m-%d"
html_favicon = None
html_logo = None
htmlhelp_basename = "PyTorchKhmerDocs"

# ---------------------------------------------------------------------------
# Search / i18n
# ---------------------------------------------------------------------------

language = "km"
locale_dirs = ["locale/"]
gettext_compact = False

# Khmer is written without spaces between words, which confuses Sphinx's
# whitespace-based word splitter. Indexing as English keeps ASCII API names
# ("torch", "DataLoader", "backward") searchable; Khmer phrases are matched
# through the browser's in-page find as a fallback.
html_search_language = "en"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

pygments_style = "default"        # light mode
pygments_dark_style = "native"    # dark mode
highlight_language = "python"
add_function_parentheses = True
todo_include_todos = False
numfig = False
smartquotes = False               # do not convert straight quotes in Khmer text
