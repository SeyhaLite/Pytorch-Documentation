"""Registers the project's static assets with Sphinx.

Kept as a local extension rather than plain ``html_js_files`` in conf.py for one
reason: **the loading method**.

``html_js_files`` entries are rendered inside ``<head>`` as plain blocking
``<script src=...>`` tags, which execute *before* the page markup exists.
``app.add_js_file(..., loading_method="defer")`` emits the deferred attribute,
so the browser parses the whole document first and only then runs the script —
the copy-button decoration is guaranteed to see every ``div.highlight``,
without depending on where in the page the script tag happens to land.

(Note: ``add_js_file()`` accepts ``filename``, ``priority``, ``loading_method``
and arbitrary HTML attributes via ``**kwargs``.  It has no ``body`` parameter —
passing one silently adds a useless ``body="end"`` attribute to the tag.)

Assets registered here:
    * ``js/copybutton.js`` — plain-text "ចម្លង" button on every code block
    * ``css/print.css``    — paper-friendly stylesheet (light colours, no nav)

The dark/light *control* is deliberately not registered here: requirement is
that its ``<script>`` sits inline, immediately after the element itself, so it
is rendered by ``_templates/searchbox.html`` instead.
"""

from __future__ import annotations

from sphinx.application import Sphinx

__version__ = "1.1.0"


def _add_assets(app: Sphinx) -> None:
    # loading_method must be "async" or "defer"; anything else is dropped.
    app.add_js_file("js/copybutton.js", loading_method="defer")
    app.add_js_file("js/km-search-i18n.js", loading_method="defer")
    app.add_css_file("css/print.css", media="print")


def setup(app: Sphinx) -> dict[str, object]:
    """Hook into ``builder-inited``, when ``add_js_file``/``add_css_file`` exist."""
    app.connect("builder-inited", _add_assets)
    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
