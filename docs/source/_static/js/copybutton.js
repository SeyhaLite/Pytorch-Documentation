/* Plain-text "copy" control for every code block.
 *
 * Loaded through _ext/khmer_assets.py as
 *     app.add_js_file("js/copybutton.js", loading_method="defer")
 * The deferred attribute means the browser parses the whole document before
 * running this, so every div.highlight already exists -- without depending on
 * html_js_files ordering (those tags are blocking and live in <head>).
 *
 * Design constraints: plain text only (no icon), no border box of its own
 * beyond a chip that inherits the theme palette, and it must degrade to
 * document.execCommand on file:// and inside sandboxed iframes.
 */
(function () {
  "use strict";

  var LABEL = "ចម្លង";        /* copy       */
  var DONE = "បានចម្លង";       /* copied     */
  var ERR = "បរាជ័យ";           /* failed     */
  var SELECTOR = "div.highlight";

  function textOf(block) {
    var pre = block.querySelector("pre");
    return pre ? pre.innerText.replace(/\n$/, "") : block.innerText;
  }

  function fallbackCopy(text) {
    /* navigator.clipboard is unavailable on file:// and in some iframes. */
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }

  function attach(block) {
    if (block.querySelector(":scope > .kh-copy-btn")) { return; }
    if (!block.querySelector("pre")) { return; }

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "kh-copy-btn";
    btn.textContent = LABEL;
    btn.title = LABEL;
    btn.setAttribute("aria-label", LABEL);

    var reset;
    btn.addEventListener("click", function () {
      var text = textOf(block);
      var finish = function (ok) {
        btn.textContent = ok ? DONE : ERR;
        btn.classList.toggle("is-done", !!ok);
        btn.classList.toggle("is-error", !ok);
        window.clearTimeout(reset);
        reset = window.setTimeout(function () {
          btn.textContent = LABEL;
          btn.classList.remove("is-done", "is-error");
        }, 1800);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(function () { finish(true); },
                                                function () { finish(fallbackCopy(text)); });
      } else {
        finish(fallbackCopy(text));
      }
    });

    block.appendChild(btn);
  }

  function run() {
    var blocks = document.querySelectorAll(SELECTOR);
    for (var i = 0; i < blocks.length; i += 1) { attach(blocks[i]); }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
