/* Khmer translations for the strings searchtools.js resolves at runtime.
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
  var CATALOG = {
  "locale": "km",
  "messages": {
    ", in ": " ក្នុង ",
    "Preparing search...": "កំពុងរ្កៀបចំការស្វែងរក…",
    "Search Results": "លទ្ធផលស្វែងរក",
    "Search finished, found ${resultCount} page(s) matching the search query.": "ការស្វែងរកបានបញ្ចប់ រកឃើញ ${resultCount} ទំព័រដែលត្រូវនឹងពាក្យស្វែងរក។",
    "Searching": "កំពុងស្វែងរក",
    "Your search did not match any documents. Please make sure that all words are spelled correctly and that you've selected enough categories.": "ការស្វែងរករបស់អ្នកមិនត្រូវនឹងឯកសារណាទេ។ សូមប្រាកដថាពាក្យទាំងអស់សរសេរត្រូវ ហើយអ្នកបានជ្រើសរើសប្រភេទគ្រាន់គ្រប់គ្រាន់។"
  },
  "plural_expr": "n === 1 ? 0 : 1"
};

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
