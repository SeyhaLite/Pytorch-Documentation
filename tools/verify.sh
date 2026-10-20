#!/usr/bin/env bash
# Run every check this repository defines, and report one line per check.
#
#   bash tools/verify.sh
#
# Exits non-zero if anything fails, so it can be used as a pre-push gate.
# The Sphinx build is included and must produce zero warnings (-W).
set -u
cd "$(dirname "$0")/.."
pass=0; fail=0
run() {
  name="$1"; shift
  if out=$("$@" 2>&1); then
    printf "  PASS  %-34s %s\n" "$name" "$(echo "$out" | tail -1 | cut -c1-72)"
    pass=$((pass+1))
  else
    printf "  FAIL  %-34s\n" "$name"
    echo "$out" | tail -8 | sed 's/^/          /'
    fail=$((fail+1))
  fi
}
echo "================ FULL VERIFICATION ================"
run "code examples (40)"         python3 tests/run_examples.py --quiet
run "example references"        python3 tests/check_examples_referenced.py
run "markdown fences (25 files)" python3 tools/check_markdown.py
run "contrast (120 pairs)"      python3 tools/check_contrast.py -q
run "locale catalog"            python3 tests/check_locale.py
run "no English UI strings"     python3 tests/check_ui_strings.py
run "JS syntax: copybutton"     node --check docs/source/_static/js/copybutton.js
run "JS syntax: km-search-i18n" node --check docs/source/_static/js/km-search-i18n.js
run "RTD config is valid YAML"  python3 -c "import yaml;yaml.safe_load(open('.readthedocs.yaml',encoding='utf-8'))"
echo
echo "--- sphinx build (-W --keep-going) ---"
rm -rf docs/_build
if python3 -m sphinx -b html docs/source docs/_build/html -q -W --keep-going > /tmp/b.log 2>&1; then
  w=$(grep -c WARNING /tmp/b.log || true)
  printf "  PASS  %-34s %s pages, %s warnings\n" "sphinx build" "$(ls docs/_build/html/*.html | wc -l)" "$w"
  pass=$((pass+1))
else
  printf "  FAIL  sphinx build\n"; tail -6 /tmp/b.log | sed 's/^/          /'; fail=$((fail+1))
fi
echo "==================================================="
echo "passed $pass · failed $fail"
[ "$fail" -eq 0 ]
