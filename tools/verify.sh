#!/usr/bin/env bash
# Run every check this repository defines, in the order they must run.
#
#   bash tools/verify.sh
#
# Exits non-zero if anything fails, so it works as a pre-push gate.
#
# ORDER MATTERS. Three of the checks read the *built* HTML site, so the Sphinx
# build has to happen before them, not after. Running them first only appears to
# work when a previous build is still on disk -- which is exactly how a fresh
# checkout or a clean CI runner would fail for no obvious reason.
#
#   stage 1  source-only checks   (need nothing but the repo)
#   stage 2  sphinx build         (-W: any warning is a failure)
#   stage 3  post-build checks    (need docs/_build/html)
set -u
cd "$(dirname "$0")/.."

pass=0
fail=0

run() {
  name="$1"; shift
  if out=$("$@" 2>&1); then
    printf "  PASS  %-32s %s\n" "$name" "$(printf '%s' "$out" | tail -1 | cut -c1-70)"
    pass=$((pass + 1))
  else
    printf "  FAIL  %-32s\n" "$name"
    printf '%s\n' "$out" | tail -10 | sed 's/^/          /'
    fail=$((fail + 1))
  fi
}

build_site() {
  rm -rf docs/_build
  if python3 -m sphinx -b html docs/source docs/_build/html -q -W --keep-going > /tmp/kh-build.log 2>&1; then
    pages=$(find docs/_build/html -maxdepth 1 -name '*.html' | wc -l | tr -d ' ')
    warns=$(grep -c WARNING /tmp/kh-build.log || true)
    printf "  PASS  %-32s %s pages, %s warnings\n" "sphinx build (-W)" "$pages" "$warns"
    pass=$((pass + 1))
  else
    printf "  FAIL  %-32s\n" "sphinx build (-W)"
    tail -12 /tmp/kh-build.log | sed 's/^/          /'
    fail=$((fail + 1))
    return 1
  fi
}

echo "============ stage 1: source-only checks ============"
run "code examples (40)"          python3 tests/run_examples.py --quiet
run "example references"          python3 tests/check_examples_referenced.py
run "markdown fences (25 files)"  python3 tools/check_markdown.py
run "contrast (120 pairs)"        python3 tools/check_contrast.py -q
run "JS syntax: copybutton"       node --check docs/source/_static/js/copybutton.js
run "JS syntax: km-search-i18n"   node --check docs/source/_static/js/km-search-i18n.js
run "RTD config is valid YAML"    python3 -c "import yaml;yaml.safe_load(open('.readthedocs.yaml',encoding='utf-8'))"

echo
echo "============ stage 2: build ============"
if ! build_site; then
  echo
  echo "build failed; skipping the post-build checks"
  echo "==================================================="
  echo "passed $pass · failed $fail"
  exit 1
fi

echo
echo "============ stage 3: post-build checks ============"
run "locale catalog"              python3 tests/check_locale.py
run "no English UI strings"       python3 tests/check_ui_strings.py
run "search console verification" python3 tests/check_search_console.py
run "sitemap.xml"                 python3 tests/check_sitemap.py

echo "==================================================="
echo "passed $pass · failed $fail"
[ "$fail" -eq 0 ]
