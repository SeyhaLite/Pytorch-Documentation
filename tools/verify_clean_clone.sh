#!/usr/bin/env bash
# Reproduce the Read the Docs build from a clean clone.
#
#   bash tools/verify_clean_clone.sh
#
# Why this exists
# ---------------
# Two consecutive Read the Docs builds failed on checks that passed locally:
#
#   #35266806  check_locale.py hard-coded docs/_build/html, but RTD writes to
#              $READTHEDOCS_OUTPUT/html
#   #35269960  check_locale.py compared file mtimes, which git does not preserve
#              (a clone gives every file the checkout timestamp, in index order)
#
# Both are the same class of bug: a check that depends on the author's machine
# rather than on the repository. `tools/verify.sh` cannot catch that, because it
# runs against a working tree that already has a build directory and whatever
# mtimes happen to be there.
#
# This script clones the repository fresh -- exactly what RTD does -- and runs
# RTD's own sequence against it: pre_build jobs, the Sphinx command with RTD's
# flags, then the post_build jobs with READTHEDOCS_OUTPUT set. Anything that
# only works because of local state fails here first.
#
# It clones over HTTPS from GitHub by default so it also proves the pushed
# commit is complete; pass a path to clone locally instead.
set -u

REPO_URL="${1:-https://github.com/SeyhaLite/Pytorch-Documentation.git}"
WORK="${TMPDIR:-/tmp}/rtd-clean-clone-$$"
OUT="$WORK-output"

cleanup() { rm -rf "$WORK" "$OUT"; }
trap cleanup EXIT

fail=0
step() { printf "\n=== %s ===\n" "$1"; }

step "clone (what Read the Docs does first)"
rm -rf "$WORK"
if ! git clone --quiet --depth 1 "$REPO_URL" "$WORK"; then
  echo "  FAIL: could not clone $REPO_URL"
  exit 2
fi
cd "$WORK" || exit 2
echo "  cloned: $(git log --oneline -1)"

step "mtimes after clone (must not be relied upon)"
# `ls` prints a header per file when given several arguments, so filter on the
# timestamp column instead of skipping the first line.
for f in tools/build_locale.py \
         docs/source/_static/js/km-search-i18n.js \
         docs/source/locale/km/LC_MESSAGES/sphinx.mo; do
  [ -f "$f" ] && printf "  %s  %s\n" "$(date -r "$f" +%H:%M:%S 2>/dev/null || stat -c %y "$f" | cut -c12-19)" "$f"
done
echo "  ^ identical timestamps prove mtime comparisons are meaningless here"

step "pre_build jobs"
for t in tests/check_examples_referenced.py tools/check_markdown.py; do
  if python3 "$t" > /tmp/rtd-cc-$$.log 2>&1; then
    echo "  PASS $t"
  else
    echo "  FAIL $t"; tail -6 /tmp/rtd-cc-$$.log | sed 's/^/       /'; fail=1
  fi
done

step "sphinx build with RTD's exact flags"
mkdir -p "$OUT/html"
export READTHEDOCS_OUTPUT="$OUT"
( cd docs/source && python3 -m sphinx -T -W --keep-going -j auto -b html \
    -d _build/doctrees -D language=km . "$OUT/html" ) > /tmp/rtd-cc-build-$$.log 2>&1
rc=$?
pages=$(find "$OUT/html" -maxdepth 1 -name '*.html' 2>/dev/null | wc -l | tr -d ' ')
warns=$(grep -c WARNING /tmp/rtd-cc-build-$$.log 2>/dev/null || true)
echo "  exit=$rc · warnings=$warns · pages=$pages"
grep -E "^build succeeded|copying extra files" /tmp/rtd-cc-build-$$.log | sed 's/^/    /'
[ "$rc" -eq 0 ] || { tail -12 /tmp/rtd-cc-build-$$.log | sed 's/^/       /'; fail=1; }

step "post_build jobs"
for t in check_locale check_ui_strings check_extra_files; do
  if python3 "tests/$t.py" > /tmp/rtd-cc-$t-$$.log 2>&1; then
    echo "  PASS $t"
  else
    echo "  FAIL $t (exit $?)"; tail -8 /tmp/rtd-cc-$t-$$.log | sed 's/^/       /'; fail=1
  fi
done

rm -f /tmp/rtd-cc-*$$*.log

printf "\n===========================================\n"
if [ "$fail" -eq 0 ]; then
  echo "clean-clone RTD simulation: PASS"
else
  echo "clean-clone RTD simulation: FAIL"
fi
exit "$fail"
