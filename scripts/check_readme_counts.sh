#!/usr/bin/env bash
#
# check_readme_counts.sh - verify the auto-counted figures in README.md match
# reality, and refresh them in place when invoked with --update.
#
# The README claims figures that drift silently as the codebase grows:
#   1. The number of tests reported by `cargo test` (Build & test section).
#   2. The validation matrix figures: total cases, estimator families, and
#      passing cases (from validation/cases/*/case.yml).
#   3. The number of example scripts under examples/.
#
# These numbers are anchored to stable sentences in README.md. This guard keeps
# them honest, in the same spirit as scripts/check_pt_errors.sh. Run locally
# after large batches of new estimators/tests and from CI on every PR. If a
# sentence is reworded, the anchor pattern no longer matches and the check fails
# loudly so the maintainer deliberately re-anchors (see --update output).
#
# Usage:
#   scripts/check_readme_counts.sh           # verify; exit 1 on any mismatch
#   scripts/check_readme_counts.sh --update  # rewrite README.md in place
#
# If `cargo` is unavailable (e.g. a doc-only environment), the test count is
# skipped with a warning; all other counts are still verified.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README="$ROOT/README.md"
VALIDATION="$ROOT/validation/cases"

MODE="check"
if [[ "${1:-}" == "--update" ]]; then
  MODE="update"
fi

# --- Compute reality ---------------------------------------------------------

TEST_COUNT=""
if command -v cargo >/dev/null 2>&1; then
  TEST_COUNT="$(cd "$ROOT" && cargo test -- --list 2>/dev/null | grep -c ': test$')"
else
  echo "warning: cargo not found; skipping the test-count check." >&2
fi

CASE_COUNT="$(ls "$VALIDATION"/*/case.yml 2>/dev/null | wc -l)"
PASS_COUNT="$(grep -rl '^status: pass' "$VALIDATION"/*/case.yml 2>/dev/null | wc -l)"
FAMILY_COUNT="$(grep -rh '^estimator_family:' "$VALIDATION"/*/case.yml 2>/dev/null | sort -u | wc -l)"
EXAMPLE_COUNT="$(find "$ROOT/examples" -type f -name '*.hay' 2>/dev/null | wc -l)"

# --- Run verification / update against README --------------------------------

README="${README}" MODE="${MODE}" \
TEST_COUNT="${TEST_COUNT}" \
CASE_COUNT="${CASE_COUNT}" \
PASS_COUNT="${PASS_COUNT}" \
FAMILY_COUNT="${FAMILY_COUNT}" \
EXAMPLE_COUNT="${EXAMPLE_COUNT}" \
python3 - <<'PY'
import os
import re
import sys

path = os.environ["README"]
mode = os.environ["MODE"]

def num(name):
    v = os.environ.get(name, "")
    return None if v == "" else int(v)

tests = num("TEST_COUNT")
cases = num("CASE_COUNT")
passed = num("PASS_COUNT")
families = num("FAMILY_COUNT")
examples = num("EXAMPLE_COUNT")

# (anchor regex, human label, expected int or None)
# Anchors are matched against whitespace-collapsed README text so the patterns
# are immune to line wrapping.
TEST_PAT = r"cargo test\s+#\s+(\d+) tests, <1s"
VAL_PAT = (r"The matrix now covers \*\*(\d+) cases across (\d+) estimator families\*\*, "
           r"with (\d+)\s+passing")
EX_PAT = r"(\d+) example scripts in"

with open(path, encoding="utf-8") as fh:
    original = fh.read()
collapsed = re.sub(r"\s+", " ", original)

errors = []

m = re.search(VAL_PAT, collapsed)
if m:
    got = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    want = (cases, families, passed)
    for g, w, lab in zip(got, want, ("validation cases", "estimator families", "passing cases")):
        if g != w:
            errors.append(f"{lab}: README says {g}, reality is {w}")
else:
    errors.append("validation matrix sentence not found (re-anchor README)")

for pat, label, expected in ((TEST_PAT, "test count", tests), (EX_PAT, "example scripts", examples)):
    m = re.search(pat, collapsed)
    if not m:
        errors.append(f"{label}: anchor not found (re-anchor README)")
        continue
    got = int(m.group(1))
    if expected is not None and got != expected:
        errors.append(f"{label}: README says {got}, reality is {expected}")

if mode == "update":
    text = original
    done = 0
    if tests is not None:
        text, n = re.subn(r"(cargo test\s+#\s+)\d+( tests, <1s)", r"\g<1>%d\g<2>" % tests, text)
        done += n
    text, n = re.subn(
        r"(The matrix now covers \*\*)\d+( cases across )\d+( estimator families\*\*, with )\d+(\s+passing)",
        lambda m: m.group(1) + str(cases) + m.group(2) + str(families) + m.group(3) + str(passed) + m.group(4),
        text,
    )
    done += n
    text, n = re.subn(r"(\d+)( example scripts in)", r"%d\g<2>" % examples, text)
    done += n
    expected = (0 if tests is None else 1) + 2
    if done < expected:
        print(f"error: --update matched {done} anchors, expected {expected}; aborting.", file=sys.stderr)
        sys.exit(1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("updated README counts: tests=%s cases=%s families=%s passing=%s examples=%s"
          % (tests, cases, families, passed, examples))
    sys.exit(0)

if errors:
    print("error: README counts are out of date.")
    print("   Run: scripts/check_readme_counts.sh --update")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)

print("ok: README counts match reality"
      + ("" if tests is None else f" ({tests} tests, {cases} cases, {passed} passing, {examples} examples)"))
PY