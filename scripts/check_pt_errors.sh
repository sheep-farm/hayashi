#!/usr/bin/env bash
#
# check_pt_errors.sh - guard against NEW Portuguese terms in user-facing
# Rust source under src/. The codebase still contains legacy Portuguese strings, so this
# check is intentionally diff-scoped: it only fails on lines ADDED relative to
# the base branch. As later PRs translate more strings, the legacy set shrinks
# and this guard keeps the trend one-directional.
#
# Usage:
#   scripts/check_pt_errors.sh [base_ref]
#
# If base_ref is omitted, the script tries the GitHub PR base branch, origin/dev,
# then upstream/dev. In CI, passing the PR base branch explicitly is safest.
set -euo pipefail

if [[ $# -gt 0 ]]; then
  BASE_REF="$1"
else
  BASE_BRANCH="${GITHUB_BASE_REF:-dev}"
  BASE_REF=""
  for candidate in "origin/${BASE_BRANCH}" "upstream/${BASE_BRANCH}" "origin/dev" "upstream/dev"; do
    if git rev-parse --verify --quiet "$candidate" >/dev/null; then
      BASE_REF="$candidate"
      break
    fi
  done
  if [[ -z "$BASE_REF" ]]; then
    echo "warning: no base ref found; skipping diff-scoped PT check." >&2
    exit 0
  fi
fi

# Word-bounded Portuguese terms common in user-facing messages.
# Keep this list conservative to avoid false positives on English text.
FORBIDDEN='\b(requer|coluna|colunas|inválid[oa]|variáve[li]|variáveis|esperad[oa]|vazi[oa]|deve ser|não|nao|encontrad[oa]|ao menos|pelo menos|forneça|nome de variável|nome da coluna|primeiro argumento|segundo argumento|terceiro argumento|devem ser|podem ser)\b'

if ! git rev-parse --verify --quiet "$BASE_REF" >/dev/null; then
  echo "warning: base ref '$BASE_REF' not found; skipping diff-scoped PT check." >&2
  exit 0
fi

# Collect added lines (excluding the +++ file headers) in production Rust files only.
added="$(git diff "$BASE_REF"...HEAD -U0 -- 'src/**/*.rs' \
          | grep '^+' \
          | grep -v '^+++' \
          | grep -iE "$FORBIDDEN" || true)"

if [[ -n "$added" ]]; then
  echo "error: Portuguese terms found in newly added Rust code."
  echo "   User-facing errors must be written in English (see docs/error-style.md)."
  echo
  echo "$added"
  exit 1
fi

echo "ok: no new Portuguese terms in added Rust lines (base: $BASE_REF)."
