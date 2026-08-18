#!/usr/bin/env bash

set -euo pipefail

readonly GREENERS_BRANCH="2.0-proposal/workspace"
readonly GREENERS_REV="3b56579443f3abed9f58882b90e4efb07b3f80d0"
readonly GREENERS_URL="https://github.com/sheep-farm/Greeners.git"
readonly GREENERS_DIR="${GREENERS_DIR:-${GITHUB_WORKSPACE:-$PWD}/../Greeners}"

git clone --depth 1 --branch "$GREENERS_BRANCH" --no-tags "$GREENERS_URL" "$GREENERS_DIR"

actual_rev="$(git -C "$GREENERS_DIR" rev-parse HEAD)"
if [[ "$actual_rev" != "$GREENERS_REV" ]]; then
    printf 'Greeners branch %s resolved to %s; expected %s\n' \
        "$GREENERS_BRANCH" "$actual_rev" "$GREENERS_REV" >&2
    exit 1
fi

printf 'Using Greeners %s at %s\n' "$GREENERS_BRANCH" "$actual_rev"
