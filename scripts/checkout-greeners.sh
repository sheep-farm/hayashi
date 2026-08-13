#!/usr/bin/env bash

set -euo pipefail

readonly GREENERS_TAG="v1.6.1"
readonly GREENERS_REV="5568d757b234c813ac59872df5f975a246eb1c06"
readonly GREENERS_URL="https://github.com/sheep-farm/Greeners.git"
readonly GREENERS_DIR="${GREENERS_DIR:-${GITHUB_WORKSPACE:-$PWD}/../Greeners}"

git clone --depth 1 --branch "$GREENERS_TAG" --no-tags "$GREENERS_URL" "$GREENERS_DIR"

actual_rev="$(git -C "$GREENERS_DIR" rev-parse HEAD)"
if [[ "$actual_rev" != "$GREENERS_REV" ]]; then
    printf 'Greeners tag %s resolved to %s; expected %s\n' \
        "$GREENERS_TAG" "$actual_rev" "$GREENERS_REV" >&2
    exit 1
fi

printf 'Using Greeners %s at %s\n' "$GREENERS_TAG" "$actual_rev"
