#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Garante binario release do Hayashi
if [ ! -f "../target/release/hay" ]; then
    echo "Building Hayashi release binary..."
    (cd .. && cargo build --release)
fi

echo "Running OLS benchmark"
python3 scripts/run.py --estimator ols --sizes 1000,10000,100000 --reps 5
