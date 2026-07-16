#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Ensure Hayashi release binary is built
if [ ! -f "../target/release/hay" ]; then
    echo "Building Hayashi release binary..."
    (cd .. && cargo build --release)
fi

ESTIMATORS=(ols logit arima garch panel)
SIZES="1000,10000"
REPS=5

for est in "${ESTIMATORS[@]}"; do
    echo "=== Benchmarking $est ==="
    python3 scripts/run.py --estimator "$est" --sizes "$SIZES" --reps "$REPS"
done
