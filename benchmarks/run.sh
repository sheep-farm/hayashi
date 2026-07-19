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
ITERS=30
RUNS=5
WARMUP=3

for est in "${ESTIMATORS[@]}"; do
    echo "=== Benchmarking $est ==="
    python3 scripts/run.py --estimator "$est" --sizes "$SIZES" \
        --iters "$ITERS" --runs "$RUNS" --warmup "$WARMUP"
done
