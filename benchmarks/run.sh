#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
Usage: ./run.sh [options]

Run all Hayashi/Greeners benchmarks (estimators, DataFrame operations, and
Greeners Rust microbenchmarks). With no options, runs everything using the
quick Criterion mode.

Options:
  --estimators     Run only the cross-language estimator benchmarks
  --ops            Run only the DataFrame/language operation benchmarks
  --rust           Run only the Greeners Criterion microbenchmarks
  --quick          Use Criterion's --quick mode (default)
  --full           Run Criterion with full statistics (slower)
  --help           Show this help

Environment variables:
  ESTIMATORS       Space-separated estimator list (default: ols logit probit iv qreg arima garch var panel)
  SIZES            Comma-separated dataset sizes (default: 1000,10000)
  ITERS            Timed iterations per subprocess run (default: 30)
  RUNS             Number of subprocess runs (default: 5)
  WARMUP           Untimed warmup iterations (default: 3)

Examples:
  ./run.sh                              # all benchmarks, Criterion quick
  ./run.sh --estimators                 # only estimator benchmarks
  ./run.sh --rust --full                # full Criterion run
  ./run.sh --estimators --ops --quick   # estimators + ops, rust quick
EOF
}

RUN_ESTIMATORS=0
RUN_OPS=0
RUN_RUST=0
RUST_ARGS="--quick"

ESTIMATORS=(ols logit probit iv qreg arima garch var panel)
: "${SIZES:=1000,10000}"
: "${ITERS:=30}"
: "${RUNS:=5}"
: "${WARMUP:=3}"

while [ $# -gt 0 ]; do
    case "$1" in
        --estimators) RUN_ESTIMATORS=1 ;;
        --ops) RUN_OPS=1 ;;
        --rust) RUN_RUST=1 ;;
        --quick) RUST_ARGS="--quick" ;;
        --full) RUST_ARGS="" ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

# Default: run all phases.
if [ "$RUN_ESTIMATORS" -eq 0 ] && [ "$RUN_OPS" -eq 0 ] && [ "$RUN_RUST" -eq 0 ]; then
    RUN_ESTIMATORS=1
    RUN_OPS=1
    RUN_RUST=1
fi

# Ensure Hayashi release binary is built.
if [ ! -f "../target/release/hay" ]; then
    echo "Building Hayashi release binary..."
    (cd .. && cargo build --release)
fi

if [ "$RUN_ESTIMATORS" -eq 1 ]; then
    echo "=== Hayashi estimator benchmarks ==="
    for est in "${ESTIMATORS[@]}"; do
        echo "--- $est ---"
        python3 scripts/run.py --estimator "$est" --sizes "$SIZES" \
            --iters "$ITERS" --runs "$RUNS" --warmup "$WARMUP"
    done
fi

if [ "$RUN_OPS" -eq 1 ]; then
    echo "=== Hayashi DataFrame/language operation benchmarks ==="
    python3 scripts/benchmark_ops.py \
        --iters "$ITERS" --runs "$RUNS" --warmup "$WARMUP"
fi

if [ "$RUN_RUST" -eq 1 ]; then
    echo "=== Greeners Rust microbenchmarks ==="
    # From hayashi/benchmarks to Greeners root.
    (cd ../../Greeners && cargo bench --bench micro ${RUST_ARGS})
fi

echo "=== All selected benchmarks finished ==="
echo "Hayashi JSON results are in results/"
echo "Greeners Criterion reports are in target/criterion/"
