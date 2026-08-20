#!/usr/bin/env python3
"""Hayashi estimator benchmark orchestrator against R and Python.

Each implementation script is run as a subprocess and must emit one
`  elapsed: X.XXXXs` line per timed iteration. The runner parses these
lines, measures peak RSS, and aggregates statistics across runs.
"""

import argparse
import sys
from pathlib import Path

import common

BENCH_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = BENCH_DIR / ".tmp"

ESTIMATORS = {
    "ols": {
        "hay": "ols(y ~ x1 + x2 + x3, df)",
        "python": "ols_python.py",
        "r": "ols_r.R",
    },
    "logit": {
        "hay": "logit(y ~ x1 + x2, df)",
        "python": "logit_python.py",
        "r": "logit_r.R",
    },
    "probit": {
        "hay": "probit(y ~ x1 + x2, df)",
        "python": "probit_python.py",
        "r": "probit_r.R",
    },
    "iv": {
        "hay": "iv(y ~ x, ~ z, df)",
        "python": "iv_python.py",
        "r": "iv_r.R",
    },
    "qreg": {
        "hay": "qreg(y ~ x1 + x2, df, tau=0.5, boot=0)",
        "python": "qreg_python.py",
        "r": "qreg_r.R",
    },
    "arima": {
        "hay": "arima(df, y, p=1, d=0, q=0)",
        "python": "arima_python.py",
        "r": "arima_r.R",
    },
    "garch": {
        "hay": "garch(df, y, p=1, q=1)",
        "python": "garch_python.py",
        "r": "garch_r.R",
    },
    "var": {
        "hay": "var(df, y1, y2, lags=1)",
        "python": "var_python.py",
        "r": "var_r.R",
    },
    "panel": {
        "hay": "fe(y ~ x, df, id=firm)",
        "python": "panel_python.py",
        "r": "panel_r.R",
    },
}


def _write_hayashi_script(
    estimator: str, dataset: Path, iters: int, warmup: int
) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    script = TMP_DIR / f"{estimator}_hayashi_{dataset.stem}.hay"
    call = ESTIMATORS[estimator]["hay"]
    source = (
        f'load "{dataset}" as df\n'
        f"for i in 1..={warmup} {{ let _ = {call} }}\n"
        f"for i in 1..={iters} {{ let _ = timer({call}, digits=6) }}\n"
        f'print("done")\n'
    )
    script.write_text(source)
    return script


def run_hayashi(
    estimator: str, dataset: Path, iters: int, warmup: int, runs: int
) -> dict:
    script = _write_hayashi_script(estimator, dataset, iters, warmup)
    cmd = [str(common.HAY_EXE), str(script)]
    return _run_impl(cmd, iters, warmup, runs)


def run_python(
    estimator: str, dataset: Path, iters: int, warmup: int, runs: int
) -> dict:
    script = BENCH_DIR / "implementations" / ESTIMATORS[estimator]["python"]
    cmd = [sys.executable, str(script), str(dataset), str(iters), str(warmup)]
    return _run_impl(cmd, iters, warmup, runs)


def run_r(
    estimator: str, dataset: Path, iters: int, warmup: int, runs: int
) -> dict:
    script = BENCH_DIR / "implementations" / ESTIMATORS[estimator]["r"]
    cmd = ["Rscript", str(script), str(dataset), str(iters), str(warmup)]
    return _run_impl(cmd, iters, warmup, runs)


def _run_impl(cmd: list[str], iters: int, warmup: int, runs: int) -> dict:
    all_times = []
    memory_peaks = []
    wall_times = []
    for _ in range(runs):
        times, peak_kb, wall = common.measure_run(cmd)
        all_times.extend(times)
        memory_peaks.append(peak_kb)
        wall_times.append(wall)
    return common.aggregate(all_times, memory_peaks, wall_times, iters, warmup, runs)


def main():
    parser = argparse.ArgumentParser(description="Run Hayashi benchmarks")
    parser.add_argument(
        "--estimator",
        default="ols",
        choices=list(ESTIMATORS.keys()),
        help="estimator to benchmark",
    )
    parser.add_argument(
        "--sizes",
        default="1000,10000,100000",
        help="comma-separated dataset sizes",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=30,
        help="timed iterations per subprocess run",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="warmup iterations per subprocess run (untimed)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="number of timed subprocess runs",
    )
    parser.add_argument(
        "--lang",
        default="hayashi,python,r",
        help="comma-separated languages to benchmark",
    )
    args = parser.parse_args()

    if not common.HAY_EXE.exists():
        print(f"Hayashi binary not found at {common.HAY_EXE}; build with: cargo build --release")
        sys.exit(1)

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    langs = [l.strip().lower() for l in args.lang.split(",")]

    runners = {
        "hayashi": run_hayashi,
        "python": run_python,
        "r": run_r,
    }

    print(f"Benchmarking {args.estimator}: sizes={sizes}, iters={args.iters}, warmup={args.warmup}, runs={args.runs}")
    datasets = common.generate_datasets(args.estimator, sizes)

    results = []
    for dataset in datasets:
        print(f"\nDataset: {dataset.name}")
        n = int(dataset.stem.split("_n")[-1])
        for lang in langs:
            if lang not in runners:
                print(f"  skip unknown language: {lang}")
                continue
            print(f"  running {lang}...", end="", flush=True)
            try:
                stats = runners[lang](
                    args.estimator, dataset, args.iters, args.warmup, args.runs
                )
                stats["estimator"] = args.estimator
                stats["language"] = lang
                stats["n"] = n
                stats["dataset"] = str(dataset)
                results.append(stats)
                mem_mb = stats["memory_kb_mean"] / 1024
                print(
                    f" mean={stats['mean']:.4f}s std={stats['std']:.4f}s "
                    f"min={stats['min']:.4f}s max={stats['max']:.4f}s mem={mem_mb:.1f}MB "
                    f"samples={stats['samples']}"
                )
            except Exception as e:
                print(f" ERROR: {e}")

    out = common.save_results(results, args.estimator)
    print(f"\nResults written to {out}")

    print("\nSummary:")
    print(
        f"{'estimator':<12} {'n':>10} {'language':>10} {'mean_s':>12} "
        f"{'std_s':>12} {'min_s':>12} {'max_s':>12} {'mem_mb':>12} {'samples':>8}"
    )
    for r in results:
        mem_mb = r["memory_kb_mean"] / 1024
        print(
            f"{r['estimator']:<12} {r['n']:>10} {r['language']:>10} "
            f"{r['mean']:>12.4f} {r['std']:>12.4f} {r['min']:>12.4f} "
            f"{r['max']:>12.4f} {mem_mb:>12.1f} {r['samples']:>8}"
        )


if __name__ == "__main__":
    main()
