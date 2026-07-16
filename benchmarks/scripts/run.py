#!/usr/bin/env python3
"""Orquestrador de benchmarks Hayashi vs R/Python."""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BENCH_DIR.parent
HAY_EXE = ROOT_DIR / "target" / "release" / "hay"
DATASETS_DIR = BENCH_DIR / "datasets" / "generated"
RESULTS_DIR = BENCH_DIR / "results"
TMP_DIR = BENCH_DIR / ".tmp"


def generate_datasets(estimator: str, sizes: list[int]) -> list[Path]:
    gen = BENCH_DIR / "datasets" / "generate.py"
    sizes_str = ",".join(str(s) for s in sizes)
    subprocess.run(
        [sys.executable, str(gen), "--estimator", estimator, "--sizes", sizes_str],
        check=True,
    )
    return [DATASETS_DIR / f"{estimator}_n{n}.csv" for n in sizes]


def run_hayashi_ols(dataset: Path, reps: int) -> dict:
    """Benchmark OLS no Hayashi."""
    script = TMP_DIR / f"ols_hayashi_{dataset.stem}.hay"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        f'load "{dataset}" as df\n'
        f"for i in 1..{reps} {{\n"
        f"    let m = ols(y ~ x1 + x2 + x3, df)\n"
        f"}}\n"
        f'print("done")\n'
    )
    cmd = [str(HAY_EXE), str(script)]
    return _measure_command(cmd, reps, warmup=True)


def run_python_ols(dataset: Path, reps: int) -> dict:
    script = BENCH_DIR / "implementations" / "ols_python.py"
    return _run_python_script(script, dataset, reps)


def run_r_ols(dataset: Path, reps: int) -> dict:
    script = BENCH_DIR / "implementations" / "ols_r.R"
    return _run_r_script(script, dataset, reps)


def _run_python_script(script: Path, dataset: Path, reps: int) -> dict:
    cmd = [sys.executable, str(script), str(dataset), str(reps)]
    return _measure_command(cmd, reps, warmup=True)


def _run_r_script(script: Path, dataset: Path, reps: int) -> dict:
    cmd = ["Rscript", str(script), str(dataset), str(reps)]
    return _measure_command(cmd, reps, warmup=True)


def _measure_command(cmd: list[str], reps: int, warmup: bool = True) -> dict:
    """Roda o comando varias vezes, descarta warmup, retorna estatisticas."""
    if warmup:
        # primeira execucao descartada para cache/startup
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "mean": sum(times) / len(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "raw": times,
    }


def save_results(results: list[dict], estimator: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{estimator}_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description="Run Hayashi benchmarks")
    parser.add_argument("--estimator", default="ols", choices=["ols"], help="estimator to benchmark")
    parser.add_argument("--sizes", default="1000,10000,100000", help="comma-separated dataset sizes")
    parser.add_argument("--reps", type=int, default=5, help="repetitions per size")
    parser.add_argument("--lang", default="hayashi,python,r", help="languages to benchmark")
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    langs = [l.strip().lower() for l in args.lang.split(",")]

    if not HAY_EXE.exists():
        print(f"Hayashi binary not found at {HAY_EXE}; build with: cargo build --release")
        sys.exit(1)

    print(f"Benchmarking {args.estimator} with sizes {sizes}")
    datasets = generate_datasets(args.estimator, sizes)

    runners = {
        "hayashi": run_hayashi_ols,
        "python": run_python_ols,
        "r": run_r_ols,
    }

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
                stats = runners[lang](dataset, args.reps)
                stats["estimator"] = args.estimator
                stats["language"] = lang
                stats["n"] = n
                stats["dataset"] = str(dataset)
                results.append(stats)
                print(f" mean={stats['mean']:.4f}s std={stats['std']:.4f}s")
            except Exception as e:
                print(f" ERROR: {e}")

    out = save_results(results, args.estimator)
    print(f"\nResults written to {out}")

    # tabela resumo
    print("\nSummary:")
    print(f"{'estimator':<12} {'n':>10} {'language':>10} {'mean_s':>12} {'std_s':>12}")
    for r in results:
        print(f"{r['estimator']:<12} {r['n']:>10} {r['language']:>10} {r['mean']:>12.4f} {r['std']:>12.4f}")


if __name__ == "__main__":
    main()
