#!/usr/bin/env python3
"""Hayashi benchmark orchestrator against R/Python."""

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

ESTIMATORS = {
    "ols": {
        "hay": 'load "{dataset}" as df\nfor i in 1..{reps} {{\n    let m = ols(y ~ x1 + x2 + x3, df)\n}}\nprint("done")\n',
        "python": "ols_python.py",
        "r": "ols_r.R",
    },
    "logit": {
        "hay": 'load "{dataset}" as df\nfor i in 1..{reps} {{\n    let m = logit(y ~ x1 + x2, df)\n}}\nprint("done")\n',
        "python": "logit_python.py",
        "r": "logit_r.R",
    },
    "arima": {
        "hay": 'load "{dataset}" as df\nfor i in 1..{reps} {{\n    let m = arima(df, y, p=1, d=0, q=0)\n}}\nprint("done")\n',
        "python": "arima_python.py",
        "r": "arima_r.R",
    },
    "garch": {
        "hay": 'load "{dataset}" as df\nfor i in 1..{reps} {{\n    let m = garch(df, y, p=1, q=1)\n}}\nprint("done")\n',
        "python": "garch_python.py",
        "r": "garch_r.R",
    },
    "panel": {
        "hay": 'load "{dataset}" as df\nfor i in 1..{reps} {{\n    let m = fe(y ~ x, df, id=firm)\n}}\nprint("done")\n',
        "python": "panel_python.py",
        "r": "panel_r.R",
    },
}


def generate_datasets(estimator: str, sizes: list[int]) -> list[Path]:
    gen = BENCH_DIR / "datasets" / "generate.py"
    sizes_str = ",".join(str(s) for s in sizes)
    subprocess.run(
        [sys.executable, str(gen), "--estimator", estimator, "--sizes", sizes_str],
        check=True,
    )
    return [DATASETS_DIR / f"{estimator}_n{n}.csv" for n in sizes]


def run_hayashi(estimator: str, dataset: Path, reps: int) -> dict:
    config = ESTIMATORS[estimator]
    script = TMP_DIR / f"{estimator}_hayashi_{dataset.stem}.hay"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(config["hay"].format(dataset=dataset, reps=reps))
    cmd = [str(HAY_EXE), str(script)]
    return _measure_command(cmd, reps, warmup=True)


def run_python(estimator: str, dataset: Path, reps: int) -> dict:
    script = BENCH_DIR / "implementations" / ESTIMATORS[estimator]["python"]
    cmd = [sys.executable, str(script), str(dataset), str(reps)]
    return _measure_command(cmd, reps, warmup=True)


def run_r(estimator: str, dataset: Path, reps: int) -> dict:
    script = BENCH_DIR / "implementations" / ESTIMATORS[estimator]["r"]
    cmd = ["Rscript", str(script), str(dataset), str(reps)]
    return _measure_command(cmd, reps, warmup=True)


def _read_rss_kb(pid: int) -> int:
    """Read VmRSS from /proc/<pid>/status in KB."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])  # KB
    except Exception:
        pass
    return 0


def _find_child_pids(ppid: int) -> list[int]:
    """Return PIDs whose PPid is ppid."""
    children = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(entry.name)
            if pid == ppid:
                continue
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        if int(line.split()[1]) == ppid:
                            children.append(pid)
                        break
        except Exception:
            continue
    return children


def _sample_memory_kb(pid: int) -> int:
    """Sum RSS of the main process plus children."""
    total = _read_rss_kb(pid)
    for child in _find_child_pids(pid):
        total += _read_rss_kb(child)
    return total


def _measure_command(cmd: list[str], reps: int, warmup: bool = True) -> dict:
    """Run the command several times, discard warmup, return statistics."""
    if warmup:
        # first run discarded for cache/startup
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    times = []
    memory_peaks_kb = []
    for _ in range(reps):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        peak_kb = 0
        t0 = time.perf_counter()
        while proc.poll() is None:
            peak_kb = max(peak_kb, _sample_memory_kb(proc.pid))
            time.sleep(0.01)
        stdout, stderr = proc.communicate()
        t1 = time.perf_counter()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, stdout, stderr)
        times.append(t1 - t0)
        memory_peaks_kb.append(peak_kb)

    return {
        "mean": sum(times) / len(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "raw": times,
        "memory_kb_mean": sum(memory_peaks_kb) / len(memory_peaks_kb),
        "memory_kb_max": max(memory_peaks_kb),
        "memory_kb_std": statistics.stdev(memory_peaks_kb) if len(memory_peaks_kb) > 1 else 0.0,
    }


def save_results(results: list[dict], estimator: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{estimator}_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description="Run Hayashi benchmarks")
    parser.add_argument("--estimator", default="ols", choices=list(ESTIMATORS.keys()), help="estimator to benchmark")
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
        "hayashi": run_hayashi,
        "python": run_python,
        "r": run_r,
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
                stats = runners[lang](args.estimator, dataset, args.reps)
                stats["estimator"] = args.estimator
                stats["language"] = lang
                stats["n"] = n
                stats["dataset"] = str(dataset)
                results.append(stats)
                mem_mb = stats['memory_kb_mean'] / 1024
                print(f" mean={stats['mean']:.4f}s std={stats['std']:.4f}s mem={mem_mb:.1f}MB")
            except Exception as e:
                print(f" ERROR: {e}")

    out = save_results(results, args.estimator)
    print(f"\nResults written to {out}")

    # summary table
    print("\nSummary:")
    print(f"{'estimator':<12} {'n':>10} {'language':>10} {'mean_s':>12} {'std_s':>12} {'mem_mb':>12}")
    for r in results:
        mem_mb = r['memory_kb_mean'] / 1024
        print(f"{r['estimator']:<12} {r['n']:>10} {r['language']:>10} {r['mean']:>12.4f} {r['std']:>12.4f} {mem_mb:>12.1f}")


if __name__ == "__main__":
    main()
