#!/usr/bin/env python3
"""Hayashi benchmark orchestrator against R and Python.

Each implementation script is run as a subprocess and must emit one
`  elapsed: X.XXXXs` line per timed iteration. The runner parses these
lines, measures peak RSS, and aggregates statistics across runs.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BENCH_DIR.parent
HAY_EXE = ROOT_DIR / "target" / "release" / "hay"
DATASETS_DIR = BENCH_DIR / "datasets" / "generated"
RESULTS_DIR = BENCH_DIR / "results"
TMP_DIR = BENCH_DIR / ".tmp"
ELAPSED_RE = re.compile(r"^\s*elapsed:\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$", re.IGNORECASE)

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
    "panel": {
        "hay": "fe(y ~ x, df, id=firm)",
        "python": "panel_python.py",
        "r": "panel_r.R",
    },
}


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


def _parse_elapsed(stdout: bytes) -> list[float]:
    """Extract elapsed times (in seconds) from stdout."""
    times = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        m = ELAPSED_RE.match(line)
        if m:
            times.append(float(m.group(1)))
    return times


def _measure_run(cmd: list[str]) -> tuple[list[float], int, float]:
    """Run cmd once, return (elapsed_times, peak_memory_kb, wall_time)."""
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
    wall_time = time.perf_counter() - t0
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, stdout, stderr
        )
    times = _parse_elapsed(stdout)
    return times, peak_kb, wall_time


def generate_datasets(estimator: str, sizes: list[int]) -> list[Path]:
    gen = BENCH_DIR / "datasets" / "generate.py"
    sizes_str = ",".join(str(s) for s in sizes)
    subprocess.run(
        [sys.executable, str(gen), "--estimator", estimator, "--sizes", sizes_str],
        check=True,
    )
    return [DATASETS_DIR / f"{estimator}_n{n}.csv" for n in sizes]


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
    cmd = [str(HAY_EXE), str(script)]
    all_times = []
    memory_peaks = []
    wall_times = []
    for _ in range(runs):
        times, peak_kb, wall = _measure_run(cmd)
        all_times.extend(times)
        memory_peaks.append(peak_kb)
        wall_times.append(wall)
    return _aggregate(all_times, memory_peaks, wall_times, iters, warmup, runs)


def run_python(
    estimator: str, dataset: Path, iters: int, warmup: int, runs: int
) -> dict:
    script = BENCH_DIR / "implementations" / ESTIMATORS[estimator]["python"]
    cmd = [sys.executable, str(script), str(dataset), str(iters), str(warmup)]
    all_times = []
    memory_peaks = []
    wall_times = []
    for _ in range(runs):
        times, peak_kb, wall = _measure_run(cmd)
        all_times.extend(times)
        memory_peaks.append(peak_kb)
        wall_times.append(wall)
    return _aggregate(all_times, memory_peaks, wall_times, iters, warmup, runs)


def run_r(
    estimator: str, dataset: Path, iters: int, warmup: int, runs: int
) -> dict:
    script = BENCH_DIR / "implementations" / ESTIMATORS[estimator]["r"]
    cmd = ["Rscript", str(script), str(dataset), str(iters), str(warmup)]
    all_times = []
    memory_peaks = []
    wall_times = []
    for _ in range(runs):
        times, peak_kb, wall = _measure_run(cmd)
        all_times.extend(times)
        memory_peaks.append(peak_kb)
        wall_times.append(wall)
    return _aggregate(all_times, memory_peaks, wall_times, iters, warmup, runs)


def _aggregate(
    times: list[float],
    memory_peaks: list[int],
    wall_times: list[float],
    iters: int,
    warmup: int,
    runs: int,
) -> dict:
    if len(times) < 2:
        std = 0.0
    else:
        std = statistics.stdev(times)
    return {
        "mean": statistics.mean(times) if times else 0.0,
        "std": std,
        "min": min(times) if times else 0.0,
        "max": max(times) if times else 0.0,
        "wall_time": statistics.mean(wall_times) if wall_times else 0.0,
        "memory_kb_mean": statistics.mean(memory_peaks) if memory_peaks else 0,
        "memory_kb_max": max(memory_peaks) if memory_peaks else 0,
        "memory_kb_std": statistics.stdev(memory_peaks)
        if len(memory_peaks) > 1
        else 0.0,
        "iters": iters,
        "warmup": warmup,
        "runs": runs,
        "samples": len(times),
    }


def save_results(results: list[dict], estimator: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{estimator}_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


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

    if not HAY_EXE.exists():
        print(f"Hayashi binary not found at {HAY_EXE}; build with: cargo build --release")
        sys.exit(1)

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    langs = [l.strip().lower() for l in args.lang.split(",")]

    runners = {
        "hayashi": run_hayashi,
        "python": run_python,
        "r": run_r,
    }

    print(f"Benchmarking {args.estimator}: sizes={sizes}, iters={args.iters}, warmup={args.warmup}, runs={args.runs}")
    datasets = generate_datasets(args.estimator, sizes)

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

    out = save_results(results, args.estimator)
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
