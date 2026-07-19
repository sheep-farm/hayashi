#!/usr/bin/env python3
"""Benchmark Hayashi DataFrame and language operations.

This runner focuses on single operations (load, generate, filter, sort,
merge, group, loops, etc.). It reuses the measurement helpers from
`common.py` and produces `results/ops_<op>_<timestamp>.json`.
"""

import argparse
import sys
from pathlib import Path

import common

BENCH_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = BENCH_DIR / ".tmp"

# Dataset cache directory under benchmarks/datasets/generated
def _ops_dir():
    d = common.DATASETS_DIR / "ops"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Each op declares:
#   - sizes: default n (or file rows) to benchmark
#   - timer: whether the script emits per-call `elapsed:` lines (False = use wall time)
#   - needs_csv / needs_n / needs_merge: what data to prepare
#   - hay_script: template with placeholders {n}, {csv}, {left_csv}, {right_csv}, {iters}, {warmup}
#   - python_script: optional comparison script (None = Hayashi-only)
#   - r_script: optional R script (None = not implemented)
OPS = {
    "load_csv": {
        "sizes": [100_000, 1_000_000],
        "timer": False,
        "needs_csv": True,
        "hay_script": 'load "{csv}" as df\nprint("done")',
        "python_script": "ops_load_csv.py",
        "r_script": None,
    },
    "generate_random": {
        "sizes": [1_000, 10_000, 100_000],
        "timer": True,
        "needs_n": True,
        "hay_script": (
            "let df = dataframe({n})\n"
            "generate df x = rnormal(0, 1)\n"
            "for i in 1..={warmup} {{ let _ = mutate(df, r = rnormal(0, 1)) }}\n"
            "for i in 1..={iters} {{ let _ = timer(mutate(df, r = rnormal(0, 1)), digits=6) }}\n"
            'print("done")\n'
        ),
        "python_script": "ops_generate_random.py",
        "r_script": None,
    },
    "generate_expr": {
        "sizes": [1_000, 10_000, 100_000],
        "timer": True,
        "needs_n": True,
        "hay_script": (
            "let df = dataframe({n})\n"
            "generate df x = rnormal(0, 1)\n"
            "generate df y = rnormal(0, 1)\n"
            "for i in 1..={warmup} {{ let _ = mutate(df, z = (x + y) * 2) }}\n"
            "for i in 1..={iters} {{ let _ = timer(mutate(df, z = (x + y) * 2), digits=6) }}\n"
            'print("done")\n'
        ),
        "python_script": "ops_generate_expr.py",
        "r_script": None,
    },
    "filter": {
        "sizes": [1_000, 10_000, 100_000],
        "timer": True,
        "needs_csv": True,
        "hay_script": (
            'load "{csv}" as df\n'
            "for i in 1..={warmup} {{ let _ = filter(df, x > 0) }}\n"
            "for i in 1..={iters} {{ let _ = timer(filter(df, x > 0), digits=6) }}\n"
            'print("done")\n'
        ),
        "python_script": "ops_filter.py",
        "r_script": None,
    },
    "sort": {
        "sizes": [1_000, 10_000, 100_000],
        "timer": True,
        "needs_csv": True,
        "hay_script": (
            'load "{csv}" as df\n'
            "for i in 1..={warmup} {{ let _ = sort(df, x) }}\n"
            "for i in 1..={iters} {{ let _ = timer(sort(df, x), digits=6) }}\n"
            'print("done")\n'
        ),
        "python_script": "ops_sort.py",
        "r_script": None,
    },
    "groupby_mean": {
        "sizes": [1_000, 10_000, 100_000],
        "timer": True,
        "needs_csv": True,
        "hay_script": (
            'load "{csv}" as df\n'
            "for i in 1..={warmup} {{ let _ = group_by(df, group, mean, x) }}\n"
            "for i in 1..={iters} {{ let _ = timer(group_by(df, group, mean, x), digits=6) }}\n"
            'print("done")\n'
        ),
        "python_script": "ops_groupby_mean.py",
        "r_script": None,
    },
    "merge": {
        "sizes": [1_000, 10_000, 100_000],
        "timer": True,
        "needs_merge": True,
        "hay_script": (
            'load "{left_csv}" as df1\n'
            'load "{right_csv}" as df2\n'
            "for i in 1..={warmup} {{ let _ = merge(df1, df2, key=id) }}\n"
            "for i in 1..={iters} {{ let _ = timer(merge(df1, df2, key=id), digits=6) }}\n"
            'print("done")\n'
        ),
        "python_script": "ops_merge.py",
        "r_script": None,
    },
    "loop": {
        "sizes": [1_000, 10_000],
        "timer": True,
        "needs_n": True,
        "hay_script": (
            "fn work() {{\n"
            "    let s = 0\n"
            "    for i in 1..={n} {{ let s = s + i }}\n"
            "    return s\n"
            "}}\n"
            "for i in 1..={warmup} {{ let _ = work() }}\n"
            "for i in 1..={iters} {{ let _ = timer(work(), digits=6) }}\n"
            'print("done")\n'
        ),
        "python_script": "ops_loop.py",
        "r_script": None,
    },
    "function_call": {
        "sizes": [1],
        "timer": True,
        "needs_n": True,
        "hay_script": (
            "fn f(x) {{ return x + 1 }}\n"
            "for i in 1..={warmup} {{ let _ = f(i) }}\n"
            "for i in 1..={iters} {{ let _ = timer(f(i), digits=6) }}\n"
            'print("done")\n'
        ),
        "python_script": "ops_function_call.py",
        "r_script": None,
    },
}


def _csv_path(n: int) -> Path:
    return _ops_dir() / f"ops_n{n}.csv"


def _left_csv_path(n: int) -> Path:
    return _ops_dir() / f"ops_left_n{n}.csv"


def _right_csv_path(n: int) -> Path:
    return _ops_dir() / f"ops_right_n{n}.csv"


def _generate_csv(n: int) -> Path:
    """Generate a single CSV with id, x, y, group."""
    import numpy as np
    import pandas as pd

    path = _csv_path(n)
    if path.exists():
        return path
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "x": rng.normal(0, 1, size=n),
        "y": rng.normal(0, 1, size=n),
        "group": np.arange(n) % 10,
    })
    df.to_csv(path, index=False)
    print(f"  generated {path}")
    return path


def _generate_merge_csvs(n: int) -> tuple[Path, Path]:
    """Generate two CSVs with matching id columns for merge benchmark."""
    import numpy as np
    import pandas as pd

    left = _left_csv_path(n)
    right = _right_csv_path(n)
    if left.exists() and right.exists():
        return left, right
    rng = np.random.default_rng(42)
    ids = np.arange(1, n + 1)
    df_left = pd.DataFrame({"id": ids, "x": rng.normal(0, 1, size=n)})
    df_right = pd.DataFrame({"id": ids, "y": rng.normal(0, 1, size=n)})
    df_left.to_csv(left, index=False)
    df_right.to_csv(right, index=False)
    print(f"  generated {left}, {right}")
    return left, right


def _write_hayashi_script(op_name: str, op: dict, n: int, iters: int, warmup: int) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    script = TMP_DIR / f"ops_{op_name}_n{n}.hay"
    fmt = {"n": n, "iters": iters, "warmup": warmup}
    if op.get("needs_csv"):
        fmt["csv"] = _csv_path(n)
    if op.get("needs_merge"):
        fmt["left_csv"] = _left_csv_path(n)
        fmt["right_csv"] = _right_csv_path(n)
    source = op["hay_script"].format(**fmt)
    script.write_text(source)
    return script


def _run_timed(cmd: list[str], iters: int, warmup: int, runs: int) -> dict:
    all_times = []
    memory_peaks = []
    wall_times = []
    for _ in range(runs):
        times, peak_kb, wall = common.measure_run(cmd)
        all_times.extend(times)
        memory_peaks.append(peak_kb)
        wall_times.append(wall)
    return common.aggregate(all_times, memory_peaks, wall_times, iters, warmup, runs)


def _run_wall_timed(cmd: list[str], runs: int) -> dict:
    """For ops that cannot emit per-call elapsed lines (e.g. load)."""
    memory_peaks = []
    wall_times = []
    for _ in range(runs):
        _, peak_kb, wall = common.measure_run(cmd)
        memory_peaks.append(peak_kb)
        wall_times.append(wall)
    # each run is one call; wall_time becomes the per-call time
    return common.aggregate(wall_times, memory_peaks, wall_times, 1, 0, runs)


def run_hayashi(op_name: str, n: int, iters: int, warmup: int, runs: int) -> dict:
    op = OPS[op_name]
    script = _write_hayashi_script(op_name, op, n, iters, warmup)
    cmd = [str(common.HAY_EXE), str(script)]
    if op.get("timer", True):
        return _run_timed(cmd, iters, warmup, runs)
    return _run_wall_timed(cmd, runs)


def _python_script(op: dict) -> Path | None:
    name = op.get("python_script")
    if not name:
        return None
    return BENCH_DIR / "implementations" / name


def _r_script(op: dict) -> Path | None:
    name = op.get("r_script")
    if not name:
        return None
    return BENCH_DIR / "implementations" / name


def run_python(op_name: str, n: int, iters: int, warmup: int, runs: int) -> dict:
    op = OPS[op_name]
    script = _python_script(op)
    if script is None or not script.exists():
        raise FileNotFoundError(f"Python script for {op_name} not found: {script}")
    cmd = [sys.executable, str(script)]
    if op.get("needs_csv"):
        cmd.append(str(_csv_path(n)))
    elif op.get("needs_merge"):
        left, right = _generate_merge_csvs(n)
        cmd.extend([str(left), str(right)])
    elif op.get("needs_n"):
        cmd.append(str(n))
    cmd.extend([str(iters), str(warmup)])
    if op.get("timer", True):
        return _run_timed(cmd, iters, warmup, runs)
    return _run_wall_timed(cmd, runs)


def run_r(op_name: str, n: int, iters: int, warmup: int, runs: int) -> dict:
    op = OPS[op_name]
    script = _r_script(op)
    if script is None or not script.exists():
        raise FileNotFoundError(f"R script for {op_name} not found: {script}")
    cmd = ["Rscript", str(script)]
    if op.get("needs_csv"):
        cmd.append(str(_csv_path(n)))
    elif op.get("needs_merge"):
        left, right = _generate_merge_csvs(n)
        cmd.extend([str(left), str(right)])
    elif op.get("needs_n"):
        cmd.append(str(n))
    cmd.extend([str(iters), str(warmup)])
    if op.get("timer", True):
        return _run_timed(cmd, iters, warmup, runs)
    return _run_wall_timed(cmd, runs)


def _prepare_data(op_name: str, n: int) -> None:
    op = OPS[op_name]
    if op.get("needs_csv"):
        _generate_csv(n)
    if op.get("needs_merge"):
        _generate_merge_csvs(n)


def main():
    all_ops = ",".join(OPS.keys())
    parser = argparse.ArgumentParser(description="Run Hayashi DataFrame/language operation benchmarks")
    parser.add_argument(
        "--op",
        default=all_ops,
        help=f"comma-separated operations to run (default: {all_ops})",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="comma-separated dataset sizes (overrides per-op defaults)",
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
        default="hayashi,python",
        help="comma-separated languages to benchmark (hayashi, python, r)",
    )
    args = parser.parse_args()

    if not common.HAY_EXE.exists():
        print(f"Hayashi binary not found at {common.HAY_EXE}; build with: cargo build --release")
        sys.exit(1)

    selected_ops = [o.strip() for o in args.op.split(",")]
    for op_name in selected_ops:
        if op_name not in OPS:
            print(f"Unknown op: {op_name}")
            sys.exit(1)

    langs = [l.strip().lower() for l in args.lang.split(",")]
    runners = {
        "hayashi": run_hayashi,
        "python": run_python,
        "r": run_r,
    }

    results = []
    for op_name in selected_ops:
        op = OPS[op_name]
        sizes = [int(s.strip()) for s in args.sizes.split(",")] if args.sizes else op["sizes"]
        print(f"\n=== Operation: {op_name} ===")
        for n in sizes:
            print(f"\n  n={n}")
            _prepare_data(op_name, n)
            for lang in langs:
                if lang not in runners:
                    print(f"    skip unknown language: {lang}")
                    continue
                # load_csv benchmarks one call per run; iters is ignored
                iters = 1 if not op.get("timer", True) else args.iters
                print(f"    running {lang}...", end="", flush=True)
                try:
                    stats = runners[lang](op_name, n, iters, args.warmup, args.runs)
                    stats["op"] = op_name
                    stats["estimator"] = op_name
                    stats["language"] = lang
                    stats["n"] = n
                    results.append(stats)
                    mem_mb = stats["memory_kb_mean"] / 1024
                    print(
                        f" mean={stats['mean']:.4f}s std={stats['std']:.4f}s "
                        f"min={stats['min']:.4f}s max={stats['max']:.4f}s mem={mem_mb:.1f}MB "
                        f"samples={stats['samples']}"
                    )
                except Exception as e:
                    print(f" ERROR: {e}")

    out = common.save_results(results, "ops")
    print(f"\nResults written to {out}")

    print("\nSummary:")
    print(
        f"{'op':<16} {'n':>10} {'language':>10} {'mean_s':>12} "
        f"{'std_s':>12} {'min_s':>12} {'max_s':>12} {'mem_mb':>12} {'samples':>8}"
    )
    for r in results:
        mem_mb = r["memory_kb_mean"] / 1024
        print(
            f"{r['op']:<16} {r['n']:>10} {r['language']:>10} "
            f"{r['mean']:>12.4f} {r['std']:>12.4f} {r['min']:>12.4f} "
            f"{r['max']:>12.4f} {mem_mb:>12.1f} {r['samples']:>8}"
        )


if __name__ == "__main__":
    main()
