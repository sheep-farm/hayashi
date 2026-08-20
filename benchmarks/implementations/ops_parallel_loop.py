#!/usr/bin/env python3
"""Benchmark parallel CSV load + rbind with ordered output.

Each iteration loads `n` CSV files in parallel using multiprocessing.Pool,
then concatenates them with pandas (the closest standard-library equivalent
to Hayashi's `parallel for` + `rbind`).
"""

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

BENCH_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BENCH_DIR / "datasets" / "generated" / "ops" / "parallel_loop"
ROWS_PER_FILE = 1_000


def _generate_files(n: int) -> None:
    """Generate the same CSV files that the Hayashi benchmark expects."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(1, n + 1):
        path = CSV_DIR / f"file_{i}.csv"
        if path.exists():
            continue
        ids = np.arange(1, ROWS_PER_FILE + 1)
        x = i + rng.random(ROWS_PER_FILE)
        y = rng.random(ROWS_PER_FILE)
        pd.DataFrame({"id": ids, "x": x, "y": y}).to_csv(path, index=False)


def load_one(i: int) -> pd.DataFrame:
    """Load a single CSV; called by Pool workers in order."""
    return pd.read_csv(CSV_DIR / f"file_{i}.csv")


def main():
    n = int(sys.argv[1])
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    _generate_files(n)

    workers = os.cpu_count() or 1
    chunksize = max(1, n // (workers * 4))

    with Pool(processes=workers) as pool:
        for _ in range(warmup):
            dfs = pool.map(load_one, range(1, n + 1), chunksize=chunksize)
            pd.concat(dfs, ignore_index=True)

        for _ in range(iters):
            t0 = time.perf_counter()
            dfs = pool.map(load_one, range(1, n + 1), chunksize=chunksize)
            pd.concat(dfs, ignore_index=True)
            t1 = time.perf_counter()
            print(f"  elapsed: {t1 - t0:.6f}s")


if __name__ == "__main__":
    main()
