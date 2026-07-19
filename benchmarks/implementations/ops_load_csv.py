#!/usr/bin/env python3
"""Benchmark CSV loading with pandas."""

import sys
import time

import pandas as pd


def main():
    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    for _ in range(warmup):
        pd.read_csv(path)

    for _ in range(iters):
        t0 = time.perf_counter()
        pd.read_csv(path)
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.6f}s")


if __name__ == "__main__":
    main()
