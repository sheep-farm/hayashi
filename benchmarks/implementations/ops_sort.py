#!/usr/bin/env python3
"""Benchmark sorting by a column."""

import sys
import time

import pandas as pd


def main():
    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    df = pd.read_csv(path)

    for _ in range(warmup):
        df.sort_values("x")

    for _ in range(iters):
        t0 = time.perf_counter()
        df.sort_values("x")
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.6f}s")


if __name__ == "__main__":
    main()
