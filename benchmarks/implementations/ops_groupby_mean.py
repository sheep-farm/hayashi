#!/usr/bin/env python3
"""Benchmark groupby mean aggregation."""

import sys
import time

import pandas as pd


def main():
    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    df = pd.read_csv(path)

    for _ in range(warmup):
        df.groupby("group")["x"].mean()

    for _ in range(iters):
        t0 = time.perf_counter()
        df.groupby("group")["x"].mean()
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.4f}s")


if __name__ == "__main__":
    main()
