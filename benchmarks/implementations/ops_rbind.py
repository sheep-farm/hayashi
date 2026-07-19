#!/usr/bin/env python3
"""Benchmark vertical concatenation (rbind) with pandas."""

import sys
import time

import pandas as pd


def main():
    left_path = sys.argv[1]
    right_path = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)

    for _ in range(warmup):
        pd.concat([left, right])

    for _ in range(iters):
        t0 = time.perf_counter()
        pd.concat([left, right])
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.6f}s")


if __name__ == "__main__":
    main()
