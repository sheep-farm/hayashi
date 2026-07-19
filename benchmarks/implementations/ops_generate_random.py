#!/usr/bin/env python3
"""Benchmark adding a random normal column."""

import sys
import time

import numpy as np
import pandas as pd


def main():
    n = int(sys.argv[1])
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(0, 1, size=n)})

    for _ in range(warmup):
        df["r"] = rng.normal(0, 1, size=n)

    for _ in range(iters):
        t0 = time.perf_counter()
        df["r"] = rng.normal(0, 1, size=n)
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.4f}s")


if __name__ == "__main__":
    main()
