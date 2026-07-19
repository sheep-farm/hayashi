#!/usr/bin/env python3
"""Quantile regression benchmark with statsmodels."""

import sys
import time

import pandas as pd
import statsmodels.api as sm


def main():
    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    df = pd.read_csv(path)
    y = df["y"]
    X = sm.add_constant(df[["x1", "x2"]])

    # warmup
    for _ in range(warmup):
        sm.QuantReg(y, X).fit(q=0.5)

    for _ in range(iters):
        t0 = time.perf_counter()
        sm.QuantReg(y, X).fit(q=0.5)
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.4f}s")


if __name__ == "__main__":
    main()
