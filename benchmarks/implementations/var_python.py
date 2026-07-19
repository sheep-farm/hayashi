#!/usr/bin/env python3
"""VAR benchmark with statsmodels."""

import sys
import time

import pandas as pd
from statsmodels.tsa.api import VAR


def main():
    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    df = pd.read_csv(path)[["y1", "y2"]]

    # warmup
    for _ in range(warmup):
        VAR(df).fit(maxlags=1, trend="c")

    for _ in range(iters):
        t0 = time.perf_counter()
        VAR(df).fit(maxlags=1, trend="c")
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.4f}s")


if __name__ == "__main__":
    main()
