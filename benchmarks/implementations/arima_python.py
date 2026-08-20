#!/usr/bin/env python3
"""ARIMA benchmark with statsmodels."""

import sys
import time

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def main():
    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    df = pd.read_csv(path)
    y = df["y"]

    # warmup
    for _ in range(warmup):
        ARIMA(y, order=(1, 0, 0)).fit()

    for _ in range(iters):
        t0 = time.perf_counter()
        ARIMA(y, order=(1, 0, 0)).fit()
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.6f}s")


if __name__ == "__main__":
    main()
