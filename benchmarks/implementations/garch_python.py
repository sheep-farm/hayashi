#!/usr/bin/env python3
"""GARCH(1,1) benchmark with arch."""

import sys
import time

import pandas as pd
from arch import arch_model


def main():
    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    df = pd.read_csv(path)
    y = df["y"] * 100  # arch works better with percent-scale returns

    # warmup
    for _ in range(warmup):
        arch_model(y, vol="GARCH", p=1, q=1).fit(update_freq=0, disp="off")

    for _ in range(iters):
        t0 = time.perf_counter()
        arch_model(y, vol="GARCH", p=1, q=1).fit(update_freq=0, disp="off")
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.6f}s")


if __name__ == "__main__":
    main()
