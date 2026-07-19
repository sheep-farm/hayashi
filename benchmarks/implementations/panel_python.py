#!/usr/bin/env python3
"""Fixed Effects panel benchmark with linearmodels."""

import sys
import time

import pandas as pd
from linearmodels.panel import PanelOLS


def main():
    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    df = pd.read_csv(path)
    df = df.set_index(["firm", "year"])

    # warmup
    for _ in range(warmup):
        PanelOLS.from_formula("y ~ x + EntityEffects", df).fit()

    for _ in range(iters):
        t0 = time.perf_counter()
        PanelOLS.from_formula("y ~ x + EntityEffects", df).fit()
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.4f}s")


if __name__ == "__main__":
    main()
