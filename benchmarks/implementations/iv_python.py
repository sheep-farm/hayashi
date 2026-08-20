#!/usr/bin/env python3
"""IV/2SLS benchmark with linearmodels."""

import sys
import time

import pandas as pd
from linearmodels.iv import IV2SLS


def main():
    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    df = pd.read_csv(path)

    # warmup
    for _ in range(warmup):
        IV2SLS.from_formula("y ~ 1 + [x ~ z]", df).fit()

    for _ in range(iters):
        t0 = time.perf_counter()
        IV2SLS.from_formula("y ~ 1 + [x ~ z]", df).fit()
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.6f}s")


if __name__ == "__main__":
    main()
