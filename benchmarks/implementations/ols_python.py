#!/usr/bin/env python3
"""OLS benchmark with statsmodels."""

import json
import sys
import time

import pandas as pd
import statsmodels.api as sm


def main():
    path = sys.argv[1]
    reps = int(sys.argv[2])
    df = pd.read_csv(path)
    y = df["y"]
    X = sm.add_constant(df[["x1", "x2", "x3"]])

    # warmup
    sm.OLS(y, X).fit()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model = sm.OLS(y, X).fit()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print(
        json.dumps(
            {
                "mean": sum(times) / len(times),
                "std": __import__("statistics").stdev(times) if len(times) > 1 else 0.0,
                "min": min(times),
                "max": max(times),
                "reps": reps,
            }
        )
    )


if __name__ == "__main__":
    main()
