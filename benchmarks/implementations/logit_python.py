#!/usr/bin/env python3
"""Benchmark Logit com statsmodels."""

import json
import statistics
import sys
import time

import pandas as pd
import statsmodels.api as sm


def main():
    path = sys.argv[1]
    reps = int(sys.argv[2])
    df = pd.read_csv(path)
    y = df["y"]
    X = sm.add_constant(df[["x1", "x2"]])

    # warmup
    sm.Logit(y, X).fit(disp=0)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model = sm.Logit(y, X).fit(disp=0)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print(
        json.dumps(
            {
                "mean": sum(times) / len(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0.0,
                "min": min(times),
                "max": max(times),
                "reps": reps,
            }
        )
    )


if __name__ == "__main__":
    main()
