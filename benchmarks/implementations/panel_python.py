#!/usr/bin/env python3
"""Benchmark Fixed Effects panel com linearmodels."""

import json
import statistics
import sys
import time

import pandas as pd
from linearmodels.panel import PanelOLS


def main():
    path = sys.argv[1]
    reps = int(sys.argv[2])
    df = pd.read_csv(path)
    df = df.set_index(["firm", "year"])

    # warmup
    PanelOLS.from_formula("y ~ x + EntityEffects", df).fit()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model = PanelOLS.from_formula("y ~ x + EntityEffects", df).fit()
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
