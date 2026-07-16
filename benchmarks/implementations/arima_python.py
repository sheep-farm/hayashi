#!/usr/bin/env python3
"""ARIMA benchmark with statsmodels."""

import json
import statistics
import sys
import time

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def main():
    path = sys.argv[1]
    reps = int(sys.argv[2])
    df = pd.read_csv(path)
    y = df["y"]

    # warmup
    ARIMA(y, order=(1, 0, 0)).fit()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model = ARIMA(y, order=(1, 0, 0)).fit()
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
