#!/usr/bin/env python3
"""GARCH(1,1) benchmark with arch."""

import json
import statistics
import sys
import time

import pandas as pd
from arch import arch_model


def main():
    path = sys.argv[1]
    reps = int(sys.argv[2])
    df = pd.read_csv(path)
    y = df["y"] * 100  # arch works better with percent-scale returns

    # warmup
    arch_model(y, vol="GARCH", p=1, q=1).fit(update_freq=0, disp="off")

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model = arch_model(y, vol="GARCH", p=1, q=1).fit(update_freq=0, disp="off")
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
