#!/usr/bin/env python3
"""Benchmark function call overhead."""

import sys
import time


def main():
    _ = int(sys.argv[1])  # n is unused; iters controls number of calls
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    def f(x):
        return x + 1

    for i in range(warmup):
        f(i)

    for i in range(iters):
        t0 = time.perf_counter()
        f(i)
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.6f}s")


if __name__ == "__main__":
    main()
