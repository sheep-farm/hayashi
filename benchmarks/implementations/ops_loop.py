#!/usr/bin/env python3
"""Benchmark a plain integer loop."""

import sys
import time


def main():
    n = int(sys.argv[1])
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    def work():
        s = 0
        for i in range(1, n + 1):
            s += i
        return s

    for _ in range(warmup):
        work()

    for _ in range(iters):
        t0 = time.perf_counter()
        work()
        t1 = time.perf_counter()
        print(f"  elapsed: {t1 - t0:.6f}s")


if __name__ == "__main__":
    main()
