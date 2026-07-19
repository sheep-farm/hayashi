#!/usr/bin/env python3
"""Generate synthetic datasets for Hayashi benchmarks."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.special import erf
except ImportError:
    import math

    erf = np.vectorize(math.erf, otypes=[float])


def ols_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.normal(0, 1, size=n)
    eps = rng.normal(0, 1, size=n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + 0.5 * x3 + eps
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


def logit_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    z = 1.0 + 1.5 * x1 - 0.8 * x2
    pr = 1.0 / (1.0 + np.exp(-z))
    y = rng.binomial(1, pr).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def arima_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.5 + 0.7 * y[t - 1] + rng.normal()
    return pd.DataFrame({"y": y, "t": np.arange(n)})


def garch_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    h = np.ones(n)
    for t in range(1, n):
        h[t] = 0.05 + 0.1 * y[t - 1] ** 2 + 0.85 * h[t - 1]
        y[t] = np.sqrt(h[t]) * rng.normal()
    return pd.DataFrame({"y": y})


def panel_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # total n = n_firms * t
    t = 10
    n_firms = max(1, n // t)
    rows = []
    for firm in range(n_firms):
        alpha = rng.normal(0, 1)
        for year in range(t):
            x = rng.normal(0, 1)
            eps = rng.normal(0, 1)
            y = alpha + 1.0 + 2.0 * x + eps
            rows.append({"firm": firm, "year": year, "y": y, "x": x})
    df = pd.DataFrame(rows)
    # keep size close to n
    if len(df) > n:
        df = df.iloc[:n].reset_index(drop=True)
    return df


def iv_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, size=n)
    u = rng.normal(0, 1, size=n)
    # v correlated with u to create endogeneity
    rho = 0.7
    v = rho * u + rng.normal(0, np.sqrt(1 - rho**2), size=n)
    x = 1.0 + 0.8 * z + v
    y = 1.0 + 2.0 * x + u
    return pd.DataFrame({"y": y, "x": x, "z": z})


def probit_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    z = 1.0 + 1.5 * x1 - 0.8 * x2
    # standard normal CDF using error function
    pr = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    y = rng.binomial(1, pr).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def qreg_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    # add some heteroskedasticity to make quantile interesting
    eps = rng.normal(0, 1 + 0.5 * x1.abs(), size=n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + eps
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def var_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = np.array([0.5, 0.3])
    A = np.array([[0.7, 0.2], [0.1, 0.6]])
    # burn-in to reach stationarity
    burn = 100
    y_prev = rng.normal(0, 1, size=2)
    for _ in range(burn):
        e = rng.normal(0, 1, size=2)
        y_prev = c + A @ y_prev + e
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    for t in range(n):
        e = rng.normal(0, 1, size=2)
        y_prev = c + A @ y_prev + e
        y1[t] = y_prev[0]
        y2[t] = y_prev[1]
    return pd.DataFrame({"y1": y1, "y2": y2})


GENERATORS = {
    "ols": ols_data,
    "logit": logit_data,
    "arima": arima_data,
    "garch": garch_data,
    "panel": panel_data,
    "iv": iv_data,
    "probit": probit_data,
    "qreg": qreg_data,
    "var": var_data,
}


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark datasets")
    parser.add_argument("--estimator", default="ols", choices=list(GENERATORS.keys()))
    parser.add_argument("--sizes", default="1000,10000,100000", help="comma-separated sizes")
    parser.add_argument("--output", default=str(Path(__file__).parent / "generated"))
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    gen = GENERATORS[args.estimator]
    for n in sizes:
        df = gen(n)
        path = out_dir / f"{args.estimator}_n{n}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path}: {len(df)} rows, {list(df.columns)}")


if __name__ == "__main__":
    main()
