"""Full-vector robust STL reference; no Hayashi or R values enter this fit."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import statsmodels
from statsmodels.tsa.seasonal import STL

DATA = Path(__file__).resolve().parents[1] / "data/data.csv"
NOBS = 144
COUNT_SHA256 = "8c999fa9d67e1dff475b9d0d82996f06cc5b7a2598ab359814fbb1c2f4644afd"
COMPONENTS = ("trend", "seasonal", "remainder")
SETTINGS = dict(period=12, seasonal=7, trend=23, low_pass=23,
                seasonal_deg=1, trend_deg=1, low_pass_deg=1,
                seasonal_jump=1, trend_jump=1, low_pass_jump=1)


def vector(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (NOBS,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite vector of length 144")
    return values


def load_counts(path=DATA):
    with Path(path).open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["index", "passengers"]:
            raise ValueError("Expected index,passengers columns")
        rows = list(reader)
    if len(rows) != NOBS:
        raise ValueError("Expected 144 observations")
    counts = []
    for index, row in enumerate(rows, 1):
        value = row["passengers"]
        if (set(row) != {"index", "passengers"} or row["index"] != str(index)
                or value is None or not value.isascii() or not value.isdecimal()
                or int(value) <= 0 or str(int(value)) != value):
            raise ValueError("Expected ordered indices and canonical positive integers")
        counts.append(int(value))
    canonical = "".join(f"{value}\n" for value in counts).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != COUNT_SHA256:
        raise ValueError("AirPassengers count SHA256 mismatch")
    return vector(counts)


def check_components(observed, components, expected):
    observed = vector(observed)
    expected = vector(expected)
    if not np.array_equal(observed, expected):
        raise ValueError("Observed series differs from the input")
    if set(components) != set(COMPONENTS):
        raise ValueError("Expected trend, seasonal and remainder")
    trend, seasonal, remainder = (vector(components[key]) for key in COMPONENTS)
    error = float(np.max(np.abs(observed - trend - seasonal - remainder)))
    if error > 1e-12:
        raise ValueError(f"Reconstruction exceeds 1e-12: {error}")
    return error


def fit_stl(observed, outer=1):
    if statsmodels.__version__ != "0.14.6":
        raise ValueError("This contract requires statsmodels 0.14.6")
    observed = vector(observed)
    # Use a new model for each diagnostic; fit() reuses internal work arrays.
    fit = STL(observed, robust=(outer != 0), **SETTINGS).fit(inner_iter=2, outer_iter=outer)
    components = dict(trend=fit.trend, seasonal=fit.seasonal, remainder=fit.resid)
    check_components(fit.observed, components, observed)
    vector(fit.weights)
    return fit, components


def reference_output():
    observed = np.log(load_counts())
    _, components = fit_stl(observed)
    values = {}
    for name, series in {"observed": observed, **components}.items():
        values.update({f"{name}_{i}": float(value) for i, value in enumerate(series, 1)})
    # Zero is the reconstruction target, not a second measured residual to subtract.
    values.update(nobs=NOBS, reconstruction_max=0.0)
    return {"coefficients": values}


if __name__ == "__main__":
    print(json.dumps(reference_output(), allow_nan=False))
