# Explicit Python reference for the Wooldridge wagepan panel FE two-way-clustered-SE case.
#
# Within-transformed OLS with two-way (entity + time) clustered covariance.
# Mirrors the Greeners implementation: V = sandwich(X, meat_1 + meat_2 - meat_12, X) * g/(g-1) * (n-1)/(n-k)
# where g = min(G_entity, G_time).

import json
from pathlib import Path

import numpy as np
import pandas as pd

def _checked_covariance_rejection(result: dict) -> int:
    """Require this fixed case's finite, material d81/eigenvalue rejection."""
    names = list(result["covariance"])
    covariance = np.array([[result["covariance"][a][b] for b in names] for a in names], dtype=float)
    diagnostics = result["covariance_diagnostics"]
    values = diagnostics["eigenvalues"]
    if values is None or not np.isfinite(covariance).all():
        raise ValueError("Availability requires finite raw covariance and its spectrum")
    eigenvalues = np.asarray(values, dtype=float)
    bounds = [diagnostics["symmetry_error"], diagnostics["symmetry_bound"], diagnostics["eigenvalue_bound"]]
    if any(value is None for value in bounds) or not np.isfinite(bounds).all() or not np.isfinite(eigenvalues).all():
        raise ValueError("Availability requires finite covariance diagnostics")
    reason = result["inference_reason"] or ""
    expected_rejection = bool(
        result["inference_available"] is False
        and diagnostics["finite"] is True
        and diagnostics["symmetry_error"] <= diagnostics["symmetry_bound"]
        and covariance[names.index("d81"), names.index("d81")] < -diagnostics["symmetry_bound"]
        and np.any(eigenvalues < -diagnostics["eigenvalue_bound"])
        and "negative diagonal variance for d81" in reason
        and "materially negative eigenvalues" in reason
    )
    if not expected_rejection:
        raise ValueError("Expected finite materially indefinite d81 covariance; this availability contract is not met")
    return int(expected_rejection)


CASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = CASE_DIR / "data" / "wagepan.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError("wagepan.csv is missing; run data/gen.py first")

variables = [
    "lwage",
    "union",
    "married",
    "d81",
    "d82",
    "d83",
    "d84",
    "d85",
    "d86",
    "d87",
    "nr",
    "year",
]
x_names = ["union", "married", "d81", "d82", "d83", "d84", "d85", "d86", "d87"]

entity_name = "nr"
time_name = "year"

df = pd.read_csv(CSV_PATH)[variables].dropna()

# Within-transformation by entity.
y = df["lwage"] - df.groupby(entity_name)["lwage"].transform("mean")
X = df[x_names] - df.groupby(entity_name)[x_names].transform("mean")

y_arr = y.to_numpy(dtype=float)
x_arr = X.to_numpy(dtype=float)

n, k = x_arr.shape
if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all() or n <= k or np.linalg.matrix_rank(x_arr) != k:
    raise ValueError("Ordinary within design must be finite, full-rank and have residual degrees of freedom")
xtx_inv = np.linalg.inv(x_arr.T @ x_arr)
beta = xtx_inv @ x_arr.T @ y_arr
residuals = y_arr - x_arr @ beta

if not np.isfinite(beta).all():
    raise ValueError("Ordinary within coefficients must be finite")


def cluster_meat(cluster_col):
    clusters = df[cluster_col].to_numpy()
    unique_clusters = pd.unique(clusters)
    meat = np.zeros((k, k))
    for cluster in unique_clusters:
        idx = clusters == cluster
        x_g = x_arr[idx, :]
        u_g = residuals[idx]
        meat += x_g.T @ np.outer(u_g, u_g) @ x_g
    return meat, len(unique_clusters)


df["inter"] = df[entity_name].astype(str) + "_" + df[time_name].astype(str)
meat_1, g1 = cluster_meat(entity_name)
meat_2, g2 = cluster_meat(time_name)
meat_12, _ = cluster_meat("inter")

meat = meat_1 + meat_2 - meat_12
sandwich = xtx_inv @ meat @ xtx_inv

g = min(g1, g2)
if g <= 1:
    raise ValueError("Two-way covariance requires at least two groups in each dimension")
finite_sample_correction = (g / (g - 1)) * ((n - 1) / (n - k))
vcov = finite_sample_correction * sandwich
# Admissibility is assessed without changing any covariance entry. The factor
# 128*k*eps declares the floating-point budget relative to matrix/spectral scale.
roundoff_multiplier = 128
finite_covariance = bool(np.isfinite(vcov).all())
negative_variance_terms = [name for name, value in zip(x_names, np.diag(vcov)) if np.isfinite(value) and value < 0]
eigenvalues = None
symmetry_error = None
symmetry_bound = None
eigenvalue_bound = None
reasons = []
if not finite_covariance:
    reasons.append("covariance contains non-finite entries")
else:
    symmetry_error = float(np.max(np.abs(vcov - vcov.T)))
    symmetry_bound = float(roundoff_multiplier * k * np.finfo(float).eps * np.max(np.abs(vcov)))
    # The symmetric part is used only for the spectral diagnostic; raw vcov is retained.
    eigenvalues = np.linalg.eigvalsh(0.5 * vcov + 0.5 * vcov.T)
    eigenvalue_bound = float(roundoff_multiplier * k * np.finfo(float).eps * np.max(np.abs(eigenvalues)))
    if symmetry_error > symmetry_bound:
        reasons.append("covariance is materially asymmetric")
    if negative_variance_terms:
        reasons.append("negative diagonal variance for " + ", ".join(negative_variance_terms))
    if np.any(eigenvalues < -eigenvalue_bound):
        reasons.append("covariance has materially negative eigenvalues")


def _json_number(value: float) -> float | str:
    """Preserve finite numbers and label IEEE non-finite values in valid JSON."""
    if np.isfinite(value):
        return float(value)
    if np.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


inference_available = not reasons
result = {
    "coefficients": {name: float(value) for name, value in zip(x_names, beta)},
    "covariance": {
        row_name: {column_name: _json_number(value) for column_name, value in zip(x_names, row)}
        for row_name, row in zip(x_names, vcov)
    },
    "inference_available": inference_available,
    "inference_reason": "; ".join(reasons) if reasons else None,
    "covariance_diagnostics": {
        "finite": finite_covariance,
        "eigenvalues": None if eigenvalues is None else [float(value) for value in eigenvalues],
        "symmetry_error": symmetry_error,
        "symmetry_bound": symmetry_bound,
        "eigenvalue_bound": eigenvalue_bound,
        "roundoff_multiplier": roundoff_multiplier,
        "negative_variance_terms": negative_variance_terms,
    },
}
result["availability"] = {"covariance_rejected": _checked_covariance_rejection(result)}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, allow_nan=False)

print(json.dumps(result, allow_nan=False))
