# Reference implementation in Python for xtgls on Wooldridge wagepan.
# Replicates the Stata-style panel feasible GLS with panels(heteroskedastic)
# as implemented in Hayashi/Greeners.

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def compute_reference() -> dict:
    case_dir = Path("validation/cases/xtgls_wagepan")
    data_dir = case_dir / "data"
    ref_dir = case_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "wagepan.csv")

    # Sort by entity and time to match the panel extraction order in Greeners.
    df = df.sort_values(["nr", "year"]).reset_index(drop=True)

    y = df["lwage"].to_numpy(dtype=float)
    X = pd.DataFrame({"const": 1.0, "educ": df["educ"], "exper": df["exper"],
                      "expersq": df["expersq"], "married": df["married"],
                      "union": df["union"]})
    x_cols = ["const", "educ", "exper", "expersq", "married", "union"]
    X = X[x_cols].to_numpy(dtype=float)

    entity_ids = df["nr"].to_numpy(dtype=int)
    time_ids = df["year"].to_numpy(dtype=int)

    # Extract balanced panels (same logic as Greeners::extract_balanced_panels).
    unique_entities = np.unique(entity_ids)
    unique_times = np.unique(time_ids)
    n_entities = len(unique_entities)
    big_t = len(unique_times)
    k = X.shape[1]

    entity_to_idx = {e: i for i, e in enumerate(unique_entities)}
    time_to_idx = {t: i for i, t in enumerate(unique_times)}

    y_panels = []
    x_panels = []
    for e in unique_entities:
        mask = entity_ids == e
        sub = df.loc[mask].sort_values("year")
        if len(sub) != big_t:
            raise ValueError(f"Panel {e} is unbalanced: {len(sub)} observations")
        y_panels.append(sub["lwage"].to_numpy(dtype=float))
        x_panels.append(np.column_stack([
            np.ones(big_t),
            sub["educ"].to_numpy(dtype=float),
            sub["exper"].to_numpy(dtype=float),
            sub["expersq"].to_numpy(dtype=float),
            sub["married"].to_numpy(dtype=float),
            sub["union"].to_numpy(dtype=float),
        ]))

    # Step 1: pooled OLS for initial residuals.
    xtx0 = np.zeros((k, k))
    xty0 = np.zeros(k)
    for xi, yi in zip(x_panels, y_panels):
        xtx0 += xi.T @ xi
        xty0 += xi.T @ yi
    beta0 = np.linalg.solve(xtx0, xty0)
    resid0 = [yi - xi @ beta0 for yi, xi in zip(y_panels, x_panels)]

    # Step 2: estimate diagonal Omega and compute (X' Omega^{-1} X) and (X' Omega^{-1} y).
    # For panels=hetero: sigma2_i = e_i' e_i / T.
    xtox = np.zeros((k, k))
    xtoy = np.zeros(k)
    for i in range(n_entities):
        sigma2_i = resid0[i] @ resid0[i] / big_t
        if sigma2_i < 1e-15:
            raise ValueError(f"sigma2_i is near zero for entity {i}")
        w = 1.0 / sigma2_i
        xtox += x_panels[i].T @ x_panels[i] * w
        xtoy += x_panels[i].T @ y_panels[i] * w

    # Step 3: beta_fgls = (X' Omega^{-1} X)^{-1} X' Omega^{-1} y.
    xtox_inv = np.linalg.inv(xtox)
    beta = xtox_inv @ xtoy

    # Residuals and sigma.
    resid_gls = [yi - xi @ beta for yi, xi in zip(y_panels, x_panels)]
    ssr_gls = sum(e @ e for e in resid_gls)
    df_resid = n_entities * big_t - k
    sigma = math.sqrt(ssr_gls / df_resid)

    # Standard errors from (X' Omega^{-1} X)^{-1}, asymptotic (Normal).
    std_errors = np.sqrt(np.maximum(np.diag(xtox_inv), 0.0))
    z_values = beta / std_errors
    p_values = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(z_values / math.sqrt(2))))

    result = {
        "coefficients": {name: float(beta[i]) for i, name in enumerate(x_cols)},
        "standard_errors": {name: float(std_errors[i]) for i, name in enumerate(x_cols)},
    }

    out_path = ref_dir / "expected.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    compute_reference()
