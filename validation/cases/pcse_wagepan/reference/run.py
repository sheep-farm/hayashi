# Manual Python reference for PCSE on Wooldridge wagepan.
#
# This mirrors the Hayashi/Greeners convention:
#   beta = (X'X)^-1 X'y
#   sigma_ij = e_i'e_j / T
#   meat = sum_i sum_j sigma_ij X_i'X_j
#   V = (X'X)^-1 meat (X'X)^-1

import json
from pathlib import Path

import numpy as np
import pandas as pd


CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
REF_DIR = CASE_DIR / "reference"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "wagepan.csv"


def load_data() -> pd.DataFrame:
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)

    try:
        from wooldridge import data

        df = data("wagepan")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/wagepan.csv"
        df = pd.read_csv(url)
        if "rownames" in df.columns:
            df = df.drop(columns=["rownames"])
    df.to_csv(CSV_PATH, index=False)
    return df


def compute_reference() -> dict:
    df = load_data()
    vars_ = ["nr", "year", "lwage", "educ", "exper", "expersq", "married", "union"]
    df = df[vars_].dropna().sort_values(["nr", "year"]).reset_index(drop=True)

    x_cols = ["const", "educ", "exper", "expersq", "married", "union"]
    k = len(x_cols)

    entities = sorted(df["nr"].astype(int).unique())
    counts = df.groupby("nr").size()
    if counts.nunique() != 1:
        raise ValueError("wagepan panel is not balanced")
    big_t = int(counts.iloc[0])

    y_panels = []
    x_panels = []
    time_template = None
    for entity in entities:
        sub = df[df["nr"] == entity].sort_values("year")
        times = tuple(sub["year"].astype(int).to_numpy())
        if time_template is None:
            time_template = times
        elif times != time_template:
            raise ValueError("wagepan panel time indexes differ across entities")

        y_panels.append(sub["lwage"].to_numpy(dtype=float))
        x_panels.append(
            np.column_stack(
                [
                    np.ones(big_t, dtype=float),
                    sub["educ"].to_numpy(dtype=float),
                    sub["exper"].to_numpy(dtype=float),
                    sub["expersq"].to_numpy(dtype=float),
                    sub["married"].to_numpy(dtype=float),
                    sub["union"].to_numpy(dtype=float),
                ]
            )
        )

    xtx = np.zeros((k, k), dtype=float)
    xty = np.zeros(k, dtype=float)
    for xi, yi in zip(x_panels, y_panels):
        xtx += xi.T @ xi
        xty += xi.T @ yi

    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ xty

    resid_panels = [yi - xi @ beta for yi, xi in zip(y_panels, x_panels)]
    n_entities = len(entities)

    sigma_hat = np.zeros((n_entities, n_entities), dtype=float)
    for i in range(n_entities):
        for j in range(i, n_entities):
            sigma_ij = float(resid_panels[i] @ resid_panels[j] / big_t)
            sigma_hat[i, j] = sigma_ij
            sigma_hat[j, i] = sigma_ij

    meat = np.zeros((k, k), dtype=float)
    for i in range(n_entities):
        for j in range(n_entities):
            meat += (x_panels[i].T @ x_panels[j]) * sigma_hat[i, j]

    vcov = xtx_inv @ meat @ xtx_inv
    variance_diag = np.diag(vcov)
    if np.any(variance_diag < -1e-12):
        raise ValueError("PCSE covariance has a negative diagonal entry")
    std_errors = np.sqrt(np.maximum(variance_diag, 0.0))

    result = {
        "coefficients": {name: float(beta[i]) for i, name in enumerate(x_cols)},
        "standard_errors": {name: float(std_errors[i]) for i, name in enumerate(x_cols)},
    }

    with open(REF_DIR / "expected.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result))
    return result


if __name__ == "__main__":
    compute_reference()
