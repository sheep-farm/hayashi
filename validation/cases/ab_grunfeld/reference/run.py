#!/usr/bin/env python3
"""Python reference for the Arellano-Bond Grunfeld validation case.

This is an explicit one-step difference-GMM implementation. It mirrors the
contract in ``greeners::dynamic_panel::ArellanoBond::fit`` and the base-R
reference, rather than relying on a package default with its own instrument and
weight conventions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
REF_DIR = CASE_DIR / "reference"
CSV_PATH = DATA_DIR / "grunfeld.csv"


def ensure_data() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        from wooldridge import data as wd_data

        df = wd_data("grunfeld")
        df.to_csv(CSV_PATH, index=False)
    return pd.read_csv(CSV_PATH)


def build_ab_matrices(
    y: np.ndarray,
    x: np.ndarray,
    entity_ids: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Build dy, W, Z and entity equation counts for diff-GMM."""
    order = np.lexsort((time_ids, entity_ids))
    ys = y[order]
    xs = x[order]
    ids = entity_ids[order]
    times = time_ids[order]

    slices: list[tuple[int, int]] = []
    start = 0
    while start < len(ys):
        entity = ids[start]
        end = start + 1
        while end < len(ys) and ids[end] == entity:
            end += 1
        slices.append((start, end))
        start = end

    dy_rows: list[float] = []
    dyl_rows: list[float] = []
    dx_rows: list[np.ndarray] = []
    z_lag_rows: list[list[float]] = []
    entity_fd_count: list[int] = []

    for start, end in slices:
        idx = list(range(start, end))
        count = 0
        if len(idx) < 3:
            entity_fd_count.append(0)
            continue

        for j in range(2, len(idx)):
            if times[idx[j]] != times[idx[j - 1]] + 1:
                continue
            if times[idx[j - 1]] != times[idx[j - 2]] + 1:
                continue

            dy_rows.append(ys[idx[j]] - ys[idx[j - 1]])
            dyl_rows.append(ys[idx[j - 1]] - ys[idx[j - 2]])
            dx_rows.append(xs[idx[j], :] - xs[idx[j - 1], :])

            entity_times = {times[k]: ys[k] for k in idx}
            instruments = []
            for lag in range(max_lags):
                target_time = times[idx[j]] - (lag + 2)
                instruments.append(float(entity_times.get(target_time, 0.0)))
            z_lag_rows.append(instruments)
            count += 1

        entity_fd_count.append(count)

    if not dy_rows:
        raise ValueError("No effective first-difference equations")

    dy = np.asarray(dy_rows, dtype=float)
    dyl = np.asarray(dyl_rows, dtype=float)
    dx = np.asarray(dx_rows, dtype=float)
    z_lags = np.asarray(z_lag_rows, dtype=float)

    active_x = [c for c in range(x.shape[1]) if np.any(np.abs(dx[:, c]) > 1e-12)]
    k_dx = len(active_x)
    k_reg = 1 + k_dx
    n_inst = max_lags + k_dx
    if n_inst < k_reg:
        raise ValueError(f"Under-identified: {n_inst} instruments < {k_reg} regressors")

    w_mat = np.zeros((len(dy), k_reg))
    z_mat = np.zeros((len(dy), n_inst))
    w_mat[:, 0] = dyl
    z_mat[:, :max_lags] = z_lags
    for new_col, old_col in enumerate(active_x):
        w_mat[:, 1 + new_col] = dx[:, old_col]
        z_mat[:, max_lags + new_col] = dx[:, old_col]

    return dy, w_mat, z_mat, entity_fd_count


def fit_arellano_bond(
    y: np.ndarray,
    x: np.ndarray,
    entity_ids: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    dy, w_mat, z_mat, entity_fd_count = build_ab_matrices(
        y, x, entity_ids, time_ids, max_lags
    )

    n_inst = z_mat.shape[1]
    zthz = np.zeros((n_inst, n_inst))
    row = 0
    for count in entity_fd_count:
        if count == 0:
            continue
        zi = z_mat[row : row + count, :]
        h_i = np.zeros((count, count))
        for s in range(count):
            h_i[s, s] = 2.0
            if s > 0:
                h_i[s, s - 1] = -1.0
            if s < count - 1:
                h_i[s, s + 1] = -1.0
        zthz += zi.T @ h_i @ zi
        row += count

    a1 = np.linalg.inv(zthz)
    wtz = w_mat.T @ z_mat
    zty = z_mat.T @ dy
    wtz_a1 = wtz @ a1
    lhs = wtz_a1 @ wtz.T
    lhs_inv = np.linalg.inv(lhs)
    params = lhs_inv @ wtz_a1 @ zty

    residuals = dy - w_mat @ params
    sigma = np.zeros((n_inst, n_inst))
    row = 0
    for count in entity_fd_count:
        if count == 0:
            continue
        zi = z_mat[row : row + count, :]
        ui = residuals[row : row + count]
        zui = zi.T @ ui
        sigma += np.outer(zui, zui)
        row += count

    meat = wtz_a1 @ sigma @ a1 @ wtz.T
    variance = lhs_inv @ meat @ lhs_inv
    std_errors = np.sqrt(np.maximum(np.diag(variance), 0.0))
    return params, std_errors


def compute_reference() -> dict[str, dict[str, float]]:
    df = ensure_data().sort_values(["firm", "year"]).reset_index(drop=True)

    params, std_errors = fit_arellano_bond(
        y=df["inv"].to_numpy(dtype=float),
        x=df[["value", "capital"]].to_numpy(dtype=float),
        entity_ids=df["firm"].to_numpy(dtype=int),
        time_ids=df["year"].to_numpy(dtype=int),
        max_lags=1,
    )

    labels = ["LD.y", "Δvalue", "Δcapital"]
    result = {
        "coefficients": {
            label: float(params[index]) for index, label in enumerate(labels)
        },
        "standard_errors": {
            label: float(std_errors[index]) for index, label in enumerate(labels)
        },
    }

    REF_DIR.mkdir(parents=True, exist_ok=True)
    with open(REF_DIR / "expected.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    compute_reference()
