# Reference implementation in Python for system GMM on Wooldridge wagepan.
# Replicates the Blundell-Bond two-step System GMM estimator with:
#   - first-difference equations instrumented with lags 2 and 3 of y in levels,
#   - level equations instrumented with lagged first differences of y and X.
# This matches the implementation in Hayashi/Greeners.

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def build_system_gmm(y, X, entity_ids, time_ids, max_lags):
    """Build System GMM matrices and estimates two-step coefficients."""
    n_total = len(y)
    k_x = X.shape[1]

    # Sort by entity and time.
    ord_idx = np.lexsort((time_ids, entity_ids))
    ys = y[ord_idx]
    xs = X[ord_idx]
    ids = entity_ids[ord_idx]

    # Group by entity.
    entity_slices = []
    start = 0
    while start < n_total:
        eid = ids[start]
        end = start + 1
        while end < n_total and ids[end] == eid:
            end += 1
        entity_slices.append((start, end))
        start = end

    # Build first-difference and level data.
    dy_vec = []
    dyl_vec = []
    dx_rows = []
    zinst_fd = []

    y_lev = []
    yl_lev = []
    x_lev = []
    zinst_lv_base = []
    dx_lv_rows = []

    entity_fd_count = []
    entity_lev_count = []

    for s_start, s_end in entity_slices:
        t_i = s_end - s_start
        if t_i < 3:
            entity_fd_count.append(0)
            entity_lev_count.append(0)
            continue
        idx = list(range(s_start, s_end))
        for j in range(2, t_i):
            dy_vec.append(ys[idx[j]] - ys[idx[j - 1]])
            dyl_vec.append(ys[idx[j - 1]] - ys[idx[j - 2]])
            dx_rows.append(xs[idx[j]] - xs[idx[j - 1]])
            inst = []
            for l in range(max_lags):
                lag = l + 2
                inst.append(ys[idx[j - lag]] if j >= lag else 0.0)
            zinst_fd.append(inst)

            y_lev.append(ys[idx[j]])
            yl_lev.append(ys[idx[j - 1]])
            x_lev.append(xs[idx[j]])
            zinst_lv_base.append(ys[idx[j - 1]] - ys[idx[j - 2]])
            dx_lv_rows.append(xs[idx[j - 1]] - xs[idx[j - 2]])

        entity_fd_count.append(t_i - 2)
        entity_lev_count.append(t_i - 2)

    dy_vec = np.array(dy_vec)
    dyl_vec = np.array(dyl_vec)
    dx_rows = np.array(dx_rows)
    zinst_fd = np.array(zinst_fd)
    y_lev = np.array(y_lev)
    yl_lev = np.array(yl_lev)
    x_lev = np.array(x_lev)
    zinst_lv_base = np.array(zinst_lv_base)
    dx_lv_rows = np.array(dx_lv_rows)

    n_fd = len(dy_vec)
    n_lev = len(y_lev)
    n_sys = n_fd + n_lev

    # Active exogenous columns (non-constant after first-differencing).
    active_x = [c for c in range(k_x) if np.any(np.abs(dx_rows[:, c]) > 1e-12)]
    k_dx = len(active_x)
    k_reg = 1 + k_dx

    # Level instruments: lagged diff of y + lagged diffs of active exogenous cols.
    zinst_lv = []
    for dy, dx in zip(zinst_lv_base, dx_lv_rows):
        row = [dy]
        for oc in active_x:
            row.append(dx[oc])
        zinst_lv.append(row)
    zinst_lv = np.array(zinst_lv)

    n_inst_fd = max_lags + k_dx
    n_inst_lv = 1 + k_dx
    n_inst_sys = n_inst_fd + n_inst_lv

    # System matrices W and Z.
    w_sys = np.zeros((n_sys, k_reg))
    z_sys = np.zeros((n_sys, n_inst_sys))

    for i in range(n_fd):
        w_sys[i, 0] = dyl_vec[i]
        for nc, oc in enumerate(active_x):
            w_sys[i, 1 + nc] = dx_rows[i, oc]
            z_sys[i, max_lags + nc] = dx_rows[i, oc]
        for l in range(max_lags):
            z_sys[i, l] = zinst_fd[i, l]

    for i in range(n_lev):
        row = n_fd + i
        w_sys[row, 0] = yl_lev[i]
        for nc, oc in enumerate(active_x):
            w_sys[row, 1 + nc] = x_lev[i, oc]
            z_sys[row, n_inst_fd + 1 + nc] = zinst_lv[i, 1 + nc]
        z_sys[row, n_inst_fd] = zinst_lv[i, 0]

    # One-step weight matrix: (Z' H Z)^{-1}.
    zthz = np.zeros((n_inst_sys, n_inst_sys))
    rptr_fd = 0
    rptr_lev = n_fd
    for fc_fd, fc_lev in zip(entity_fd_count, entity_lev_count):
        if fc_fd == 0:
            continue
        zfd = z_sys[rptr_fd:rptr_fd + fc_fd, :]
        h_fd = np.zeros((fc_fd, fc_fd))
        for s in range(fc_fd):
            h_fd[s, s] = 2.0
            if s > 0:
                h_fd[s, s - 1] = -1.0
            if s < fc_fd - 1:
                h_fd[s, s + 1] = -1.0
        zthz += zfd.T @ h_fd @ zfd

        zlv = z_sys[rptr_lev:rptr_lev + fc_lev, :]
        zthz += zlv.T @ zlv

        rptr_fd += fc_fd
        rptr_lev += fc_lev

    a1 = np.linalg.inv(zthz)
    dy_sys = np.concatenate([dy_vec, y_lev])
    wtz = w_sys.T @ z_sys
    zty = z_sys.T @ dy_sys
    wtz_a1 = wtz @ a1
    lhs1 = wtz_a1 @ wtz.T
    lhs1_inv = np.linalg.inv(lhs1)
    params1 = lhs1_inv @ wtz_a1 @ zty
    resid1 = dy_sys - w_sys @ params1

    # Robust sandwich.
    sigma = np.zeros((n_inst_sys, n_inst_sys))
    rfd = 0
    rlev = n_fd
    for fc_fd, fc_lev in zip(entity_fd_count, entity_lev_count):
        if fc_fd == 0:
            continue
        fc = fc_fd + fc_lev
        z_ent = np.vstack([z_sys[rfd:rfd + fc_fd], z_sys[rlev:rlev + fc_lev]])
        u_ent = np.concatenate([resid1[rfd:rfd + fc_fd], resid1[rlev:rlev + fc_lev]])
        zu = z_ent.T @ u_ent
        sigma += np.outer(zu, zu)
        rfd += fc_fd
        rlev += fc_lev

    # Two-step GMM.
    a2 = np.linalg.inv(sigma)
    wtz_a2 = wtz @ a2
    lhs2 = wtz_a2 @ wtz.T
    lhs2_inv = np.linalg.inv(lhs2)
    params2 = lhs2_inv @ wtz_a2 @ zty
    std_errors = np.sqrt(np.maximum(np.diag(lhs2_inv), 0.0))

    return params2, std_errors


def compute_reference() -> dict:
    case_dir = Path("validation/cases/sysgmm_wagepan")
    data_dir = case_dir / "data"
    ref_dir = case_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "wagepan.csv")
    df = df.sort_values(["nr", "year"]).reset_index(drop=True)

    y = df["lwage"].to_numpy(dtype=float)
    X = df[["exper", "expersq", "married", "union"]].to_numpy(dtype=float)
    entity_ids = df["nr"].to_numpy(dtype=int)
    time_ids = df["year"].to_numpy(dtype=int)

    params, std_errors = build_system_gmm(y, X, entity_ids, time_ids, max_lags=2)

    names = ["lwage_lag", "exper", "expersq", "married", "union"]
    result = {
        "coefficients": {name: float(params[i]) for i, name in enumerate(names)},
        "standard_errors": {name: float(std_errors[i]) for i, name in enumerate(names)},
    }

    out_path = ref_dir / "expected.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    compute_reference()
