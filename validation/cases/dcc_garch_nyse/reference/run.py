# Reference implementation in Python for DCC-GARCH.

import numpy as np
import pandas as pd
import json
import os

# Load the CSV written by the R reference so both references agree.
data_dir = "validation/cases/dcc_garch_nyse/data"
nyse = pd.read_csv(f"{data_dir}/nyse.csv")

# Ensure the data directory exists.
os.makedirs(data_dir, exist_ok=True)

# Use only complete cases for the reference.
Z = nyse[['ret', 'return_1']].dropna().values
t, k = Z.shape

# GARCH(1,1) grid search
def garch11(r):
    var_init = float(np.mean(r**2))
    best = [0.01, 0.05, 0.90]
    best_ll = float('-inf')
    best_vols = None
    n_grid = 8
    for oi in range(n_grid):
        omega = 0.001 + 0.1 * oi / (n_grid - 1) * var_init
        for ai in range(n_grid):
            alpha = 0.01 + 0.3 * ai / (n_grid - 1)
            for bi in range(n_grid):
                beta = 0.5 + 0.48 * bi / (n_grid - 1)
                if alpha + beta >= 0.99:
                    continue
                vols = np.full(len(r), var_init)
                ll = 0.0
                for i in range(len(r)):
                    if i > 0:
                        vols[i] = omega + alpha * r[i-1]**2 + beta * vols[i-1]
                    vol = max(vols[i], 1e-10)
                    ll += -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (r[i]**2) / vol
                if ll > best_ll:
                    best_ll = ll
                    best = [omega, alpha, beta]
                    best_vols = vols.copy()
    return {'params': best, 'vols': best_vols, 'll': best_ll}

std_resids = np.zeros((t, k))
conditional_vols = np.zeros((t, k))
garch_params = np.zeros((k, 3))
garch_ll = 0.0
for j in range(k):
    fit = garch11(Z[:, j])
    garch_params[j, :] = fit['params']
    conditional_vols[:, j] = fit['vols']
    std_resids[:, j] = Z[:, j] / np.maximum(fit['vols'], 1e-10)**0.5
    garch_ll += fit['ll']

# Unconditional correlation of standardized residuals
q_bar = np.zeros((k, k))
for i in range(t):
    s = std_resids[i, :]
    q_bar += np.outer(s, s)
q_bar /= t

# DCC log-likelihood
def dcc_ll(alpha, beta):
    q_prev = q_bar.copy()
    ll = 0.0
    for i in range(t):
        s = std_resids[i, :]
        q_t = (1 - alpha - beta) * q_bar + alpha * np.outer(s, s) + beta * q_prev
        d_inv = 1.0 / np.maximum(np.sqrt(np.diag(q_t)), 1e-10)
        r_t = np.diag(d_inv) @ q_t @ np.diag(d_inv)
        r_det = max(np.linalg.det(r_t), 1e-300)
        r_inv = np.linalg.inv(r_t + np.eye(k) * 1e-8)
        quad = float(s @ r_inv @ s)
        ll += -0.5 * (np.log(r_det) + quad)
        q_prev = q_t
    return ll

# DCC grid search
best_alpha = 0.01
best_beta = 0.95
best_ll = float('-inf')
n_grid = 15
for i in range(n_grid):
    for j in range(n_grid):
        alpha = 0.01 + 0.48 * i / (n_grid - 1)
        beta = 0.01 + 0.48 * j / (n_grid - 1)
        if alpha + beta >= 0.99:
            continue
        ll = dcc_ll(alpha, beta)
        if ll > best_ll:
            best_ll = ll
            best_alpha = alpha
            best_beta = beta

total_ll = garch_ll + best_ll
n_params = k * 3 + 2
aic = -2 * total_ll + 2 * n_params
bic = -2 * total_ll + t * n_params

result = {
    'dcc_alpha': best_alpha,
    'dcc_beta': best_beta,
    'ret_omega': float(garch_params[0, 0]),
    'ret_alpha': float(garch_params[0, 1]),
    'ret_beta': float(garch_params[0, 2]),
    'return_1_omega': float(garch_params[1, 0]),
    'return_1_alpha': float(garch_params[1, 1]),
    'return_1_beta': float(garch_params[1, 2]),
    'log_likelihood': total_ll,
    'aic': aic,
    'bic': bic
}

out_dir = "validation/cases/dcc_garch_nyse/reference"
os.makedirs(out_dir, exist_ok=True)

with open(f"{out_dir}/expected.json", 'w') as f:
    json.dump(result, f, indent=2)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
print(json.dumps(result))
