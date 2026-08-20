import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
events = df["time"].values
T = float(open(Path(__file__).resolve().parent.parent / "data" / "T.txt").read().strip())

def nll(theta):
    mu, alpha, beta = theta
    if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
        return 1e9
    comp = np.zeros(len(events))
    for i, t in enumerate(events):
        if i > 0:
            comp[i] = np.sum(np.exp(-beta * (t - events[:i])))
    lambdas = mu + alpha * comp
    if np.any(lambdas <= 0):
        return 1e9
    Lambda = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - events)))
    ll = np.sum(np.log(lambdas)) - Lambda
    return -ll

res = minimize(nll, [0.5, 0.3, 2.0], bounds=[(0.001, None), (0.0, None), (0.001, None)], method="L-BFGS-B")
mu, alpha, beta = res.x
result = {
    "coefficients": {
        "mu": float(mu),
        "alpha": float(alpha),
        "beta": float(beta),
        "branching_ratio": float(alpha / beta),
    },
    "standard_errors": {
        "mu": float("nan"),
        "alpha": float("nan"),
        "beta": float("nan"),
        "branching_ratio": float("nan"),
    },
}
print(json.dumps(result, indent=2))
