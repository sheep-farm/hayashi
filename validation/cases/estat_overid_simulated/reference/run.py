import scipy.stats as stats

import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
Y = df["y"].to_numpy()
X = np.column_stack((np.ones(len(df)), df["x"].to_numpy()))
Z = np.column_stack((np.ones(len(df)), df["z1"].to_numpy(), df["z2"].to_numpy()))
n = len(Y)
# 2SLS
Pi = np.linalg.solve(Z.T @ Z, Z.T @ X)
Xhat = Z @ Pi
b2sls = np.linalg.solve(Xhat.T @ X, Xhat.T @ Y)
u = Y - X @ b2sls
# Sargan = n * (u' P_Z u) / (u' u)
Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)
stat = n * float((u @ Pz @ u) / (u @ u))
pval = 1 - float(stats.chi2.cdf(stat, 1))

result = {
    "coefficients": {"j_stat": stat, "p_value": pval},
    "standard_errors": {"j_stat": 0.0, "p_value": 0.0},
}
print(json.dumps(result))
