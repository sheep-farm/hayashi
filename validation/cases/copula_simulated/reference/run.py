import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["y"].values
x = df["x"].values

# Convert to pseudo-observations (rank / (n + 1))
n = len(y)
u = stats.rankdata(y) / (n + 1)
v = stats.rankdata(x) / (n + 1)

# Gaussian copula: rho = sin(pi/2 * tau) from Kendall's tau
tau, _ = stats.kendalltau(y, x)
rho = np.sin(np.pi / 2 * tau)

# Log-likelihood of bivariate Gaussian copula
Z0 = stats.norm.ppf(u)
Z1 = stats.norm.ppf(v)

# bivariate normal log-likelihood
log_dens = (
    -np.log(2 * np.pi)
    - 0.5 * np.log(1 - rho ** 2)
    - 0.5 / (1 - rho ** 2) * (Z0 ** 2 + Z1 ** 2 - 2 * rho * Z0 * Z1)
)
ll = float(np.sum(log_dens))
k = 1
aic = 2 * k - 2 * ll
bic = k * np.log(n) - 2 * ll

result = {
    "theta": float(rho),
    "kendall_tau_yx": float(tau),
    "spearman_rho_yx": float(stats.spearmanr(y, x).correlation),
    "log_likelihood": ll,
    "aic": float(aic),
    "bic": float(bic),
}

print(json.dumps(result, indent=2))
