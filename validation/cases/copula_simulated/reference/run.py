import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["y"].values
x = df["x"].values

tau, _ = stats.kendalltau(y, x)
spearman, _ = stats.spearmanr(y, x)
# Gaussian copula parameter derived from Kendall's tau: rho = sin(pi * tau / 2)
corr = np.sin(np.pi * tau / 2)

result = {
    "corr_yx": float(corr),
    "kendall_yx": float(tau),
    "spearman_yx": float(spearman),
}

print(json.dumps(result))
