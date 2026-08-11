import scipy.stats as stats

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
Y = df["y"].to_numpy()
X = np.column_stack((np.ones(len(df)), df["x"].to_numpy()))
Z = np.column_stack((np.ones(len(df)), df["z"].to_numpy()))
# first stage
pi = np.linalg.solve(Z.T @ Z, Z.T @ X[:, 1])
vhat = X[:, 1] - Z @ pi
# OLS of y on x + vhat
Xaug = np.column_stack((X, vhat))
fit = sm.OLS(Y, Xaug).fit()
t_vhat = fit.tvalues[2]
f = float(t_vhat ** 2)
df1 = 1
df2 = int(fit.df_resid)
pval = 1 - float(stats.f.cdf(f, df1, df2))

result = {
    "coefficients": {"f_stat": f, "p_value": pval},
    "standard_errors": {"f_stat": 0.0, "p_value": 0.0},
}
print(json.dumps(result))
