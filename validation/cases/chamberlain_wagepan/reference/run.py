import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = CASE_DIR / "data" / "wagepan.csv"

df = pd.read_csv(DATA_PATH)
y = df["lwage"].values
X = df[["union", "married"]].values
entity = df["nr"].values
time = df["year"].values

entities = np.unique(entity)
years = np.unique(time)
T = len(years)

# Build the Chamberlain augmentation: for each regressor j and period s,
# include the value of X_{i,s} for every observation (i,t).
X_time = np.zeros((len(df), X.shape[1] * T))
for j in range(X.shape[1]):
    for s_idx, yv in enumerate(years):
        # map entity -> value of X[j] in year yv
        entity_value = {}
        for idx, (e, yr) in enumerate(zip(entity, time)):
            if yr == yv:
                entity_value[e] = X[idx, j]
        col = np.array([entity_value.get(e, 0.0) for e in entity])
        X_time[:, j * T + s_idx] = col

# Drop columns with no within-entity/time variation (some X are time-invariant
# in certain years and produce zero variance).
keep = X_time.std(axis=0) > 1e-12
X_time = X_time[:, keep]

# Unrestricted model: const + X + X_time
X_ur = np.column_stack([np.ones(len(df)), X, X_time])
beta_ur = np.linalg.lstsq(X_ur, y, rcond=None)[0]
resid_ur = y - X_ur @ beta_ur
ssr_ur = float(np.sum(resid_ur**2))

# Restricted model: const + X
X_r = np.column_stack([np.ones(len(df)), X])
beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
resid_r = y - X_r @ beta_r
ssr_r = float(np.sum(resid_r**2))

q = X_time.shape[1]
df_denom = len(df) - X_ur.shape[1]

f_stat = ((ssr_r - ssr_ur) / q) / (ssr_ur / df_denom)
p_value = 1 - stats.f.cdf(f_stat, q, df_denom)

out = {
    "coefficients": {
        "f_stat": f_stat,
        "p_value": p_value,
    },
    "standard_errors": {
        "f_stat": 0.0,
        "p_value": 0.0,
    },
}

print(json.dumps(out, indent=2))
