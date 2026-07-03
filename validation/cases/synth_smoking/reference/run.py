# Reference implementation in Python for synthetic control.
#
# Uses a simple simulated panel (10 donors + 1 treated unit, 20 periods).
# The treated unit is unit 1; intervention starts in period 11.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "synth_smoking.csv"

np.random.seed(42)

# Build the simulated panel if the CSV is not present.
if not CSV_PATH.exists():
    n_units = 10
    n_periods = 20
    unit = np.repeat(np.arange(1, n_units + 1), n_periods)
    year = np.tile(np.arange(1, n_periods + 1), n_units)
    alpha = np.where(unit == 1, 5.0, unit * 1.0)
    d = (unit == 1) * (year >= 11)
    e = np.random.normal(size=len(unit))
    y = alpha + 0.3 * year + 3.0 * d + e
    df = pd.DataFrame({"unit": unit, "year": year, "y": y, "alpha": alpha, "d": d})
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Pre- and post-treatment periods.
T0 = 10  # periods 1..10 are pre-treatment
T1 = 10  # periods 11..20 are post-treatment

# Pivot to wide format: rows = period, columns = unit.
y_wide = df.pivot(index="year", columns="unit", values="y")

y_pre = y_wide.iloc[:T0, :]
y_post = y_wide.iloc[T0:, :]

# Treated unit is column 1; donors are columns 2..10.
y_t_pre = y_pre[1].values
Y_d_pre = y_pre.iloc[:, 1:].values  # donors in pre-treatment
Y_d_post = y_post.iloc[:, 1:].values  # donors in post-treatment

# Estimate donor weights: minimize squared pre-treatment prediction error,
# subject to weights summing to 1 and being non-negative.
n_donors = Y_d_pre.shape[1]


def objective(w):
    return np.sum((y_t_pre - Y_d_pre @ w) ** 2)


constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
bounds = [(0.0, 1.0) for _ in range(n_donors)]
result = minimize(
    objective,
    x0=np.ones(n_donors) / n_donors,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 1000},
)

w = result.x

# Synthetic control prediction in the post-treatment period.
y_sc_post = Y_d_post @ w

# ATT = average effect in post-treatment period.
att = float(np.mean(y_post[1].values - y_sc_post))

result = {"coefficients": {"ATT": att}}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))
