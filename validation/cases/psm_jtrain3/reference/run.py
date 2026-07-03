# Reference implementation in Python for PSM on Wooldridge jtrain3.
#
# 1:1 nearest-neighbor propensity score matching with caliper and bootstrap SE.

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "jtrain3.csv"

# Load or download Wooldridge jtrain3 via the Python wooldridge package.
if not CSV_PATH.exists():
    from wooldridge import data as wd_data

    df = wd_data("jtrain3")
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Variable definitions.
outcome = "re78"
treatment = "train"
covariates = [
    "age",
    "educ",
    "black",
    "hisp",
    "married",
    "unem74",
    "unem75",
    "re74",
    "re75",
]

# Build the propensity-score model.
formula = f"{treatment} ~ " + " + ".join(covariates)
ps_model = smf.logit(formula, data=df).fit(disp=0)
ps = ps_model.predict(df)

# Caliper: absolute 0.2 on the propensity-score scale (Hayashi default).
caliper = 0.2

treated_idx = np.where(df[treatment] == 1)[0]
control_idx = np.where(df[treatment] == 0)[0]

ps_treated = ps.iloc[treated_idx].values.reshape(-1, 1)
ps_control = ps.iloc[control_idx].values.reshape(-1, 1)

# 1:1 nearest-neighbor matching without replacement (sequential greedy).
def match_without_replacement(ps_t, ps_c, treated_idx, control_idx, caliper):
    used = set()
    matched_t = []
    matched_c = []
    for i, ti in enumerate(treated_idx):
        ps_ti = ps_t[i]
        best_ci = None
        best_dist = np.inf
        for j, ci in enumerate(control_idx):
            if ci in used:
                continue
            dist = abs(ps_ti - ps_c[j])
            if dist <= caliper and dist < best_dist:
                best_dist = dist
                best_ci = ci
        if best_ci is not None:
            matched_t.append(ti)
            matched_c.append(best_ci)
            used.add(best_ci)
    return np.array(matched_t), np.array(matched_c)

matched_treated, matched_control = match_without_replacement(
    ps_treated.ravel(), ps_control.ravel(), treated_idx, control_idx, caliper
)

y_treated = df[outcome].iloc[matched_treated].values
y_control = df[outcome].iloc[matched_control].values
att = float(np.mean(y_treated - y_control))

# Bootstrap SE (200 reps, same seed used by Hayashi for the case).
random.seed(42)
np.random.seed(42)
N = len(df)
B = 200
boot_atts = []
for _ in range(B):
    boot_idx = np.random.choice(N, size=N, replace=True)
    boot_df = df.iloc[boot_idx].reset_index(drop=True)
    boot_ps = ps_model.predict(boot_df)
    boot_treated = np.where(boot_df[treatment] == 1)[0]
    boot_control = np.where(boot_df[treatment] == 0)[0]
    if len(boot_treated) == 0 or len(boot_control) == 0:
        continue
    boot_ps_t = boot_ps.iloc[boot_treated].values
    boot_ps_c = boot_ps.iloc[boot_control].values
    boot_mt, boot_mc = match_without_replacement(
        boot_ps_t, boot_ps_c, boot_treated, boot_control, caliper
    )
    if len(boot_mt) == 0:
        continue
    boot_yt = boot_df[outcome].iloc[boot_mt].values
    boot_yc = boot_df[outcome].iloc[boot_mc].values
    boot_atts.append(float(np.mean(boot_yt - boot_yc)))

se = float(np.std(boot_atts, ddof=1)) if len(boot_atts) > 1 else 0.0

result = {
    "coefficients": {"ATT": att},
    "standard_errors": {"ATT": se},
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))
