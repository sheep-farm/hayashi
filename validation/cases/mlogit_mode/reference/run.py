# Reference implementation in Python for the multinomial logit mode-choice case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "TravelMode.csv"
CSV_PATH = DATA_DIR / "mode.csv"

if not RAW_CSV.exists():
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/AER/TravelMode.csv"
    pd.read_csv(url).to_csv(RAW_CSV, index=False)

raw = pd.read_csv(RAW_CSV)

# Collapse the long-format travel-mode data to one observation per individual.
# The alternative-specific attributes (wait, vcost, travel) are averaged per
# individual so that they become individual-specific covariates suitable for a
# standard multinomial logit.
avg = raw.groupby("individual")[["wait", "vcost", "travel"]].mean().reset_index()
chosen = raw[raw["choice"] == "yes"][["individual", "mode", "income"]].copy()
chosen = chosen.merge(avg, on="individual")

# Encode mode as numeric: air=1, train=2, bus=3, car=4 (base category).
mode_map = {"air": 1, "train": 2, "bus": 3, "car": 4}
chosen["mode"] = chosen["mode"].map(mode_map)
chosen = chosen.drop(columns=["individual"])

# Standardise covariates to improve numerical stability of the Newton-Raphson solver.
for col in ["income", "wait", "vcost", "travel"]:
    chosen[col] = (chosen[col] - chosen[col].mean()) / chosen[col].std()

chosen.to_csv(CSV_PATH, index=False)

df = pd.read_csv(CSV_PATH)

# Make the outcome a categorical with car=4 as the first level. statsmodels
# MNLogit uses the first category as the reference, matching Hayashi's use of
# the highest numeric category as the base for the original encoding.
df["mode"] = pd.Categorical(df["mode"], categories=[4, 1, 2, 3])
X = sm.add_constant(df[["income", "wait", "vcost", "travel"]], has_constant="add")
y = df["mode"]
model = sm.MNLogit(y, X).fit(disp=0)

# The non-base categories are the levels after the reference: 1, 2, 3.
non_base_cats = [1, 2, 3]

coefs: dict[str, float] = {}
std_errors: dict[str, float] = {}
for i, var in enumerate(model.params.index):
    var_key = "Intercept" if var == "const" else var
    for j, cat in enumerate(non_base_cats):
        key = f"{cat}:{var_key}"
        coefs[key] = float(model.params.iloc[i, j])
        std_errors[key] = float(model.bse.iloc[i, j])

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))
