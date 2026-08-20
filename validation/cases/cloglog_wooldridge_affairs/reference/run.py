# Reference implementation in Python for complementary log-log on Wooldridge affairs.

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from wooldridge import data

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "affairs.csv"

df = data("affairs")
df.to_csv(CSV_PATH, index=False)

model = smf.glm(
    "affair ~ age + yrsmarr + kids + educ + relig + ratemarr",
    data=df,
    family=sm.families.Binomial(link=sm.families.links.CLogLog()),
).fit(disp=0)

params = model.params
bse = model.bse

result = {
    "coefficients": {name: float(params[name]) for name in params.index},
    "standard_errors": {name: float(bse[name]) for name in bse.index},
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))
