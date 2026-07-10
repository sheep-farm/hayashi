# Reference implementation in Python/statsmodels for the deterministic
# Fama-MacBeth validation case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "fmb_panel.csv"


def build_dataset() -> pd.DataFrame:
    rows = []
    for period in range(1, 9):
        for firm in range(1, 13):
            beta = (
                0.55
                + 0.08 * firm
                + 0.012 * period
                + 0.015 * ((firm + 2 * period) % 4)
            )
            size = (
                7.0
                + 0.35 * ((firm * firm) % 13)
                + 0.05 * firm
                + 0.10 * period
                + 0.02 * ((firm + period) % 5)
            )
            eps = 0.006 * (((firm * 3 + period * 2) % 7) - 3)
            alpha = 0.015 + 0.0015 * period
            beta_slope = 0.040 + 0.0025 * ((period % 4) - 1.5)
            size_slope = 0.0045 + 0.0004 * ((period % 3) - 1)
            ret = alpha + beta_slope * beta + size_slope * size + eps
            rows.append(
                {
                    "ret": ret,
                    "beta": beta,
                    "size": size,
                    "firm": firm,
                    "period": period,
                }
            )
    return pd.DataFrame(rows)


df = build_dataset()
df.to_csv(CSV_PATH, index=False)

period_coefs = []
for _, group in df.groupby("period", sort=True):
    model = smf.ols("ret ~ beta + size", data=group).fit()
    period_coefs.append(model.params)

coef_df = pd.DataFrame(period_coefs)
coefs = {name: float(value) for name, value in coef_df.mean().items()}
std_errors = {
    name: float(value)
    for name, value in (coef_df.std(ddof=1) / (len(coef_df) ** 0.5)).items()
}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))
