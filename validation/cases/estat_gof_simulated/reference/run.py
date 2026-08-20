import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["y"].to_numpy()
model = smf.logit("y ~ x", data=df).fit(disp=0)
p = model.predict(df).to_numpy()

n_groups = 10
df2 = pd.DataFrame({"y": y, "p": p})
df2["g"] = pd.qcut(df2["p"], n_groups, duplicates="drop")
hl = 0.0
for _, gg in df2.groupby("g"):
    o = gg["y"].sum()
    n = len(gg)
    e = gg["p"].sum()
    if e > 0 and (n - e) > 0:
        hl += (o - e) ** 2 / (e * (n - e) / n)

df_used = len(df2["g"].unique())
df_gof = max(1, df_used - 2)
pval = 1 - float(stats.chi2.cdf(hl, df_gof))

result = {
    "coefficients": {"hl_stat": float(hl), "p_value": float(pval)},
    "standard_errors": {"hl_stat": 0.0, "p_value": 0.0},
}
print(json.dumps(result))
