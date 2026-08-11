import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
fit = smf.ols("x ~ z", data=df).fit()
ftest = fit.f_test("z=0")
f = float(ftest.fvalue)
pval = float(ftest.pvalue)

result = {
    "coefficients": {"first_stage_f": f, "first_stage_p": pval},
    "standard_errors": {"first_stage_f": 0.0, "first_stage_p": 0.0},
}
print(json.dumps(result))
