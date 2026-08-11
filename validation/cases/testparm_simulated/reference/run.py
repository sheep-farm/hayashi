import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
full = smf.ols("y ~ x1 + x2", data=df).fit()
red = smf.ols("y ~ 1", data=df).fit()
ftest = full.compare_f_test(red)
f = float(ftest[0])
pval = float(ftest[1])

result = {
    "coefficients": {"f_stat": f, "p_value": pval},
    "standard_errors": {"f_stat": 0.0, "p_value": 0.0},
}
print(json.dumps(result))
