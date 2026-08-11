import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
fit = sm.OLS.from_formula("y ~ x", data=df).fit()
infl = fit.get_influence()
maxd = float(np.max(np.abs(infl.dffits[0])))

result = {
    "coefficients": {"max_dffits": maxd},
    "standard_errors": {"max_dffits": 0.0},
}
print(json.dumps(result))
