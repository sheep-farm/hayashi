# Reference implementation in Python for robust F-test.

import numpy as np
import pandas as pd
import json
import statsmodels.api as sm
from scipy.stats import f
import os

# Load the CSV written by the R reference so both references agree.
data_dir = "validation/cases/ftest_robust_wooldridge_wage1/data"
wage1 = pd.read_csv(f"{data_dir}/wage1.csv")

# Estimate OLS model.
X = sm.add_constant(wage1[['educ', 'exper', 'tenure']])
y = wage1['wage']
model = sm.OLS(y, X).fit()

# Extract coefficients and standard errors for the tested variables.
coefs = model.params
ses = model.bse
test_names = ['exper', 'tenure']
idx = [i for i, name in enumerate(coefs.index) if name in test_names]
q = len(idx)
p = len(coefs)
n = len(wage1)

beta_r = coefs.iloc[idx].values
vcov_r = np.diag(ses.iloc[idx].values ** 2)

wald = float(beta_r @ np.linalg.inv(vcov_r) @ beta_r)
f_stat = wald / q
df_num = q
df_denom = n - p
p_value = 1 - f.cdf(f_stat, df_num, df_denom)

result = {
    'test_statistic': f_stat,
    'p_value': p_value,
    'degrees_of_freedom_num': df_num,
    'degrees_of_freedom_denom': df_denom
}

out_dir = "validation/cases/ftest_robust_wooldridge_wage1/reference"
os.makedirs(out_dir, exist_ok=True)

with open(f"{out_dir}/expected.json", 'w') as f:
    json.dump(result, f, indent=2)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
print(json.dumps(result))
