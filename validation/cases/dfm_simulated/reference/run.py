import json
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

case_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(case_dir / "data" / "data.csv")

# Work with standardised data so that observation-noise variances are
# interpretable as uniquenesses.
X = (df - df.mean()) / df.std(ddof=0)

mod = DynamicFactor(X, k_factors=2, factor_order=1)
res = mod.fit(disp=False)

# The estimated observation-noise covariance is returned as a (k, k, nobs) array.
# We take the final estimate and compute 1 - diagonal / total variance (1.0 because
# the data are standardised), giving the share of variance explained by the factors.
obs_cov = res.filter_results.obs_cov[:, :, -1]
sigma2 = np.diag(obs_cov)
comm = 1.0 - sigma2

result = {
    "coefficients": {name: float(comm[i]) for i, name in enumerate(df.columns)},
    "standard_errors": {name: float("nan") for name in df.columns},
}
print(json.dumps(result, indent=2))
