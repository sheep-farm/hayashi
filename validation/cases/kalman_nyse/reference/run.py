#!/usr/bin/env python3
"""Python reference for the local-level Kalman filter on NYSE returns.

Equivalent to the R dlm MLE by fitting a local-level model with
statsmodels UnobservedComponents.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = CASE_DIR / "data" / "nyse.csv"

df = pd.read_csv(DATA_PATH)
y = df["return"].dropna().to_numpy()

model = UnobservedComponents(y, "local level")
result = model.fit(disp=False)

params = dict(zip(model.param_names, result.params))

out = {
    "sigma_obs": float(np.sqrt(params["sigma2.irregular"])),
    "sigma_state": float(np.sqrt(params["sigma2.level"])),
}

print(json.dumps(out, indent=2))
