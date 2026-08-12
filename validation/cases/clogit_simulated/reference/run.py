#!/usr/bin/env python3
"""Python reference for the conditional logit simulated case.

Uses statsmodels ConditionalLogit to reproduce the R survival::clogit result.
"""

import json
from pathlib import Path

import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = CASE_DIR / "data" / "data.csv"

df = pd.read_csv(DATA_PATH)

model = ConditionalLogit(endog=df["y"], exog=df[["x"]], groups=df["group"])
result = model.fit(disp=False)

out = {
    "coefficients": {"x": float(result.params.iloc[0])},
    "standard_errors": {"x": float(result.bse.iloc[0])},
}

print(json.dumps(out, indent=2))
