# Reference implementation in Python for the book GARCH(1,1) case.

import json
from pathlib import Path

import pandas as pd
from arch import arch_model

VALIDATION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = VALIDATION_DIR / "data"
REF_DIR = VALIDATION_DIR / "reference"

df = pd.read_csv(DATA_DIR / "garch.csv")

model = arch_model(df["e"], vol="Garch", p=1, q=1, dist="Normal").fit(disp="off")

params = model.params
std_err = model.std_err

result = {
    "coefficients": {
        "mu": float(params["mu"]),
        "omega": float(params["omega"]),
        "alpha[1]": float(params["alpha[1]"]),
        "beta[1]": float(params["beta[1]"]),
    },
    "standard_errors": {
        "mu": float(std_err["mu"]),
        "omega": float(std_err["omega"]),
        "alpha[1]": float(std_err["alpha[1]"]),
        "beta[1]": float(std_err["beta[1]"]),
    },
}

REF_DIR.mkdir(parents=True, exist_ok=True)
out = json.dumps(result)
(REF_DIR / "expected.json").write_text(out)
print(out)
