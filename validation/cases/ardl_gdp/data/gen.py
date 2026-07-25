# Data generation for the ARDL GDP case.
# Loads statsmodels macro data and writes the CSV for references to consume.

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "macrodata.csv"

if not CSV_PATH.exists():
    macro = sm.datasets.macrodata.load_pandas().data
    macro = macro[["year", "quarter", "realgdp", "realcons"]]
    macro = macro.rename(columns={"realgdp": "gdp", "realcons": "cons"})
    macro.to_csv(CSV_PATH, index=False)
