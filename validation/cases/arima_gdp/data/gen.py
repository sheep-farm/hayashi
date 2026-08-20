# Data generation for the ARIMA GDP case.

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "macrodata.csv"

if not CSV_PATH.exists():
    macro = sm.datasets.macrodata.load_pandas().data
    macro = macro[["year", "quarter", "realgdp"]]
    macro = macro.rename(columns={"realgdp": "gdp"})
    macro.to_csv(CSV_PATH, index=False)
