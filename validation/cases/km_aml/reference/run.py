from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from statsmodels.duration.survfunc import SurvfuncRight

CSV_PATH = Path("validation/cases/km_aml/data/aml.csv")
if not CSV_PATH.exists():
    raise FileNotFoundError(
        "missing generated AML CSV: run validation/cases/km_aml/data/gen.py first"
    )

aml = pd.read_csv(CSV_PATH)
fit = SurvfuncRight(aml["time"].to_numpy(), aml["event"].to_numpy())

expected = {
    "t10": 0.7826086956521741,
    "t20": 0.6459627329192548,
    "t30": 0.44168391994478956,
    "t40": 0.27605244996549344,
    "t50": 0.08281573498964803,
    "t60": 0.08281573498964803,
    "t70": 0.08281573498964803,
}


def survival_at(time: float) -> float:
    available = [
        idx for idx, event_time in enumerate(fit.surv_times) if event_time <= time
    ]
    if not available:
        return 1.0
    return float(fit.surv_prob[available[-1]])


survival_probabilities = {
    name: survival_at(float(name.removeprefix("t"))) for name in expected
}

for name, value in survival_probabilities.items():
    if abs(value - expected[name]) > 1e-12:
        raise RuntimeError(
            f"{name} differs from fixed expected value: {value} != {expected[name]}"
        )

print(json.dumps({"survival_probabilities": survival_probabilities}))
