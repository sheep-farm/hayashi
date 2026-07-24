# Reference implementation for the LP-DiD quickstart validation case.
# Uses pylpdid on the deterministic quickstart panel and writes the same CSV
# consumed by the Hayashi script.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pylpdid import LPDID

VALIDATION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = VALIDATION_DIR / "data"
REF_DIR = VALIDATION_DIR / "reference"
CSV_PATH = DATA_DIR / "panel.csv"


def make_panel(n_units: int = 200, n_periods: int = 15, seed: int = 0) -> pd.DataFrame:
    """Mirror pylpdid/examples/01_quickstart.py exactly."""
    rng = np.random.default_rng(seed)
    cohorts = rng.choice([0, 6, 9, 12], size=n_units)
    unit_fe = rng.normal(0, 1.0, n_units)
    rows = []
    for i in range(n_units):
        g = cohorts[i]
        for t in range(1, n_periods + 1):
            treated = int(g > 0 and t >= g)
            y = unit_fe[i] + 0.3 * t + 2.0 * treated + rng.normal(0, 0.5)
            rows.append((i, t, g, y))
    return pd.DataFrame(rows, columns=["id", "t", "g", "y"])


def write_panel() -> pd.DataFrame:
    df = make_panel()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


def main() -> None:
    df = write_panel()

    res = LPDID(target_estimand="vw", max_pre=5, max_post=8).fit(
        df,
        outcome="y",
        unit="id",
        time="t",
        first_treat="g",
    )

    es = res.event_study.set_index("horizon").sort_index()
    coefficients = {}
    standard_errors = {}
    for horizon, row in es.iterrows():
        key = f"h={int(horizon)}"
        coefficients[key] = float(row["estimate"])
        standard_errors[key] = float(row["se"])

    result = {
        "coefficients": coefficients,
        "standard_errors": standard_errors,
        "n_obs": int(res.n_obs),
    }

    REF_DIR.mkdir(parents=True, exist_ok=True)
    out = json.dumps(result)
    (REF_DIR / "expected.json").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
