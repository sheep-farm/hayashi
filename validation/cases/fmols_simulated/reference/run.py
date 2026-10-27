"""Native arch FMOLS with the explicit Greeners finite-sample contract."""

import json
from pathlib import Path

import arch
import numpy as np
import pandas as pd
from arch.unitroot.cointegration import FullyModifiedOLS


def main():
    if arch.__version__ != "8.0.0":
        raise ValueError("FMOLS reference requires arch8.0.0")
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/data.csv", float_precision="round_trip")
    if list(df.columns) != ["y", "x"] or len(df) != 300 or not np.isfinite(df.to_numpy()).all():
        raise ValueError("Expected 300 complete finite y/x observations")
    bandwidth = max(1, int(4*(len(df)/100)**(2/9)))
    fit = FullyModifiedOLS(df.y, df[["x"]], trend="c").fit(
        kernel="bartlett", bandwidth=bandwidth, force_int=True, diff=False, df_adjust=False)
    result = {
        "coefficients": {"Intercept": float(fit.params["const"]), "x": float(fit.params["x"]),
                         "n_obs": len(df)-1, "bandwidth": bandwidth},
        "standard_errors": {"Intercept": float(fit.std_errors["const"]), "x": float(fit.std_errors["x"])},
    }
    if not all(np.isfinite(v) for group in result.values() for v in group.values()):
        raise ValueError("Native FMOLS inference is non-finite")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
