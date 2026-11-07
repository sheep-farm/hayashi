"""Native arch FMOLS with the explicit Greeners finite-sample contract."""

import hashlib
import json
import platform
from pathlib import Path

import arch
import numpy as np
import pandas as pd
import scipy
from arch.unitroot.cointegration import FullyModifiedOLS


FIXTURES = ("iid", "endogeneity", "serial_correlation", "holdout")
DATA = Path(__file__).resolve().parents[1] / "data/data.csv"


def reference_results(data: pd.DataFrame, fixtures=FIXTURES) -> dict:
    """Validate ordered fixture rows, then fit native arch independently."""
    if (list(data.columns) != ["fixture", "row", "y", "x"] or len(data) != 1200
            or set(data.fixture) != set(FIXTURES)
            or not np.isfinite(data[["row", "y", "x"]].to_numpy(dtype=float)).all()
            or any(not np.array_equal(data.loc[data.fixture == name, "row"], np.arange(300))
                   for name in FIXTURES)):
        raise ValueError("Invalid FMOLS fixture data: require four ordered, finite 300-row designs")
    if not fixtures or len(set(fixtures)) != len(fixtures) or not set(fixtures) <= set(FIXTURES):
        raise ValueError("Invalid FMOLS fixture selection")
    if arch.__version__ != "8.0.0":
        raise ValueError("FMOLS reference requires arch8.0.0")
    result = {"coefficients": {}, "standard_errors": {}, "diagnostics": {
        "versions": {"Python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "arch": arch.__version__},
        "fixtures": {}}}
    for name in fixtures:
        df = data.loc[data.fixture == name, ["y", "x"]]
        bandwidth = max(1, int(4*(len(df)/100)**(2/9)))
        fit = FullyModifiedOLS(df.y, df[["x"]], trend="c").fit(
            kernel="bartlett", bandwidth=bandwidth, force_int=True, diff=False, df_adjust=False)
        # arch reports residuals on all N rows; its FM fit uses rows 2..N.
        n = len(df)-1
        if len(fit.resid) != len(df) or fit.bandwidth != bandwidth:
            raise ValueError("Native FMOLS sample or realised bandwidth disagrees with its contract")
        params = fit.params[["const", "x"]].to_numpy()
        se = fit.std_errors[["const", "x"]].to_numpy()
        if not np.isfinite(params).all() or not np.isfinite(se).all() or (se <= 0).any():
            raise ValueError("Native FMOLS inference is non-finite or non-positive")
        prefix = "" if name == "iid" else name + "_"
        for field, value in zip(("Intercept", "x", "n_obs", "bandwidth"),
                                (*params, n, int(fit.bandwidth))):
            result["coefficients"][prefix+field] = float(value)
        for field, value in zip(("Intercept", "x"), se):
            result["standard_errors"][prefix+field] = float(value)
        result["diagnostics"]["fixtures"][name] = {
            "N": len(df), "n": n, "returned_residual_rows": len(fit.resid), "requested_bandwidth": bandwidth,
            "realised_bandwidth": float(fit.bandwidth), "bias_multiplier": len(df)-1,
            "kernel": "bartlett", "trend": "c", "diff": False, "df_adjust": False}
    return result


def main():
    result = reference_results(pd.read_csv(DATA, float_precision="round_trip"))
    result["diagnostics"]["input_sha256"] = hashlib.sha256(DATA.read_bytes()).hexdigest()
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
