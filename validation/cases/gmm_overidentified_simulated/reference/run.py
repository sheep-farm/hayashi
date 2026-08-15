"""Independent Python reference for two-step heteroskedastic linear IV-GMM."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from linearmodels.iv import IVGMM


def main() -> None:
    case_dir = Path(__file__).resolve().parent.parent
    data = pd.read_csv(case_dir / "data" / "data.csv")
    fit = IVGMM(
        dependent=data["y"],
        exog=pd.DataFrame({"const": 1.0, "x": data["x"]}),
        endog=data[["endog"]],
        instruments=data[["z1", "z2"]],
        weight_type="robust",
    ).fit(iter_limit=2, cov_type="robust", debiased=False)

    result = {
        "coef_const": float(fit.params["const"]),
        "coef_x": float(fit.params["x"]),
        "coef_endog": float(fit.params["endog"]),
        "se_const": float(fit.std_errors["const"]),
        "se_x": float(fit.std_errors["x"]),
        "se_endog": float(fit.std_errors["endog"]),
        "j_stat": float(fit.j_stat.stat),
        "df_overid": float(fit.j_stat.df),
        "n_obs": float(fit.nobs),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
