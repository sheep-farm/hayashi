"""Generate a deterministic overidentified, heteroskedastic IV-GMM data set."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(20260815)
    n = 1_000
    x = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    v = rng.normal(size=n)
    e = rng.normal(size=n)

    # endog and the outcome error share v, while z1 and z2 remain excluded.
    endog = 0.4 * x + 0.8 * z1 + 0.6 * z2 + v
    outcome_error = 0.65 * v + (0.5 + 0.3 * np.abs(x)) * e
    y = 1.0 + 0.5 * x + 1.25 * endog + outcome_error

    data = pd.DataFrame({"y": y, "x": x, "endog": endog, "z1": z1, "z2": z2})
    data.to_csv(Path(__file__).with_name("data.csv"), index=False)


if __name__ == "__main__":
    main()
