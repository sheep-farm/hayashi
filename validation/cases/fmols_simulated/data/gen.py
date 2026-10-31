"""Fixed FMOLS reference fixtures; no estimator or reference is called here."""

from pathlib import Path

import numpy as np
import pandas as pd


def datasets() -> dict[str, pd.DataFrame]:
    """Return the original IID control, two engine designs and a fresh holdout."""
    n = 300
    rng = np.random.RandomState(42)
    x = rng.normal(0.0, 0.1, n).cumsum()
    frames = {"iid": pd.DataFrame({"y": 0.5*x + rng.normal(0.0, 0.5, n), "x": x})}

    for name, seed in (("endogeneity", 314159), ("serial_correlation", 271828)):
        rng = np.random.Generator(np.random.PCG64(seed))
        dx, e = rng.normal(size=n), rng.normal(size=n)
        x = dx.cumsum()
        u = 0.75*dx + e
        if name == "serial_correlation":
            u = np.empty(n)
            u[0] = e[0]/np.sqrt(1-0.7**2)
            for t in range(1, n):
                u[t] = 0.7*u[t-1] + e[t]
        frames[name] = pd.DataFrame({"y": 0.25 + 0.5*x + u, "x": x})

    # Fresh seed and combined dynamics, fixed before any holdout fit.
    rng = np.random.Generator(np.random.PCG64(2026103101))
    dx, e = rng.normal(size=n+300), rng.normal(size=n+300)
    x = dx.cumsum()
    u = np.empty(n+300)
    previous = 0.0
    for t in range(n+300):
        previous = 0.5*previous + 0.5*dx[t] + e[t]
        u[t] = previous
    frames["holdout"] = pd.DataFrame({"y": (0.25 + 0.5*x + u)[300:], "x": x[300:]})
    return frames


def main():
    frames = [frame.assign(fixture=name, row=np.arange(len(frame)))[["fixture", "row", "y", "x"]]
              for name, frame in datasets().items()]
    pd.concat(frames, ignore_index=True).to_csv(Path(__file__).with_name("data.csv"),
                                               index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
