import contextlib
import io
import json
import numpy as np
import pandas as pd
import pymc as pm
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "data.csv")
y = df["y"].values
n = len(y)

with pm.Model() as sv:
    mu = pm.Normal("mu", mu=0.0, sigma=10.0)
    phi = pm.Uniform("phi", lower=-1.0, upper=1.0)
    sigma = pm.HalfNormal("sigma", sigma=1.0)
    h = pm.AR(
        "h",
        rho=[phi],
        sigma=sigma,
        init_dist=pm.Normal.dist(mu, sigma / np.sqrt(1 - phi**2)),
        shape=n,
    )
    obs = pm.Normal("obs", mu=0.0, sigma=pm.math.exp(h / 2), observed=y)

    # Suppress PyMC sampling messages so only JSON is emitted
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        trace = pm.sample(
            draws=100,
            tune=100,
            chains=1,
            cores=1,
            progressbar=False,
            random_seed=42,
        )

h_post = trace.posterior["h"].values
mean_ht = float(np.mean(h_post))

result = {
    "coefficients": {"mean_ht": mean_ht},
    "standard_errors": {"mean_ht": 0.0},
}

print(json.dumps(result))
