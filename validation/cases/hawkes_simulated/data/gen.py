import json
import numpy as np
import os
import pandas as pd

np.random.seed(42)

T = 200.0
mu_true = 0.5
alpha_true = 0.3
beta_true = 2.0

t = 0.0
events = []
while t < T:
    lambda_max = mu_true
    if events:
        lambda_max += alpha_true * sum(
            np.exp(-beta_true * (t - ti)) for ti in events
        )
    s = np.random.exponential(1.0 / lambda_max)
    t += s
    if t >= T:
        break
    if events:
        intensity = mu_true + alpha_true * sum(
            np.exp(-beta_true * (t - ti)) for ti in events
        )
    else:
        intensity = mu_true
    if np.random.rand() < intensity / lambda_max:
        events.append(t)

outdir = os.path.dirname(__file__)
df = pd.DataFrame({"time": events})
df.to_csv(os.path.join(outdir, "data.csv"), index=False)
with open(os.path.join(outdir, "T.txt"), "w") as f:
    f.write(str(T))

true = {
    "coefficients": {
        "mu": mu_true,
        "alpha": alpha_true,
        "beta": beta_true,
        "branching_ratio": alpha_true / beta_true,
    },
    "standard_errors": {
        "mu": float("nan"),
        "alpha": float("nan"),
        "beta": float("nan"),
        "branching_ratio": float("nan"),
    },
}
with open(os.path.join(outdir, "true.json"), "w") as f:
    json.dump(true, f, indent=2)
