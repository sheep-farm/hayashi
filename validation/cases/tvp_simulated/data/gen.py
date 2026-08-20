import json
import numpy as np
import os
import pandas as pd

np.random.seed(42)

n = 200
t = np.arange(n)
beta0 = 2.0 + 0.05 * np.sin(2 * np.pi * t / n)
beta1 = 0.5 + 0.05 * np.cos(2 * np.pi * t / n)

x = np.random.normal(size=n)
y = beta0 + beta1 * x + np.random.normal(scale=0.5, size=n)

outdir = os.path.dirname(__file__)
df = pd.DataFrame({"y": y, "x": x})
df.to_csv(os.path.join(outdir, "data.csv"), index=False)

true = {
    "coefficients": {
        "Intercept": float(beta0[-1]),
        "x": float(beta1[-1]),
    },
    "standard_errors": {
        "Intercept": float("nan"),
        "x": float("nan"),
    },
}
with open(os.path.join(outdir, "true_final.json"), "w") as f:
    json.dump(true, f, indent=2)
