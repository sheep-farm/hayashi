import json
import numpy as np
import pandas as pd

CASE_DIR = "validation/cases/kde_simulated"
DATA_DIR = f"{CASE_DIR}/data"

df = pd.read_csv(f"{DATA_DIR}/data.csv")
x = df["x"].values
n = len(x)
h = 0.5

# build a 512-point support grid matching R density() default range
x_min, x_max = x.min() - 3.0 * h, x.max() + 3.0 * h
support = np.linspace(x_min, x_max, 512)

densities = np.zeros_like(support)
for x_i in x:
    densities += np.exp(-0.5 * ((support - x_i) / h) ** 2)
densities /= n * h * np.sqrt(2.0 * np.pi)

peak_idx = int(np.argmax(densities))

result = {
    "coefficients": {
        "bandwidth": h,
        "peak_density": float(densities[peak_idx]),
        "peak_x": float(support[peak_idx]),
    },
    "standard_errors": {
        "bandwidth": 0.0,
        "peak_density": 0.0,
        "peak_x": 0.0,
    },
}

print(json.dumps(result))
