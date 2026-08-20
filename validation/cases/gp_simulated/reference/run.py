import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "data.csv")
X = df[["x"]].values
y = df["y"].values

# RBF kernel plus a noise term; optimise hyperparameters.
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 10.0)) + WhiteKernel(noise_level=0.1)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
model.fit(X, y)

y_hat = model.predict(X)

mse = float(np.mean((y - y_hat) ** 2))
ss_res = float(np.sum((y - y_hat) ** 2))
ss_tot = float(np.sum((y - np.mean(y)) ** 2))
r2 = float(1.0 - ss_res / ss_tot)

result = {
    "coefficients": {"r2": r2, "mse": mse},
    "standard_errors": {"r2": 0.0, "mse": 0.0},
}

print(json.dumps(result))
