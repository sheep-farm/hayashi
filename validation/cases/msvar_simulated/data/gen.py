import numpy as np
import pandas as pd
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 600
regime = np.zeros(n, dtype=int)
P = np.array([[0.9, 0.1], [0.3, 0.7]])
regime[0] = 0
for t in range(1, n):
    regime[t] = np.random.choice(2, p=P[regime[t - 1]])

mu0 = np.array([0.5, -0.2])
mu1 = np.array([2.0, 1.0])
A = np.array([[0.3, 0.1], [-0.1, 0.4]])
Sigma0 = np.array([[1.0, 0.2], [0.2, 1.0]])
Sigma1 = np.array([[2.0, 0.5], [0.5, 2.0]])

y = np.zeros((n, 2))
e = np.random.multivariate_normal([0.0, 0.0], Sigma0, size=n)
for t in range(1, n):
    mu = mu0 if regime[t] == 0 else mu1
    Sig = Sigma0 if regime[t] == 0 else Sigma1
    e[t] = np.random.multivariate_normal([0.0, 0.0], Sig)
    y[t] = mu + A @ y[t - 1] + e[t]

df = pd.DataFrame({"y1": y[:, 0], "y2": y[:, 1]})
df.to_csv(DATA_DIR / "data.csv", index=False)
