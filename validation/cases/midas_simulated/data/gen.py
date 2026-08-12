import numpy as np
import pandas as pd
import os

np.random.seed(222)

T = 100  # low-frequency periods
freq = 3
n_high = T * freq
n_lags = 12

x = np.random.normal(0, 1, n_high)

# Almon polynomial weights (degree 2): gamma = [0.2, -0.3]
gamma = np.array([0.2, -0.3])
k = np.arange(n_lags)
Z = np.column_stack([np.ones(n_lags), k, k**2])
raw = np.exp(Z[:, : len(gamma) + 1].dot(np.array([0.0, gamma[0], gamma[1]])))
weights = raw / raw.sum()

y = []
for t in range(T):
    base = t * freq + (freq - 1)
    val = 0.0
    for lag in range(n_lags):
        if base >= lag:
            val += weights[lag] * x[base - lag]
    y.append(1.0 + 2.0 * val + np.random.normal(0, 0.1))

ylow = pd.DataFrame({"y": y})
xhigh = pd.DataFrame({"x": x})

ylow.to_csv(os.path.join(os.path.dirname(__file__), "ylow.csv"), index=False)
xhigh.to_csv(os.path.join(os.path.dirname(__file__), "xhigh.csv"), index=False)
