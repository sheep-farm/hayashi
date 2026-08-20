import numpy as np
import pandas as pd
import os

np.random.seed(42)
T = 250

# Stable VAR(1): y1_t = 0.5*y1_{t-1} + 0.1*y2_{t-1} + u1_t
#                y2_t = 0.2*y1_{t-1} + 0.4*y2_{t-1} + u2_t
A1 = np.array([[0.5, 0.1],
               [0.2, 0.4]])
Sigma = np.array([[1.0, 0.3],
                  [0.3, 1.0]])

u = np.random.multivariate_normal([0.0, 0.0], Sigma, size=T)
y = np.zeros((T, 2))
for t in range(1, T):
    y[t] = A1 @ y[t - 1] + u[t]

df = pd.DataFrame({
    "y1": y[:, 0],
    "y2": y[:, 1],
})
outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)
