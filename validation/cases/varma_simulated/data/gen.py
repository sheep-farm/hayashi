import numpy as np
import pandas as pd
import os

np.random.seed(789)
T = 500
k = 2

# VAR coefficient matrix (stable)
A = np.array([[0.6, 0.2], [0.1, 0.5]])
# MA coefficient matrix
B = np.array([[0.3, -0.2], [0.0, 0.4]])
# Innovation covariance
Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

E = np.random.multivariate_normal([0, 0], Sigma, size=T)
Y = np.zeros((T, k))
Y[0] = E[0]
for t in range(1, T):
    Y[t] = A @ Y[t-1] + E[t] + B @ E[t-1]

# Add a small trend to keep mean near zero, no constant
df = pd.DataFrame({
    "t": np.arange(T),
    "y1": Y[:, 0],
    "y2": Y[:, 1],
})
outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)
