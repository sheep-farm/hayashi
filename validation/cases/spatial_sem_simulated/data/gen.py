import numpy as np
import pandas as pd
import os

np.random.seed(42)

# 7 x 7 grid => 49 observations
n_rows, n_cols = 7, 7
n = n_rows * n_cols

coords = np.array([(i, j) for i in range(n_rows) for j in range(n_cols)])

# Rook contiguity W
W = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        d = np.abs(coords[i] - coords[j]).sum()
        if d == 1:
            W[i, j] = 1.0
            W[j, i] = 1.0

# Row-standardize
row_sums = W.sum(axis=1)
for i in range(n):
    if row_sums[i] > 0:
        W[i, :] = W[i, :] / row_sums[i]

x = np.random.normal(0, 1, n)
β = 0.5
λ = 0.3
e = np.random.normal(0, 1, n)

# u = (I - lambda W)^{-1} e
A = np.eye(n) - λ * W
u = np.linalg.solve(A, e)
y = x * β + u

df = pd.DataFrame({"y": y, "x": x})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

w_csv = os.path.join(os.path.dirname(__file__), "W.csv")
np.savetxt(w_csv, W, delimiter=",")

# Generate hayashi script with W as inline list
rows = ["[" + ", ".join(f"{v:.6f}" for v in row) + "]" for row in W]
w_str = "[\n  " + ",\n  ".join(rows) + "\n]"

script = f'''load "validation/cases/spatial_sem_simulated/data/data.csv" as df
let W = {w_str}
let m = spatial_sem(y ~ x, df, w=W)
print(m)
'''
with open(os.path.join(os.path.dirname(__file__), "..", "hayashi", "run.hay"), "w") as f:
    f.write(script)
