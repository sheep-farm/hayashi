import numpy as np
import pandas as pd
import os

np.random.seed(42)

# 7 x 7 grid
n_rows, n_cols = 7, 7
n = n_rows * n_cols

coords = np.array([(i, j) for i in range(n_rows) for j in range(n_cols)])

W = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        if np.abs(coords[i] - coords[j]).sum() == 1:
            W[i, j] = 1.0
            W[j, i] = 1.0

row_sums = W.sum(axis=1)
for i in range(n):
    if row_sums[i] > 0:
        W[i, :] /= row_sums[i]

x = np.random.normal(0, 1, n)
beta = 0.5
theta = 0.2
rho = 0.3
e = np.random.normal(0, 1, n)

# y = rho W y + X beta + W X theta + e
A = np.eye(n) - rho * W
X = x
WX = W.dot(x)
RHS = X * beta + WX * theta + e
y = np.linalg.solve(A, RHS)

df = pd.DataFrame({"id": np.arange(n), "y": y, "x": x})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)
np.savetxt(os.path.join(os.path.dirname(__file__), "W.csv"), W, delimiter=",")

# Hayashi script
rows = ["[" + ", ".join(f"{v:.6f}" for v in row) + "]" for row in W]
w_str = "[\n  " + ",\n  ".join(rows) + "\n]"

script = f'''load "validation/cases/spatial_durbin_simulated/data/data.csv" as df
let W = {w_str}
let m = spatial_durbin(y ~ x, df, w=W, id="id")
print("rho=" + str(m.fit.rho))
'''
with open(os.path.join(os.path.dirname(__file__), "..", "hayashi", "run.hay"), "w") as f:
    f.write(script)
