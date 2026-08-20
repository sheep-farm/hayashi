import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 400
# Generate x as random walk
x = np.cumsum(np.random.normal(0, 1, n))

# Asymmetric cumulative effects: positive shocks have larger effect
dx = np.diff(x)
x_pos = np.zeros(n)
x_neg = np.zeros(n)
cum_pos = 0.0
cum_neg = 0.0
x_pos[0] = x[0]
x_neg[0] = x[0]
for i in range(1, n):
    if dx[i-1] > 0:
        cum_pos += dx[i-1]
    else:
        cum_neg += dx[i-1]
    x_pos[i] = x[0] + cum_pos
    x_neg[i] = x[0] + cum_neg

# Long-run: y = 0.8 x_pos - 0.4 x_neg + noise
# short-run ECM dynamics
y = np.zeros(n)
for t in range(1, n):
    dy = 0.2 * (0.8 * x_pos[t-1] - 0.4 * x_neg[t-1] - y[t-1])
    dy += 0.2 * (y[t-1] - y[t-2]) if t > 1 else 0
    dy += 0.3 * (x_pos[t-1] - x_pos[t-2]) - 0.1 * (x_neg[t-1] - x_neg[t-2])
    dy += np.random.normal(0, 1)
    y[t] = y[t-1] + dy

# Skip first obs to match Greeners NARDL (it uses x[0] as starting point)
df = pd.DataFrame({"y": y[1:], "x": x[1:]})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)
