import os
import numpy as np
import pandas as pd

np.random.seed(123)
n_entities = 4
n_periods = 20
n = n_entities * n_periods

W = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0],
])

fe = np.repeat([0.0, 1.0, 2.0, 3.0], n_periods)
x = np.random.normal(0.0, 1.0, n)
eps = np.random.normal(0.0, 0.5, n)

rho = 0.3
beta = 0.8

y = np.zeros(n)
I_rhoW = np.eye(n_entities) - rho * W
for t in range(n_periods):
    idx = slice(t * n_entities, (t + 1) * n_entities)
    z = beta * x[idx] + fe[idx] + eps[idx]
    y[idx] = np.linalg.solve(I_rhoW, z)

entity = np.tile(np.arange(1, n_entities + 1), n_periods)
time = np.repeat(np.arange(1, n_periods + 1), n_entities)

pd.DataFrame({"entity": entity, "time": time, "y": y, "x": x}).to_csv(
    os.path.join(os.path.dirname(__file__), "data.csv"), index=False
)
np.savetxt(os.path.join(os.path.dirname(__file__), "W.csv"), W, delimiter=",")
