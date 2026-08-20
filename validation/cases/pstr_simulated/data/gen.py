import numpy as np
import pandas as pd
import os

np.random.seed(123)

N = 50
T = 10
n = N * T
ids = np.repeat(np.arange(N), T)
q = np.tile(np.linspace(-2, 2, T), N)
x = np.random.normal(0, 1, n)

# True parameters
beta0 = 1.0
beta1 = 2.0
gamma = 5.0
c = 0.5

g = 1.0 / (1.0 + np.exp(-gamma * (q - c)))
y = beta0 * x + beta1 * x * g + np.random.normal(0, 0.1, n)

df = pd.DataFrame({"y": y, "x": x, "q": q, "id": ids})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)
