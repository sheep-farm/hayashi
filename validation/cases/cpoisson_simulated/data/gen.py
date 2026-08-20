import numpy as np
import pandas as pd
import os

np.random.seed(456)
N_groups = 400
T = 4
n = N_groups * T

group = np.repeat(np.arange(N_groups), T)
x = np.random.normal(0, 1, n)
alpha = np.repeat(np.random.normal(0.5, 0.5, N_groups), T)
beta = 0.6
lambda_ = np.exp(alpha + beta * x)
y = np.random.poisson(lambda_, size=n)

df = pd.DataFrame({"group": group, "y": y, "x": x})
outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)
