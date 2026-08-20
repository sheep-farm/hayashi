import numpy as np
import pandas as pd
import os

np.random.seed(123)
N_groups = 500
T = 4
n = N_groups * T

group = np.repeat(np.arange(N_groups), T)
x = np.random.normal(0, 1, n)
alpha = np.repeat(np.random.normal(0, 0.5, N_groups), T)
u = np.random.logistic(0, 1, n)
beta = 0.8
latent = beta * x + alpha + u
y = (latent > 0).astype(int)

df = pd.DataFrame({"group": group, "y": y, "x": x})
# Drop groups without variation (clogit conditions them out anyway)
keep = df.groupby("group")["y"].transform(lambda s: s.min() == 0 and s.max() == 1)
df = df[keep].reset_index(drop=True)

outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)
