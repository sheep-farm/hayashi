import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 200
rho = 0.6
cov = np.array([[1.0, rho], [rho, 1.0]])
data = np.random.multivariate_normal([0.0, 0.0], cov, size=n)

df = pd.DataFrame({"y": data[:, 0], "x": data[:, 1]})
outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)
