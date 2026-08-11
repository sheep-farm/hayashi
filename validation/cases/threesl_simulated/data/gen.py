import numpy as np
import pandas as pd
import os

np.random.seed(1011)
n = 500

x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
u = np.random.multivariate_normal([0, 0], [[1.0, 0.5], [0.5, 1.0]], size=n)

# Structural parameters
a1 = 0.4
a2 = 0.3
b1 = -0.2
b2 = 0.5

# Solve reduced form for y1, y2
# y1 = a1*x1 + a2*y2 + u1
# y2 = b1*x2 + b2*y1 + u2
# => y1 = a1*x1 + a2*(b1*x2 + b2*y1 + u2) + u1
# => y1*(1 - a2*b2) = a1*x1 + a2*b1*x2 + a2*u2 + u1
# similarly y2
den = 1 - a2 * b2
y1 = (a1 * x1 + a2 * b1 * x2 + a2 * u[:, 1] + u[:, 0]) / den
y2 = (b1 * x2 + b2 * a1 * x1 + b2 * u[:, 0] + u[:, 1]) / den

df = pd.DataFrame({"y1": y1, "y2": y2, "x1": x1, "x2": x2})
outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)
