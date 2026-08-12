import os
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

e1 = np.random.normal(0.0, 0.5, n).cumsum()
e2 = np.random.normal(0.0, 0.2, n).cumsum()
shift = np.zeros(n)
shift[n // 2 :] = 2.0

y1 = e1
y2 = 2.0 * y1 + 5.0 + shift + e2

pd.DataFrame({"y1": y1, "y2": y2, "shift": shift}).to_csv(
    os.path.join(os.path.dirname(__file__), "data.csv"), index=False
)
