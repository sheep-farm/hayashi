from pathlib import Path
import numpy as np
import pandas as pd
import os
np.random.seed(42)
n = 120
x = np.arange(1, n + 1) * 1.0001
y = np.zeros(n)
y[:40] = 2.0 + np.random.normal(0, 0.05, 40)
y[40:80] = 5.0 + np.random.normal(0, 0.05, 40)
y[80:] = 8.0 + np.random.normal(0, 0.05, 40)
df = pd.DataFrame({'x': x, 'y': y})
out = Path(__file__).resolve().parent / 'data.csv'
df.to_csv(out, index=False)
