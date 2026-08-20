import numpy as np
import pandas as pd
import os

np.random.seed(42)
n_units = 20
n_periods = 10
treat_period = 6
effect = 2.0

unit = np.repeat(np.arange(n_units), n_periods)
period = np.tile(np.arange(n_periods), n_units)
u = np.repeat(np.random.normal(0, 1, n_units), n_periods)
trend = 0.5 * period
e = np.random.normal(0, 0.5, n_units * n_periods)
treated = (unit == 0).astype(int) * (period >= treat_period).astype(int)
y = 2.0 + trend + u + effect * treated + e

df = pd.DataFrame({"unit": unit, "period": period, "y": y, "treated": treated})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)
