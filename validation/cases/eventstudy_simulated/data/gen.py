import json
import numpy as np
import os
import pandas as pd

np.random.seed(42)

n_units = 60
t_periods = 5
n = n_units * t_periods

unit = np.repeat(np.arange(n_units), t_periods)
time = np.tile(np.arange(t_periods), n_units)
treated = np.repeat((np.arange(n_units) < (n_units // 2)).astype(int), t_periods)

# Treated units have event at time = 2; controls never treat.
event_time = np.where(treated == 1, time - 2, -100)

# Outcome with a calendar trend and a post-treatment effect.
y = 1.0 + 0.10 * time + 0.30 * (event_time >= 0).astype(float) + np.random.normal(scale=0.2, size=n)

outdir = os.path.dirname(__file__)
df = pd.DataFrame({"unit": unit, "time": time, "event_time": event_time, "y": y})
df.to_csv(os.path.join(outdir, "data.csv"), index=False)
