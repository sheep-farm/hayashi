import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv").sort_values("size").reset_index(drop=True)
n = 5
n_valid = len(df)
per = n_valid // n
means = []
ses = []
labels = ["Low", "P2", "P3", "P4", "High"]
for p in range(n):
    start = p * per
    end = n_valid if p == n - 1 else (p + 1) * per
    rets = df["ret"].iloc[start:end].values
    means.append(float(rets.mean()))
    ses.append(float(rets.std(ddof=1) / np.sqrt(len(rets))))
hl_mean = means[-1] - means[0]
result = {
  "coefficients": {labels[i]: means[i] for i in range(n)},
  "standard_errors": {labels[i]: ses[i] for i in range(n)}
}
result["coefficients"]["hl_mean"] = hl_mean
result["standard_errors"]["hl_mean"] = float("nan")
print(json.dumps(result))
