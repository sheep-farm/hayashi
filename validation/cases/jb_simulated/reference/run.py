# Reference implementation in Python for the Jarque-Bera normality test.

import json
import pandas as pd
from scipy import stats
from pathlib import Path

data_dir = Path("validation/cases/jb_simulated/data")
x = pd.read_csv(data_dir / "data.csv")["x"].to_numpy()

stat, p = stats.jarque_bera(x)

result = {"jb_stat": float(stat), "p_value": float(p)}

out_dir = Path("validation/cases/jb_simulated/reference")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
