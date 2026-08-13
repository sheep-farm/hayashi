# Reference implementation in Python for the Jarque-Bera normality test.

import json
import numpy as np
from scipy import stats
from pathlib import Path

rng = np.random.default_rng(42)
n = 200
x = rng.normal(loc=5.0, scale=2.0, size=n)

data_dir = Path("validation/cases/jb_simulated/data")
data_dir.mkdir(parents=True, exist_ok=True)
np.savetxt(data_dir / "data.csv", np.column_stack([x]), delimiter=",", header="x", comments="", fmt="%.17g")

stat, p = stats.jarque_bera(x)

result = {"jb_stat": float(stat), "p_value": float(p)}

out_dir = Path("validation/cases/jb_simulated/reference")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
