import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

np.random.seed(42)
df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
ir = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(df["x"].values, df["y"].values)
yf = ir.transform(df["x"].values)
steps = [float(yf[0]), float(yf[40]), float(yf[80])]
result = {
  "coefficients": {"step_1": steps[0], "step_2": steps[1], "step_3": steps[2]},
  "standard_errors": {"step_1": float("nan"), "step_2": float("nan"), "step_3": float("nan")}
}
print(json.dumps(result))
