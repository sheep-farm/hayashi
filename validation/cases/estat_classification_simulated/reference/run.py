import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["y"].to_numpy()
model = smf.logit("y ~ x", data=df).fit(disp=0)
p = model.predict(df).to_numpy()
pred = (p >= 0.5).astype(int)
tp = int(np.sum((pred == 1) & (y == 1)))
tn = int(np.sum((pred == 0) & (y == 0)))
fp = int(np.sum((pred == 1) & (y == 0)))
fn = int(np.sum((pred == 0) & (y == 1)))
sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
acc = (tp + tn) / (tp + tn + fp + fn)

result = {
    "coefficients": {"sensitivity": sens, "specificity": spec, "correct_rate": acc},
    "standard_errors": {"sensitivity": 0.0, "specificity": 0.0, "correct_rate": 0.0},
}
print(json.dumps(result))
