import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(42)

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
X = df[["x"]].values
y = df["y"].values
d = df["d"].values

# Propensity score model
ps_model = LogisticRegression(random_state=42, max_iter=1000)
ps_model.fit(X, d)
phat = ps_model.predict_proba(X)[:, 1]

# Outcome model: E[y | d, x] with a single model, varying d
Xd1 = np.column_stack([np.ones(len(d)), X])
Xd0 = np.column_stack([np.zeros(len(d)), X])
outcome_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    random_state=42,
)
outcome_model.fit(np.column_stack([d, X]), y)
mu1 = outcome_model.predict(Xd1)
mu0 = outcome_model.predict(Xd0)

# AIPW / doubly robust pseudo-outcome
aipw = (
    mu1 - mu0
    + d * (y - mu1) / phat
    - (1 - d) * (y - mu0) / (1 - phat)
)
ate = float(np.mean(aipw))
ate_se = float(np.std(aipw, ddof=1) / np.sqrt(len(aipw)))

result = {
    "coefficients": {
        "ate": ate,
    },
    "standard_errors": {
        "ate": ate_se,
    },
}

print(json.dumps(result, indent=2))
