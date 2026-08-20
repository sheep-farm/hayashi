import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import newton

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

# Group-level observations
Y = df.groupby("group")["y"].apply(np.array).values
X = df.groupby("group")["x"].apply(np.array).values


def ll_and_grad_hess(b):
    ll = 0.0
    grad = 0.0
    hess = 0.0
    for yg, xg in zip(Y, X):
        S = yg.sum()
        if S == 0:
            continue
        xb = xg * b
        # stable log-sum-exp
        xb_max = xb.max()
        exb = np.exp(xb - xb_max)
        sum_exb = exb.sum()
        ll += np.sum(yg * xb) - S * (xb_max + np.log(sum_exb))
        w = exb / sum_exb
        grad += np.sum((yg - S * w) * xg)
        mean_x = np.sum(w * xg)
        hess += -S * (np.sum(w * xg * xg) - mean_x * mean_x)
    return ll, grad, hess


# Find root of the score
b = newton(
    lambda beta: ll_and_grad_hess(beta)[1],
    0.5,
    fprime=lambda beta: ll_and_grad_hess(beta)[2],
)
_, _, hess = ll_and_grad_hess(b)
se = float(1.0 / np.sqrt(-hess))

result = {
    "coefficients": {"x": float(b)},
    "standard_errors": {"x": se},
}
print(json.dumps(result, indent=2))
