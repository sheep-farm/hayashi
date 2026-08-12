import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
X = df[["y1", "y2", "y3"]].values

# Standardise for PCA
Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
# SVD
cov = Xs.T @ Xs / len(Xs)
eigvals, eigvecs = np.linalg.eigh(cov)
# take largest eigenvector
order = np.argsort(eigvals)[::-1]
pc = eigvecs[:, order[0]]
factor = Xs @ pc

# align sign with y2 (true positive loading on y2)
if np.corrcoef(factor, Xs[:, 1])[0, 1] < 0:
    factor = -factor

# VAR(1) on [factor, y1]
Y = np.column_stack([factor, df["y1"].values])
Ylag = np.roll(Y, 1, axis=0)
Ylag[0, :] = 0.0  # first obs not used
Xmat = np.column_stack([np.ones(len(Y) - 1), Ylag[1:, :]])

coef = np.linalg.lstsq(Xmat, Y[1:, :], rcond=None)[0]
# coef shape (3,2): rows const, L1.f, L1.y1; cols f eq, y1 eq

result = {
    "coefficients": {
        "const_F1": float(coef[0, 0]),
        "L1.F1_F1": float(coef[1, 0]),
        "L1.y1_F1": float(coef[2, 0]),
        "const_y1": float(coef[0, 1]),
        "L1.F1_y1": float(coef[1, 1]),
        "L1.y1_y1": float(coef[2, 1]),
    },
    "standard_errors": {
        "const_F1": float("nan"),
        "L1.F1_F1": float("nan"),
        "L1.y1_F1": float("nan"),
        "const_y1": float("nan"),
        "L1.F1_y1": float("nan"),
        "L1.y1_y1": float("nan"),
    },
}
print(json.dumps(result, indent=2))
