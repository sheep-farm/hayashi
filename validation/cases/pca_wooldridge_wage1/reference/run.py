"""Python reference for standardised PCA on Wooldridge wage1."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from wooldridge import data
except ImportError:
    data = None

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "wage1.csv"
VARIABLES = ["educ", "exper", "tenure", "wage"]

if not CSV_PATH.exists():
    if data is None:
        raise RuntimeError("Python wooldridge package is required to create wage1.csv")
    data("wage1").to_csv(CSV_PATH, index=False)

df = pd.read_csv(CSV_PATH)[VARIABLES].dropna()
x = df.to_numpy(dtype=float)
z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
corr = z.T @ z / (len(z) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(corr)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
ratios = eigenvalues / eigenvalues.sum()
loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])

result = {
    "explained_variance": {
        f"PC{i + 1}": float(eigenvalues[i]) for i in range(2)
    },
    "explained_variance_ratio": {
        f"PC{i + 1}": float(ratios[i]) for i in range(2)
    },
    "absolute_loadings": {
        f"{variable}:PC{component + 1}": float(abs(loadings[row, component]))
        for row, variable in enumerate(VARIABLES)
        for component in range(2)
    },
}

with open(CASE_DIR / "reference" / "expected.json", "w") as handle:
    json.dump(result, handle, indent=2)

print(json.dumps(result))
