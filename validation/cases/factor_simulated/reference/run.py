import json
import numpy as np
import pandas as pd

CASE_DIR = "validation/cases/factor_simulated"
DATA_DIR = f"{CASE_DIR}/data"

df = pd.read_csv(f"{DATA_DIR}/data.csv")
cor = df.corr().values
w = np.linalg.eigvalsh(cor)
w = np.sort(w)[::-1]

result = {
    "coefficients": {
        "eigen_1": float(w[0]),
        "eigen_2": float(w[1]),
    },
    "standard_errors": {
        "eigen_1": 0.0,
        "eigen_2": 0.0,
    },
}

print(json.dumps(result))
