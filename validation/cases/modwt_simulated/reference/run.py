import json
import numpy as np
import pandas as pd
import pywt
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["y"].values

# MODWT / stationary wavelet with Haar, no L2 normalisation to match Greeners
coeffs = pywt.swt(y, wavelet="haar", level=3, norm=False)

# pywt returns coarsest (level 3) first; Greeners labels W_1 as finest.
result = {
    "coefficients": {
        "W_1": float(np.sum(coeffs[2][1] ** 2)),
        "W_2": float(np.sum(coeffs[1][1] ** 2)),
        "W_3": float(np.sum(coeffs[0][1] ** 2)),
    },
    "standard_errors": {
        "W_1": float("nan"),
        "W_2": float("nan"),
        "W_3": float("nan"),
    },
}

print(json.dumps(result, indent=2))
