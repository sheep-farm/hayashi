"""Report both R/Python contracts and weights; failed robust agreement exits 1."""

import csv
import io
import subprocess
import sys
from pathlib import Path

import numpy as np

from run import COMPONENTS, check_components, fit_stl, load_counts, vector


def median_weights(residual):
    absolute = np.abs(vector(residual))
    scale = 6 * np.median(absolute)
    if scale == 0:
        return np.ones_like(absolute)
    u = absolute / scale
    return np.where(u <= 0.001, 1.0, np.where(u <= 0.999, (1 - u * u) ** 2, 0.0))


def main():
    observed = np.log(load_counts())
    result = subprocess.run(["Rscript", str(Path(__file__).with_suffix(".R"))],
                            capture_output=True, text=True, check=True)
    print(result.stderr.strip())
    reader = csv.DictReader(io.StringIO(result.stdout))
    if reader.fieldnames != ["outer", "index", "observed", *COMPONENTS, "weight"]:
        raise ValueError("Unexpected R diagnostic columns")
    rows = list(reader)
    if [(row["outer"], row["index"]) for row in rows] != [
        (str(outer), str(i)) for outer in (0, 1) for i in range(1, 145)
    ]:
        raise ValueError("Expected two ordered full R fits")
    fits = {}
    failed = False
    for outer in (0, 1):
        block = rows[outer * 144:(outer + 1) * 144]
        r = {key: vector([float(row[key]) for row in block])
             for key in ("observed", *COMPONENTS, "weight")}
        np.testing.assert_allclose(r["observed"], observed, atol=1e-12, rtol=0, equal_nan=False)
        r_components = {key: r[key] for key in COMPONENTS}
        reconstruction = check_components(r["observed"], r_components, r["observed"])
        fit, python = fit_stl(observed, outer=outer)
        fits[outer] = (r, fit)
        print(f"outer={outer} R reconstruction max_abs={reconstruction:.17g}")
        print(f"outer={outer} Python reconstruction max_abs="
              f"{check_components(fit.observed, python, observed):.17g}")
        for key in COMPONENTS:
            error = float(np.max(np.abs(r[key] - python[key])))
            print(f"outer={outer} R/Python {key} max_abs={error:.17g}")
            failed |= error > 1e-8
    r0, p0 = fits[0]
    r1, p1 = fits[1]
    r_error = float(np.max(np.abs(r1["weight"] - median_weights(r0["remainder"]))))
    p_error = float(np.max(np.abs(p1.weights - median_weights(p0.resid))))
    print(f"R initial-fit median biweight max_abs={r_error:.17g}")
    print(f"Python initial-fit median biweight max_abs={p_error:.17g}")
    failed |= p_error > 1e-10
    print("Reference agreement gate: " + ("FAIL" if failed else "PASS"))
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
