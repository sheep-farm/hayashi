# fmb_simulated

This case validates Hayashi's `fmb()` command on a deterministic simulated
asset-pricing panel.

## Dataset

- **Name:** `simulated_fmb_panel`
- **Source:** deterministic DGP implemented in the R and Python reference scripts
- **Licence:** generated test data
- **Size:** 96 rows x 5 columns

## Analysis

The simulated panel has 12 firms observed over 8 periods. Returns are generated
from period-varying exposures to `beta` and `size`, plus deterministic
idiosyncratic variation. The Hayashi script coerces the integer `period`
column to `fperiod` for the Greeners-backed Fama-MacBeth implementation, then
estimates:

```hayashi
fmb(ret ~ beta + size, df, time=fperiod)
```

## Reference Implementation

- **R:** base `lm()` by period, coefficient averages, and classic
  Fama-MacBeth standard errors.
- **Python:** `statsmodels.formula.api.ols()` by period, coefficient averages,
  and classic Fama-MacBeth standard errors.

## Compared Quantities

- coefficients
- standard errors

## Tolerances and Rationale

| Quantity | Tolerance | Rationale |
|---|---:|---|
| coefficients | `1e-4` | Hayashi prints coefficient tables to 4 decimal places. |
| standard_errors | `1e-4` | Hayashi prints FM-SE values to 4 decimal places. |
