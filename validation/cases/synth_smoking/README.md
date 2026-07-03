# Synthetic control on a simulated smoking-style panel

This validation case estimates the average post-treatment effect (ATT) using a
synthetic control on a simulated panel with 10 donor units and 1 treated unit.

## Dataset

- **Name:** `synth_smoking`
- **Source:** Simulated DGP from the Hayashi book Chapter 32.
- **Licence:** Simulated.
- **Size:** 10 units × 20 periods = 200 observations.
- **Variables:**
  - `unit`: unit identifier (1 is treated)
  - `year`: time period (1..20)
  - `y`: outcome
  - `d`: treatment indicator (unit 1, year ≥ 11)
  - `alpha`: unit-specific intercept

## Model

DGP:

```
alpha = 5 if unit == 1 else unit
y = alpha + 0.3 * year + 3.0 * d + N(0,1)
```

Hayashi call:

```
synth("y", 1, 11, df, id="unit", time="year")
```

## Reference implementation

- **Python:** `scipy.optimize.minimize` (SLSQP) with simplex constraints on
  donor weights; ATT = mean post-treatment gap.
- **R:** `optim` (L-BFGS-B) with projection onto the unit simplex.

## Compared quantities

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients (ATT) | 1e-1 | Weight optimisation may stop at slightly different points. |
