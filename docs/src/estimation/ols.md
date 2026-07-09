# OLS

## Basic Usage

```hay
use "auto.csv"
reg(price ~ mpg + weight + foreign, auto)
```

Output displays coefficients, standard errors, t-statistics, p-values, R-squared, and F-test. Collinear variables are automatically detected and displayed as `(omitted)`.

## Formula Syntax

### Transformations

Any language function can appear directly in the RHS of a formula. The
interpreter evaluates the expression element-wise before passing the data to
the estimator, so Greeners only sees flat numeric columns.

```hay
// log of a regressor
reg(price ~ log(weight) + mpg, auto)

// square root
reg(Y ~ sqrt(X), df)

// multiple transformations
reg(output ~ log(K) + log(L), df)

// quadratic term
reg(Y ~ X + X^2, df)
```

Coefficient names in the output reflect the original expression: `log(K)`,
`sqrt(X)`, etc.

### Interactions

The `:` operator creates an interaction term (element-wise product). Each side
can be any expression:

```hay
// interaction between simple variables
reg(Y ~ X1 + X2 + X1:X2, df)

// interaction of transformations — log(K)*log(L)
reg(output ~ log(K) + log(L) + log(K):log(L), df)

// interaction term only
reg(Y ~ log(K):log(L), df)
```

The `*` operator is shorthand for full expansion (`x1*x2` ≡ `x1 + x2 + x1:x2`):

```hay
reg(Y ~ X1 * X2, df)   // equivalent to Y ~ X1 + X2 + X1:X2
```

### `I(expr)` — arbitrary arithmetic expressions

`I(...)` wraps a compound arithmetic expression, shielding `+`, `-`, and `*`
from their special meaning in formula grammar:

```hay
// square of X
reg(Y ~ X + I(X^2), df)

// explicit interaction term
reg(Y ~ X1 + X2 + I(X1 * X2), df)

// linear combination as single regressor
reg(Y ~ I(X1 + X2), df)
```

> **Note:** `I(X^2)` and `X^2` are equivalent when the term appears alone at
> the top level. Use `I(...)` to avoid ambiguity when the expression contains
> `+` or `*`.

### Categorical variables: `C()`

`C(var)` treats `var` as categorical and generates dummies automatically
(reference = first level in lexicographic order):

```hay
reg(Y ~ X + C(region), df)
```

## Robust Standard Errors

```hay
reg(price ~ mpg + weight, auto, cov=robust)  // HC1 (default)
reg(price ~ mpg + weight, auto, cov=hc3)     // HC3, better for small samples
```

`cov=robust` defaults to HC1. Variants `hc0` through `hc4` are available.

## Clustered Standard Errors

```hay
// One-way clustering
reg(wage ~ educ + exper + tenure, nlsw, cluster=industry)

// Two-way clustering (Cameron, Gelbach & Miller)
reg(ret ~ mktrf + smb + hml, stocks, cluster=firm, cluster2=month)
```

Two-way cluster SE accounts for correlation within both dimensions. Requires enough clusters in each dimension (rule of thumb: 40+).

## Newey-West HAC Standard Errors

```hay
reg(inflation ~ unemployment + lag_inflation, macro, nw=4)
```

The `nw=` argument sets the lag truncation parameter. Use for time series regressions where errors are autocorrelated and heteroskedastic.

## Subsample Estimation with `if=`

```hay
reg(price ~ mpg + weight, auto, if=(foreign == 1))
reg(price ~ mpg + weight, auto, if=(mpg > 20 & weight < 3000))
```

The `if=` clause filters the estimation sample without modifying the dataset.

## Dynamic Formulas

Build formulas programmatically when the variable list is not known at write time:

```hay
let controls = ["educ", "exper", "exper2", "tenure"]
let f = "lwage ~ " + join(controls, " + ")
reg(f, nlsw)
```

## Prediction

```hay
let m = reg(price ~ mpg + weight, auto)
predict auto price_hat = m                 // fitted values
predict auto resid = m, "residuals"         // residuals
```

## Comparing Specifications with `esttab`

```hay
let m1 = reg(price ~ mpg, auto)
let m2 = reg(price ~ mpg + weight, auto)
let m3 = reg(price ~ mpg + weight + foreign, auto)
let m4 = reg(price ~ mpg + weight + foreign, auto, cov=robust)
esttab(m1, m2, m3, m4)
```

```
──────────────────────────────────────────────────────────
                  (1)        (2)        (3)        (4)
              price      price      price      price
──────────────────────────────────────────────────────────
mpg         -238.89***   -49.51      -55.93     -55.93
             (53.08)    (86.16)     (85.65)    (93.12)
weight                     1.75***    3.32***    3.32***
                          (0.64)     (0.67)     (0.79)
foreign                              3637.00*   3637.00*
                                    (683.77)   (712.45)
_cons       11253.1***  -4942.8*   -5853.7**  -5853.7**
──────────────────────────────────────────────────────────
N               74         74         74         74
R-sq          0.220      0.293      0.500      0.500
SE type       OLS        OLS        OLS       Robust
──────────────────────────────────────────────────────────
```

## Collinearity Detection

Hayashi automatically detects and drops collinear variables. If `weight_kg` is a linear transformation of `weight_lb`, the output shows:

```
note: weight_kg omitted because of collinearity
```

The omitted variable appears in the coefficient table marked `(omitted)` with no standard error.

## Complete Example

```hay
use "cps_wages.csv"
summarize(wage educ exper tenure, cps)

let m1 = reg(lwage ~ educ, cps)
let m2 = reg(lwage ~ educ + exper + exper2, cps)
let m3 = reg(lwage ~ educ + exper + exper2 + tenure, cps, cov=robust)
esttab(m1, m2, m3)

predict cps resid = m3, "residuals"
```
