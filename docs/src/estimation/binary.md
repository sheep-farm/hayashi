# Binary Choice Models

## Linear Probability Model (LPM)

OLS on a binary dependent variable. Simple but can predict outside [0,1]:

```hay
use "mroz.csv"
let m_lpm = reg(inlf ~ nwifeinc + educ + exper + exper2 + age + kidslt6 + kidsge6, mroz,
                cov=robust)
```

## Logit

Models log-odds as a linear function of regressors:

```hay
let m_logit = logit(inlf ~ nwifeinc + educ + exper + exper2 + age + kidslt6 + kidsge6, mroz)
```

Output reports coefficients (log-odds ratios), standard errors, z-statistics, pseudo R-squared, and log-likelihood.

## Probit

Models the latent index through the standard normal CDF:

```hay
let m_probit = probit(inlf ~ nwifeinc + educ + exper + exper2 + age + kidslt6 + kidsge6, mroz)
```

Coefficients are not directly interpretable as marginal effects. Use `margins` for that.

## Marginal Effects

Average marginal effects (AME) give the average change in P(Y=1) for a unit change in X:

```hay
let m = logit(inlf ~ nwifeinc + educ + exper + exper2 + age + kidslt6 + kidsge6, mroz)
margins(m, type=ame)
```

```
Average Marginal Effects
──────────────────────────────────
             dy/dx     SE      p
──────────────────────────────────
nwifeinc    -0.004   0.002   0.012
educ         0.038   0.007   0.000
exper        0.039   0.006   0.000
exper2      -0.001   0.000   0.001
age         -0.016   0.002   0.000
kidslt6     -0.262   0.034   0.000
kidsge6      0.013   0.013   0.322
──────────────────────────────────
```

Interpretation: each additional year of education increases the probability of labor force participation by 3.8 percentage points, on average.

## Comparing All Three

```hay
let m1 = reg(inlf ~ nwifeinc + educ + exper + exper2 + age + kidslt6 + kidsge6, mroz,
             cov=robust)
let m2 = logit(inlf ~ nwifeinc + educ + exper + exper2 + age + kidslt6 + kidsge6, mroz)
let m3 = probit(inlf ~ nwifeinc + educ + exper + exper2 + age + kidslt6 + kidsge6, mroz)

esttab(m1, m2, m3)
```

LPM coefficients approximate AME from logit/probit for observations near the sample mean.

## Ordered Models

For ordinal dependent variables (e.g., satisfaction: 1-5):

```hay
// Ordered logit
ologit(satisfaction ~ income + educ + age, survey)

// Ordered probit
oprobit(satisfaction ~ income + educ + age, survey)
```

Reports cut points (thresholds) alongside coefficients. Interpretation: a positive coefficient shifts the distribution toward higher categories.

## Multinomial Logit

For unordered categorical outcomes with 3+ categories (e.g., transport mode: car, bus, train):

```hay
mlogit(transport ~ income + distance + age, commute, base=1)
```

Reports one set of coefficients per category (relative to the base). Each coefficient reflects the log-odds of choosing category j over the base.

## Prediction

```hay
let m = logit(inlf ~ nwifeinc + educ + exper + exper2 + age + kidslt6 + kidsge6, mroz)

// Predicted probabilities
predict mroz p_hat = m, "pr"
```
