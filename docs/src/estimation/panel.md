# Panel Data

## Setting the Panel Structure

Before any panel estimation, declare the panel identifiers:

```hay
use "wagepan.csv"
xtset(wagepan, id=nr, time=year)
```

Hayashi validates the panel: checks for duplicates, reports balance, and identifies gaps.

## Fixed Effects (FE)

Eliminates time-invariant unobserved heterogeneity by within-transformation:

```hay
let m_fe = fe(lwage ~ hours + tenure + exper + married + union, wagepan)
```

Variables that do not vary within a panel unit are automatically dropped:

```
note: educ dropped (time-invariant within nr)
note: black dropped (time-invariant within nr)
```

FE reports the within R-squared and the F-test for joint significance of individual effects.

## Random Effects (RE)

Assumes unobserved heterogeneity is uncorrelated with regressors:

```hay
let m_re = re(lwage ~ hours + tenure + exper + married + union + educ + black, wagepan)
```

RE is more efficient than FE when its assumptions hold, and allows estimation of time-invariant variable coefficients.

## Hausman Test

Tests whether FE and RE estimates are systematically different:

```hay
let m_fe = fe(lwage ~ hours + tenure + exper + married + union, wagepan)
let m_re = re(lwage ~ hours + tenure + exper + married + union, wagepan)

hausman(m_fe, m_re)
```

```
Hausman Test: FE vs RE
  chi2(5) = 32.41
  p-value = 0.0000

Conclusion: reject RE at 1% level; use FE.
```

A low p-value rejects RE in favor of FE -- the individual effects are correlated with regressors.

## Pooled OLS with Clustered SE

When panel structure is ignored but standard errors need to account for within-unit correlation:

```hay
let m_pool = reg(lwage ~ hours + tenure + exper + married + union + educ, wagepan,
                 cov=cluster(nr))
```

## Comparison Table

```hay
let m1 = reg(lwage ~ hours + tenure + exper + married + union + educ, wagepan,
             cov=cluster(nr))
let m2 = fe(lwage ~ hours + tenure + exper + married + union, wagepan)
let m3 = re(lwage ~ hours + tenure + exper + married + union + educ, wagepan)

esttab(m1, m2, m3)
```

```
──────────────────────────────────────────────────
              (1)         (2)         (3)
            Pooled       FE          RE
──────────────────────────────────────────────────
hours       0.001      -0.002       0.001
           (0.002)     (0.002)     (0.002)
tenure      0.014***    0.011***    0.012***
           (0.003)     (0.003)     (0.003)
exper       0.030***    0.042***    0.036***
           (0.005)     (0.007)     (0.005)
married     0.108***    0.047**     0.064***
           (0.021)     (0.018)     (0.017)
union       0.180***    0.080***    0.106***
           (0.025)     (0.019)     (0.018)
educ        0.091***   (omitted)    0.092***
           (0.006)                  (0.011)
──────────────────────────────────────────────────
N             4360       4360        4360
R-sq         0.186      0.178       0.182
Method      Pooled       FE          RE
SE type     Cluster    Default     Default
──────────────────────────────────────────────────
```

Note how `educ` is omitted in FE (time-invariant) but estimated in RE and Pooled OLS. The union premium drops under FE, suggesting positive selection into unions.

Two-way FE (entity + time): `fe(Y ~ X, df, fe=nr+year)`
