# fmols_simulated

This case compares Phillips-Hansen fully modified OLS coefficients and standard
errors with native arch 8.0.0 and convention-adjusted cointReg 0.2.0 on four
fixed N=300 fixtures. `data/gen.py` writes one CSV with fixture and ordered row
identifiers, y and x. Both references reject missing, repeated, reordered,
unknown or non-finite fixture rows before fitting. Cointegration and weak
dependence are assumed by the estimator; this case does not test them.

## Fixed designs and holdout

All innovations below are independent standard Normal vectors, except for the
original IID control's stated scales. Each generator draws all dx before all e.
The endogeneity and serial-correlation designs reproduce the existing frozen
engine fixtures; the holdout uses a new seed and combines both dynamics.

| Fixture | Generator/seed | Design |
| --- | --- | --- |
| IID control | RandomState/MT19937, 42 | x=cumsum(Normal(0,.1)); y=.5*x+Normal(0,.5), preserving the original draws |
| Endogeneity | PCG64, 314159 | x=cumsum(dx); u=.75*dx+e; y=.25+.5*x+u |
| Serial correlation | PCG64, 271828 | x=cumsum(dx); u[0]=e[0]/sqrt(1-.7^2); u[t]=.7*u[t-1]+e[t]; y=.25+.5*x+u |
| Holdout | PCG64, 2026103101 | Generate 600 rows from x=cumsum(dx), u[-1]=0, u[t]=.5*u[t-1]+.5*dx[t]+e[t], y=.25+.5*x+u; discard 300 burn-in rows, retaining the random-walk levels |

The coefficient/SE tolerances remain the existing absolute `1e-10`; sample size
and bandwidth comparisons remain exact. The 24 literal selectors and fixture
designs were frozen before new reference comparisons. The three training
fixtures passed R/Python reference-to-reference checks before the holdout was
fitted. The independent holdout passed without changing data, settings or
tolerances. The largest R/Python coefficient/SE differences were `1.40e-15`
on training and `3.73e-13` on holdout in the locked Python 3.11.15/R 4.6.1 replay.
These fixtures check calculation agreement, not repeated-sampling coverage.

## Shared finite-sample contract

Initial OLS uses all N observations. The fully modified regression uses rows
2..N, so `n_obs=299`; the one-sided bias multiplier is n=N-1. Bartlett lag count
B=5 has weights `1-l/(B+1)`, without HAC centring or degrees-of-freedom adjustment.
Uncertainty uses the conditional long-run variance and asymptotic Normal
inference. Rank or unresolved precision failures return errors rather than ridge
estimates. The retained R-squared describes initial all-N OLS.
Native arch returns residuals for all N rows; native cointReg pads the first
residual with NA. Those returned-vector lengths do not change the final FM
regression sample n=N-1. Diagnostics distinguish these counts.

The Python reference explicitly selects this contract in native arch. Native
cointReg uses bandwidth B+1=6 for the same Bartlett weights, but its bias
multiplier is N. The R reference adds `(Z'Z)^-1*q` to its native coefficient
vector to select n, where q is the slope-only one-sided bias term. This labelled
convention adjustment is not another independent reference; native coefficients
and the adjustment remain in diagnostics. Native R standard errors agree under
the common sample/covariance convention. Its Student-t probabilities do not
qualify Normal inference.

The case selects each fixture's intercept/slope coefficients and SEs with
absolute tolerance `1e-10`, plus exact sample size and Bartlett lag count. IID
keeps the original labels; other labels start with the fixture name and `_`.
It does not
select probabilities or claim general uncertainty calibration. The sample-size
and bandwidth rows have unselected CSV SE placeholders; they carry no inference.
Both reference JSON outputs record package versions, requested/realised bands
and sample conventions. R retains native coefficients and the convention
adjustment separately for every fixture; Python also records the input hash.
Engine tests
separately check Normal probabilities, four native coefficient/SE/covariance
fixtures, multivariate transforms, rank/domain failures and tiny tails.

Reference agreement and interval coverage are separate findings. A prespecified
actual-Rust study with 500 replications per design found nominal 95% slope
coverage of 90.0% at N=300 (Wilson 95% interval 87.06% to 92.33%) and 91.8% at
N=1,000 (89.06% to 93.90%) under endogenous autoregressive errors. Both Monte
Carlo intervals exclude nominal 95%. FMOLS remains experimental; these results
do not qualify general coverage or causal identification.

The original discrepancies are tracked in [Greeners #23](https://github.com/sheep-farm/Greeners/issues/23)
and [Hayashi #152](https://github.com/sheep-farm/hayashi/issues/152). The correction
adds the missing one-sided bias term, selects the N-1 FM sample and conditional
variance, and replaces hidden regularisation with rank-checked QR arithmetic.

## Reproduce

With the locked reference environments restored and an experimental `hay` on
PATH, run from the repository root:

```sh
python validation/run.py --case fmols_simulated --experimental --no-write
python -m unittest discover -s validation/cases/fmols_simulated -p test_case.py -v
python validation/run.py --check
```

The actual consumer test requires `hay` and `Rscript`; it reports a skip when
they are absent. The remaining tests check the original IID values, fixed
fixture identities, input guards and all 24 literal comparison selectors.
CI's metadata job includes these tests, and the full Validation workflow runs
the actual four-fixture case. The first holdout evaluation followed the frozen
training checks; subsequent replays are regression checks of that fixed suite,
not fresh holdouts or coverage studies. Issue #152 remains subject to review
and hosted execution; this local replay does not close it automatically.
