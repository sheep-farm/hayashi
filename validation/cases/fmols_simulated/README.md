# fmols_simulated

Fully-modified OLS is currently **not-supported** for numerical validation.

The unresolved boundary is the engine's statistical contract, rather than an
absence of reference implementations. R cointReg 0.2.0 and Python arch 8.0.0
provide native FMOLS references; cointReg is included in `renv.lock`.

A source-identified replay on 14 October 2026 at Greeners
`e7f57bc8450bdd62c6c2105788b689b839158b6a` reproduces the differences described in
[Greeners #23](https://github.com/sheep-farm/Greeners/issues/23) and
[Hayashi #152](https://github.com/sheep-farm/hayashi/issues/152).
On the shared 300-observation serial-correlation diagnostic, the slope is
`0.602853025772` with SE `0.050050917894`, versus native arch
`0.578170338194` with SE `0.048302827540` and native R
`0.578090683556` with the same SE. These are fixed-case diagnostics, not a
simulation study or coverage evidence.

The corrected comparison uses Bartlett bandwidth 5 in arch and 6 in cointReg,
which give equal weights. Both references drop the first observation from the
FM regression, use the one-sided covariance bias correction and conditional
long-run variance, and disable covariance demeaning/degrees-of-freedom scaling.
Their bias multipliers differ: R uses the original sample size, whereas arch
uses the FM sample size. The resulting coefficient difference is reconciled
algebraically while retaining both native estimates; that calculation is not a
third independent reference. Probability conventions also need an explicit
decision: R reports Student-t probabilities, whereas arch uses Normal tails.

Greeners retains the first FM observation, omits the one-sided bias term,
uses unconditional long-run variance and regularises inverses. A correction
needs a coherent finite-sample contract and regression tests before numerical
validation can be promoted. The existing loose tolerances remain inactive;
this review does not adopt them as acceptance bounds or change the estimator.
