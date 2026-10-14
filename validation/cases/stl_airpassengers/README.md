# Robust STL on log AirPassengers

**Passing provisional fork repair.** Refs [#160](https://github.com/sheep-farm/hayashi/issues/160).
The immutable Greeners fork pin reproduces this declared Python full-vector
contract at unchanged tolerances. The repair is submitted in draft
[Greeners #25](https://github.com/sheep-farm/Greeners/pull/25), with its complete
third-party notice retained. Upstream contract acceptance and contribution-terms
clarification remain open; this result does not resolve either decision.

## Data and reference contract

`data/gen.R` generates all 144 monthly counts from the installed
`datasets::AirPassengers`: January 1949 to December 1960, passengers in thousands.
No network fetch, saved raw dataset, exclusions or imputation. Both estimators
take natural logarithms of the same generated integer counts. The canonical
SHA256 (each integer followed by LF, ASCII) is:

```text
8c999fa9d67e1dff475b9d0d82996f06cc5b7a2598ab359814fbb1c2f4644afd
```

The [R dataset documentation](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html)
cites Box, Jenkins and Reinsel (1994), *Time Series Analysis: Forecasting and
Control*, third edition, Prentice Hall. The data are obtained from R's datasets
package; no separate dataset licence or public-domain status is asserted.

The sole numerical reference is **statsmodels 0.14.6** `STL`, with explicit
period 12; seasonal/trend/low-pass windows 7/23/23; all degrees 1; all jumps 1;
`robust=True`, `inner_iter=2`, `outer_iter=1` (initial fit plus one robust refit).
Low-pass 23 is deliberate, not the default 13: the pinned Greeners STL uses its
trend window at that stage. Hayashi exposes `period=12, sw=7, tw=23`; this does
not prove equivalence of its internal smoothing or robustness algorithm.

The [statsmodels documentation](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.seasonal.STL.html)
describes its corrected median-based robustness weights. Python was selected
after investigating the failed R/Python gate, on the basis of that algorithm
and the initial-fit weight audit, not proximity to Hayashi. R and Python share
STL algorithm lineage; they are not fully independent methodological sources.

## Checks, repaired agreement and limits

The existing runner compares every trend, seasonal and remainder value, including
endpoints, at absolute tolerance `1e-8` (relative tolerance zero). This tolerance
was selected **after** exploratory comparison and full-precision export review;
it was not predeclared. Passing it establishes agreement for these fixed vectors,
rather than general STL accuracy.
The input hash, exact ordered indices and finite vector shapes are checked.
Hayashi predicts the model's observed vector and checks it against logged counts.

The long CSV contains 578 unique labelled values: 432 component values, 144
model-observed values, `nobs` (exactly 144) and `reconstruction_max` (target zero,
absolute tolerance `1e-12`). CSV `coef` cells use Rust float Display precision;
Python JSON uses unrounded floats. `std_err=0` cells are transport placeholders,
not standard errors. No rounded console table is parsed. Observed values also
enter the runner comparison; the case-local test checks their identity at
`1e-12`, separately from the component tolerance.

On base Hayashi `6b96de64fb3ba5f436e1ac5b1552eac147828725`, locked Greeners
`b7a34ec4`, maximum absolute errors against Python are approximately
`0.00851666` (trend), `0.01318488` (seasonal), `0.00976853` (remainder), in log
units. Hayashi reconstructs exactly. Reconstruction is an accounting check, not
STL validation. Tests include offsetting trend/remainder perturbations which
preserve reconstruction but fail the component comparison.

Against the current immutable combined fork `e7f57bc8450bdd62c6c2105788b689b839158b6a`,
all 578 quantities pass without changing data, settings or tolerances. Maximum
absolute errors are approximately `7.91e-14` (trend), `4.92e-14` (seasonal) and
`3.01e-14` (remainder); observed values match and reconstruction error is zero.
This is agreement with the specified Python contract, not R equivalence or a
whole-domain guarantee.

## Reproduce

From the repository root, with `hay` on PATH (or a local built binary), Rscript
and the pinned Python numerical dependencies installed:

```bash
python validation/run.py --case stl_airpassengers
python validation/run.py --check
python -m unittest discover -s validation/cases/stl_airpassengers -p test_case.py -v
python validation/cases/stl_airpassengers/reference/diagnostic.py
```

The first command generates data, runs only this case and regenerates the matrix;
with the provisional repaired pin it exits **0, pass**; the historical base exited
1 with a numerical failure. The metadata check and six harness tests pass. The actual CSV test requires `hay` on PATH and otherwise
reports a skip. It uses the runner's file-output rewrite on Windows and tests
both stdout and rewritten-file transport on other platforms. Ensure the runner's
selected Python uses the pinned dependencies.
The repaired-pin verification uses Python 3.11.15, statsmodels 0.14.6, NumPy 2.4.6,
SciPy 1.17.1, pandas 3.0.5, patsy 1.0.2, PyYAML 6.0.3 and R/stats/datasets 4.6.1.
The task-local Python environment was restored from the hashed requirements;
R packages were restored from renv.lock into a task-local library. Reference
package lockfiles are unchanged. See [`ENVIRONMENTS.md`](../../ENVIRONMENTS.md)
for engine provenance and environment policy.

## Optional R diagnostic and promotion gate

`reference/diagnostic.py` invokes `diagnostic.R` on the same counts. These scripts
are deliberately absent from `reference_scripts`. They check shape, order,
finite values, observed identity and reconstruction before reporting all three
component discrepancies for separate outer-0 and outer-1 model instances.

With R 4.6.1, robust R/Python errors are approximately `2.31565e-4`, `6.85304e-4`
and `6.35180e-4`; the original `1e-8` agreement gate **remains failed**, and the
diagnostic exits 1. The separate non-robust diagnostic agrees within `2e-14`.
Initial-fit residual audits use six times the true median absolute residual,
the `<=0.001` / `<=0.999` biweight cut-offs and the zero-scale rule. Python's
weights agree (zero discrepancy in this run); R's differ by about `0.0166271`.
The 14 October 2026 source experiment identifies the cause for this dataset:
R's [`stlrwt` median selection](https://github.com/wch/r-source/blob/1471c2d587e9b872a4e9b7c7947245ef9b41ef7d/src/library/stats/src/stl.c#L373)
passes ranks 73 and 72 to a partial sorter that expects ascending ranks.
Compiling that source reproduces installed R 4.6.1's vectors. Reordering only
those two ranks reduces the robust component discrepancies to below `8e-14`
and the weight discrepancy to below `4e-14`; the 143-observation control agrees
without the change because both median ranks are 72. All windows, degrees,
jumps and iteration counts are held fixed. This is a controlled diagnostic,
not a replacement R reference or proof of equivalence for other inputs. The
installed R result still fails the robust gate. See the [R STL documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/stl.html).

Every component gate passes for the provisional immutable fork. Maintain the
draft/release boundary until maintainers settle the reference contract and the
STL contribution-terms question. MSTL and robust R equivalence are outside this
case; full-matrix and platform results are separate evidence. This fixed-case
agreement alone must not close #160.
