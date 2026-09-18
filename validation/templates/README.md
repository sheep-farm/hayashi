# <case-id>

Short description of the validation case: research question, dataset, and
estimator.

## Dataset

- **Name:** <dataset name>
- **Source:** <URL, package, or DGP from `book_pt_BR/codes/XX.hay`>
- **Licence:** <licence>
- **Size:** <rows x cols>

## Analysis

Describe the regression or workflow being validated.

## Reference implementation

- **R:** package `...`, function `...`, options `...`
- **Python:** package `...`, function `...`, options `...`
- **Stata:** command `...`, options `...` (if applicable)

Optional: document reviewed [reference evidence classes](../README.md#reference-evidence-classes)
for each reference and literal tolerance key, with a rationale in `case.yml`.
Unannotated references are unclassified, not exact. Distinguish related
diagnostics from evidence for the estimator and its inference contract.

## Compared quantities

- coefficients
- standard errors
- R-squared
- number of observations

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | ... |
| standard_errors | 1e-6 | ... |
| r_squared | 1e-8 | ... |
| nobs | 0 | exact count |
