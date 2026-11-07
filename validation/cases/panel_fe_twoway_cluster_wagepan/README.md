# panel_fe_twoway_cluster_wagepan

This is an explicit availability contract: the retained raw two-way covariance must be rejected for its demonstrated material indefiniteness. A passing result establishes that checked rejection, while fitted inference remains unavailable. It does not qualify coefficients, standard errors, confidence intervals or general fixed-effects validity.

## Executed contract

The [Hayashi script](hayashi/run.hay) loads `wagepan`, sets up the panel and fits ordinary fixed effects outside the catch. The control must use the complete sample and return finite coefficients, positive finite standard errors and residual scale, with adequate residual degrees of freedom.

Only the clustered-fit call is caught. Its error kind and message must match the observed negative-variance, material-indefiniteness or zero-variance/nonzero-cross-covariance guards. The latter is the current message for this fixed dataset because validation encounters the negative `d81` diagonal through an earlier cross-covariance check. Missing data, unknown variables, rank, environment and other covariance errors fail. If the clustered fit succeeds, an assertion outside the catch fails.

The script exports the computed numeric `availability.covariance_rejected` indicator. R and Python independently require finite raw covariance with a material negative `d81` variance and negative eigenvalues before deriving the same indicator. An admissible covariance, non-finite arithmetic, missing spectrum, asymmetric matrix or wrong reason aborts the reference instead of producing a matching zero.

Only `availability.covariance_rejected` is compared, at exact tolerance `0`. It is a numeric discrete outcome, not a boolean or a substitute coefficient/SE. The evidence class `exact` applies only to that rejection contract. No runner framework, new status, global allow-blocked policy or change to unrelated eligibility is introduced.

## Preserved statistical calculation

Both references read the shared CSV produced by `data/gen.py`, which uses Python wooldridge or the Rdatasets CSV mirror. Their covariance calculations are independent conditional on those shared observations. Both retain entity within-transformed OLS and the Cameron-Gelbach-Miller decomposition:

```
V = V_entity + V_time - V_intersection
```

The common correction remains `g/(g-1) * (n-1)/(n-k)`, where `g=min(G_entity,G_time)`. There is no diagonal clipping, eigenvalue projection, regressor deletion or altered covariance method.

The references retain unselected point diagnostics, the raw covariance under row/column names, ascending eigenvalues, `inference_available=false` and the numerical reason. They omit standard errors. The numerical diagnostic bounds remain `128*k*eps*max(abs(V))` for symmetry and `128*k*eps*max(abs(eigenvalues))` for the spectrum; raw covariance is unchanged. On the reviewed 4,360-row snapshot, `d81` variance is approximately `-7.162277e-6` and four eigenvalues are negative, with minimum approximately `-6.862758e-5`, far below roundoff bounds.

## Regression checks

`tests/panel_covariance_contract.rs` retains the independent finite/full-rank small-FE covariance rejection check. The focused suite `validation/test_panel_availability.py` exercises the actual case script's catch with genuine rejection, unexpected inference success, an unrelated clustered error and failure of the ordinary control. It also checks R/Python guard rejection of admissible, non-finite and wrong-reason reference results, and uses the unchanged generic numerical comparator at tolerance `0`.

Run that suite in the declared reference environment with the native `hay` on PATH and the restored R library selected through `R_LIBS_USER`:

```
python -B -m unittest validation/test_panel_availability.py -v
python validation/run.py --case panel_fe_twoway_cluster_wagepan --no-write
```

CI runs the focused test module after building Hayashi and adding the native binary to PATH. Windows uses the existing runner file-export transport, which is also exercised explicitly on POSIX. Changing eligibility from blocked numerical inference to a passing availability contract does not restore inference or broaden statistical qualification.
