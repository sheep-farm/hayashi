# Error-message style guide

Hayashi's user-facing runtime errors must be written in **English**. This
document defines the terminology and phrasing conventions so that messages stay
consistent as the codebase is migrated away from its earlier mixed
Portuguese/English wording.

The migration is staged across several PRs (see the tracking issue). Do **not**
attempt to translate the whole codebase in one change, and do **not** introduce
an i18n/localization layer — the target output language is English, and the
error infrastructure is already centralised at the type level via
`HayashiError` in `src/lang/error.rs`.

## Glossary (Portuguese → English)

| Portuguese | English |
|---|---|
| requer | requires |
| ao menos / pelo menos | at least |
| coluna | column |
| nome da coluna | column name |
| variável | variable |
| nome de variável | variable name |
| modelo | model |
| inválido / inválida | invalid |
| esperado / esperada | expected |
| não encontrado | not found |
| vazio / vazia | empty |
| lista vazia | empty list |
| deve ser | must be |
| devem ser | must be |
| não podem ser vazios | cannot be empty |
| argumento | argument |
| primeiro / segundo / terceiro argumento | first / second / third argument |
| observações | observations |
| fórmula | formula |
| covariável | covariate |
| opção | option |
| série | series |
| defasagem | lag |
| série temporal | time series |
| resíduos | residuals |
| graus de liberdade | degrees of freedom |

## Phrasing rules

1. **Prefix with the function/command name**, then a colon or `()`:
   - `cooks() requires an OLS model`
   - `cancorr: 'xvars' and 'yvars' cannot be empty`
2. **Lowercase after the prefix** unless it is a proper noun or acronym
   (`DataFrame`, `OLS`, `VAR`, `ARIMA`, `GARCH`).
3. Use **"requires"** for missing preconditions:
   - `rowmean() requires at least one column`
4. Use **"must be"** for type/shape constraints:
   - `archtest(df, varname): second argument must be a column name`
5. Use **"not found"** for lookup failures, and **quote** user-supplied
   identifiers:
   - `ab: id column '{id_col}' not found`
6. Use **"cannot be empty"** for empty input.
7. Prefer **"cannot"** over "can't".
8. **No trailing period** on single-sentence errors.
9. Use **`a`/`an`** correctly: `an OLS model`, `a VAR model`, `an ARIMA model`.

## Examples (before → after)

| Before | After |
|---|---|
| `{func}() requer ao menos uma coluna` | `{func}() requires at least one column` |
| `group() requer o nome de uma coluna` | `group() requires a column name` |
| `cooks() requer um modelo OLS` | `cooks() requires an OLS model` |
| `arima() requer (dataframe, variável, p=, d=, q=)` | `arima() requires arguments: dataframe, variable, p=, d=, q=` |
| `archtest(df, varname): second argument must be o nome da coluna` | `archtest(df, varname): second argument must be a column name` |

## Regression guard

`scripts/check_pt_errors.sh` is a diff-scoped CI check: it fails only when
**newly added** Rust lines contain common Portuguese terms. Legacy strings that
are still pending translation do not trip it. As the remaining strings are
migrated in follow-up PRs, the guard can eventually be tightened to scan the
whole tree.

Run locally before opening a PR:

```bash
scripts/check_pt_errors.sh upstream/dev
```
