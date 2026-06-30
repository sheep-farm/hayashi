#!/usr/bin/env python3
"""PR 1: standardise a narrow set of user-facing runtime errors to English.

Scope: argument-validation paths only -- row-wise helpers, descriptive
statistics, string/range arity checks, and estimator/diagnostic model
requirement + series validation errors. Strings are matched exactly
(including surrounding quotes) so duplicate occurrences are all updated and
nothing partial is touched. Run from the repository root.
"""
import sys
from pathlib import Path

TARGET = Path("src/lang/interpreter.rs")

# Exact (old -> new) replacements. Keys include the surrounding double quotes
# to guarantee we only touch complete string literals.
REPLACEMENTS = {
    # --- row-wise helpers (rowmean / rowsum / ...) + group ---
    '"{func}() requer ao menos uma coluna"': '"{func}() requires at least one column"',
    '"group() requer o nome de uma coluna"': '"group() requires a column name"',

    # --- descriptive statistics ---
    '"median: primeiro argumento deve ser DataFrame"': '"median: first argument must be a DataFrame"',
    '"median: segundo argumento deve ser nome de variável"': '"median: second argument must be a variable name"',
    '"median(): lista vazia"': '"median(): empty list"',
    '"variance: primeiro argumento deve ser DataFrame"': '"variance: first argument must be a DataFrame"',
    '"variance: segundo argumento deve ser nome de variável"': '"variance: second argument must be a variable name"',
    '"variance(): requer pelo menos 2 observações"': '"variance(): requires at least 2 observations"',
    '"quantile: primeiro argumento deve ser DataFrame"': '"quantile: first argument must be a DataFrame"',
    '"quantile: segundo argumento deve ser nome de variável"': '"quantile: second argument must be a variable name"',
    '"cov(): segundo argumento deve ser nome de variável"': '"cov(): second argument must be a variable name"',
    '"cov(): terceiro argumento deve ser nome de variável"': '"cov(): third argument must be a variable name"',
    '"cov(): requer pelo menos 2 observações"': '"cov(): requires at least 2 observations"',
    '"corr_pair(): segundo argumento deve ser nome de variável"': '"corr_pair(): second argument must be a variable name"',
    '"corr_pair(): terceiro argumento deve ser nome de variável"': '"corr_pair(): third argument must be a variable name"',
    '"corr_pair(): requer pelo menos 2 observações"': '"corr_pair(): requires at least 2 observations"',

    # --- string / range arity checks ---
    '"len() requires exactly 1 argumento"': '"len() requires exactly 1 argument"',
    '"substr(s, início [, comprimento]) requer 2 ou 3 argumentos"': '"substr(s, start [, length]) requires 2 or 3 arguments"',
    '"range(start, end [, step]) requer 2 ou 3 argumentos"': '"range(start, end [, step]) requires 2 or 3 arguments"',

    # --- estimator / diagnostic model-requirement errors ---
    '"irf() requer um modelo VAR"': '"irf() requires a VAR model"',
    '"fevd() requer um modelo VAR"': '"fevd() requires a VAR model"',
    '"sirf() requer um modelo SVAR"': '"sirf() requires an SVAR model"',
    '"sfevd() requer um modelo SVAR"': '"sfevd() requires an SVAR model"',
    '"cooks() requer um modelo OLS"': '"cooks() requires an OLS model"',
    '"vif() requer um modelo OLS"': '"vif() requires an OLS model"',
    '"white() requer um modelo OLS"': '"white() requires an OLS model"',
    '"reset() requer um modelo OLS"': '"reset() requires an OLS model"',
    '"leverage() requer um modelo OLS"': '"leverage() requires an OLS model"',
    '"condnum() requer um modelo OLS"': '"condnum() requires an OLS model"',
    '"durbinwatson() requer um modelo OLS"': '"durbinwatson() requires an OLS model"',
    '"bgodfrey() requer um modelo OLS"': '"bgodfrey() requires an OLS model"',
    '"forecast() requer um modelo ARIMA"': '"forecast() requires an ARIMA model"',
    '"forecast_vol() requer um modelo GARCH"': '"forecast_vol() requires a GARCH model"',
    '"forecast_vol() requer um modelo GARCH/EGARCH/GJRGARCH"': '"forecast_vol() requires a GARCH/EGARCH/GJRGARCH model"',
    '"diagnostics() requer um modelo (OLS, GARCH ou ARIMA)"': '"diagnostics() requires a model (OLS, GARCH, or ARIMA)"',
    '"margins() requer um modelo estimado como argumento"': '"margins() requires an estimated model as an argument"',

    # --- series / model argument validation ---
    '"ljungbox() requer uma série ou modelo"': '"ljungbox() requires a series or model"',
    '"ljungbox(df, varname): second argument must be o nome da coluna"': '"ljungbox(df, varname): second argument must be a column name"',
    '"ljungbox(): argumento deve ser DataFrame, GARCH, ARIMA ou OLS"': '"ljungbox(): argument must be a DataFrame, GARCH, ARIMA, or OLS"',
    '"jb() requer uma série ou modelo"': '"jb() requires a series or model"',
    '"jb(df, varname): second argument must be o nome da coluna"': '"jb(df, varname): second argument must be a column name"',
    '"jb(): argumento deve ser DataFrame, OLS, ARIMA ou GARCH"': '"jb(): argument must be a DataFrame, OLS, ARIMA, or GARCH"',
    '"archtest() requer uma série ou modelo GARCH"': '"archtest() requires a series or GARCH model"',
    '"archtest(df, varname): second argument must be o nome da coluna"': '"archtest(df, varname): second argument must be a column name"',
    '"archtest(): primeiro argumento deve ser um DataFrame ou modelo GARCH"': '"archtest(): first argument must be a DataFrame or GARCH model"',

    # --- VAR / ARIMA / VECM constructor argument errors ---
    '"arima() requer (dataframe, variável, p=, d=, q=)"': '"arima() requires arguments: dataframe, variable, p=, d=, q="',
    '"var() requer (dataframe, var1, var2, ..., lags=p)"': '"var() requires arguments: dataframe, var1, var2, ..., lags=p"',
    '"vecm() requer (dataframe, var1, var2, ..., lags=p, rank=r)"': '"vecm() requires arguments: dataframe, var1, var2, ..., lags=p, rank=r"',

    # --- companion type-errors: wrong model kind passed to OLS-only diagnostics ---
    '"lincom() suporta apenas modelos OLS"': '"lincom() only supports OLS models"',
    '"leverage() suporta apenas modelos OLS"': '"leverage() only supports OLS models"',
    '"cooks() suporta apenas modelos OLS"': '"cooks() only supports OLS models"',
    '"vif() suporta apenas modelos OLS"': '"vif() only supports OLS models"',
    '"condnum() suporta apenas modelos OLS"': '"condnum() only supports OLS models"',
    '"durbinwatson() suporta apenas modelos OLS"': '"durbinwatson() only supports OLS models"',
    '"white() suporta apenas modelos OLS"': '"white() only supports OLS models"',
    '"reset() suporta apenas modelos OLS"': '"reset() only supports OLS models"',
    '"bgodfrey() suporta apenas modelos OLS"': '"bgodfrey() only supports OLS models"',
    '"omnibus() suporta apenas modelos OLS"': '"omnibus() only supports OLS models"',
    '"harveycollier() suporta apenas modelos OLS"': '"harveycollier() only supports OLS models"',
    '"residplot() suporta apenas modelos OLS; para GLM use predict + scatter"': '"residplot() only supports OLS models; for GLM use predict + scatter"',
    '"gqtest(): suporta apenas modelos OLS"': '"gqtest(): only supports OLS models"',
    '"bphet(): suporta apenas modelos OLS"': '"bphet(): only supports OLS models"',
}



def main() -> int:
    if not TARGET.exists():
        print(f"error: {TARGET} not found (run from repo root)", file=sys.stderr)
        return 1

    text = TARGET.read_text(encoding="utf-8")
    total = 0
    missing = []
    for old, new in REPLACEMENTS.items():
        count = text.count(old)
        if count == 0:
            missing.append(old)
            continue
        text = text.replace(old, new)
        total += count
        print(f"  [{count}x] {old}  ->  {new}")

    TARGET.write_text(text, encoding="utf-8")
    print(f"\nApplied {total} replacement(s) across {len(REPLACEMENTS) - len(missing)} pattern(s).")
    if missing:
        print(f"\n{len(missing)} pattern(s) not found (already translated or moved):")
        for m in missing:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
