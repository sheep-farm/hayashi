# Exporting

## DataFrames

Export a DataFrame with `export(df, "format", "path")`:

```
export(df, "csv", "output/clean_wages.csv")
export(df, "tsv", "output/clean_wages.tsv")
export(df, "json", "output/clean_wages.json")
export(df, "xlsx", "output/clean_wages.xlsx")
export(df, "parquet", "output/clean_wages.parquet")
export(df, "sqlite", "output/results.db")
export(df, "latex", "tables/summary.tex")
export(df, "html", "tables/summary.html")
```

SQLite export creates the file if it does not exist. Table name defaults to the DataFrame variable name; override with `table=`:

```
export(df, "sqlite", "results.db", table="panel_clean")
```

## Estimation results

Export a fitted model directly to LaTeX:

```
let m = ols(lwage ~ educ + exper + tenure, df, cov=robust)
export(m, "latex", "tables/ols_robust.tex")
```

## Comparison tables

`esttab` output can also be exported:

```
let m1 = ols(lwage ~ educ, df)
let m2 = ols(lwage ~ educ + exper, df)
let m3 = ols(lwage ~ educ + exper + tenure, df)

export(esttab(m1, m2, m3), "latex", "tables/comparison.tex")
export(esttab(m1, m2, m3), "html", "tables/comparison.html")
```

## Format notes

- **CSV/TSV**: UTF-8, RFC 4180 quoting.
- **JSON**: array of objects, one per row.
- **XLSX**: single sheet; no multi-sheet export.
- **Parquet**: Snappy compression by default.
- **LaTeX**: `booktabs` style (`\toprule`, `\midrule`, `\bottomrule`). Requires `\usepackage{booktabs}`.
- **HTML**: minimal `<table>` with `<thead>` / `<tbody>`. No inline CSS.
- **SQLite**: creates or appends to the database file.

## Workflow example

```
load "raw_panel.dta" as df

generate df lwage = log(wage)
filter df year >= 2000

let m = fe(lwage ~ educ + exper | firm_id, df, cov=cluster(firm_id))

export(df, "parquet", "data/clean_panel.parquet")
export(m, "latex", "tables/fe_result.tex")
```
