# Loading Data

Hayashi reads 8 formats through a single syntax:

```
load "path" as df
```

## Formats

```
load "wages.csv" as df
load "wages.tsv" as df
load "wages.json" as df
load "wages.dta" as df
load "wages.xlsx" as df
load "wages.parquet" as df
load "survey.db" as df
load "dsn=PgProd" as df
```

Format is inferred from extension. SQLite loads the first table by default; ODBC requires a DSN string.

## Options

```
load "wages.xlsx" as df, sheet="Panel2010"
load "survey.db" as df, table="respondents"
load "survey.db" as df, query="SELECT * FROM resp WHERE year >= 2000"
load "raw.txt" as df, sep="|"
```

`query=` is raw SQL executed by SQLite or the configured ODBC database. Use `table=` for simple table loads, and see the [Trust Model](../trust-model.md#raw-sql) before running SQL against shared databases.

## Remote files

`load` accepts URLs directly:

```
load "https://example.com/data/cpi.csv" as df
```

Remote files are downloaded and parsed as data. Hayashi validates HTTP(S) URLs and applies network/download limits, but remote data is still untrusted input. See the [Trust Model](../trust-model.md#remote-data).

## Multiple DataFrames

Unlike Stata, Hayashi holds any number of DataFrames in memory at once:

```
load "firms.csv" as firms
load "returns.parquet" as returns
load "macro.dta" as macro

let merged = merge(firms, returns, on="permno")
```

No `preserve` / `restore` juggling to work with multiple datasets.

## Inline data

For small examples or tests, define data directly with `input`:

```
input df
    name    age  wage
    "Alice"  30  52000
    "Bob"    28  48000
    "Carol"  35  61000
end
```

Column types are inferred automatically.

## ODBC connections

```
load "dsn=MyPostgres" as df, query="SELECT * FROM panel WHERE country = 'BRA'"
```

Requires a configured ODBC DSN on the system. Hayashi links `libodbc` at runtime.

ODBC DSNs can point at production databases and require external system drivers. Prefer read-only credentials for analysis scripts. See the [Trust Model](../trust-model.md#odbc).

## Notes

- DTA files support Stata 12--118 format (`.dta` versions 113--118).
- Excel reads `.xlsx` only (not legacy `.xls`).
- Parquet preserves column types exactly; prefer it for large datasets.
- JSON expects an array of objects (one object per row).
