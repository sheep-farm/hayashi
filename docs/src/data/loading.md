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
load "panel.parquet" as df, columns=[ticker, date, close], where="ticker == \"AAPL\""
```

`query=` is raw SQL executed by SQLite or the configured ODBC database. Use `table=` for simple table loads, and see the [Trust Model](../trust-model.md#raw-sql) before running SQL against shared databases.

`query=` cannot be combined with `columns=` or `where=` — if you need custom SQL, write the projection and filter inside the query string yourself.

## Column projection and row filtering (`columns=`, `where=`)

For large datasets, loading every column and row into RAM can be wasteful or
impossible. The `columns=` and `where=` options push projection and filtering
down to the data source, so only the requested columns and matching rows are
materialized.

```
load "cotacoes.parquet" as aapl, columns=[ticker, date, adj_close], where="ticker == \"AAPL\""
load "panel.db" as df, table=prices, columns=[date, close], where="close > 100"
load "survey.csv" as df, where="age >= 18 && region == \"South\""
load "panel.dta" as df, columns=[id, year, y], where="year >= 2000"
load "sheet.xlsx" as df, columns=[name, score], where="score > 75"
```

### Supported sources

| Source    | `columns=`            | `where=`              | Pushdown mechanism |
|-----------|-----------------------|-----------------------|--------------------|
| Parquet   | yes                   | yes                   | Arrow `ProjectionMask` + `RowFilter` (filter evaluated during row-group scan) + **row group pruning** by min/max statistics |
| SQLite    | yes                   | yes                   | `SELECT cols FROM t WHERE pred` (validated against `PRAGMA table_info`) |
| ODBC      | yes                   | yes                   | same as SQLite |
| CSV / TSV | yes                   | yes                   | projection in read loop, row-by-row predicate evaluation |
| DTA       | yes                   | yes                   | projection in `read_record`, row-by-row predicate |
| Excel     | yes                   | yes                   | projection after `worksheet_range`, row filter on cells |
| JSON      | not yet               | not yet               | — |

Passing `columns=` or `where=` to a JSON load returns an error.

### `columns=` syntax

Accepts a list of column names (identifiers or string literals):

```
columns=[ticker, date, adj_close]
columns=["ticker", "date", "adj_close"]
columns=ticker              // single column also accepted
```

Column names are matched against the source schema before any data is read;
unknown columns produce a clear error listing the available ones.

### `where=` syntax

`where=` accepts a Hayashi expression of the form `column OP literal`,
combined with `&&`, `||`, `!`, and `in [...]`. The expression is parsed by
the Hayashi parser and normalized into a `RowPredicate` that each loader
evaluates (or pushes down to the source).

```
where="ticker == \"AAPL\""                              // string equality
where="price > 100"                                     // numeric comparison
where="price > 100 && volume > 1e6"                     // AND
where="ticker in [\"AAPL\", \"MSFT\"]"                  // membership
where="!(sector == \"Finance\")"                        // negation
where="ano >= 2022 && produto == \"Soja\""              // combined
```

Supported operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `&&` (and),
`||` (or), `!` (not). Comparisons must be `column OP literal` — comparing
two columns is not supported (use `generate` + `filter` for that).

### Row group pruning (Parquet)

In addition to `RowFilter` (which evaluates the predicate on each row during
scan), the Parquet loader reads per-row-group min/max statistics from the file
metadata and skips row groups where the predicate cannot possibly match. This
is effective when the data is sorted or clustered by the filtered column.

On a 799 MB / 30 M-row / 8 292-row-group Parquet file sorted by ticker,
`where="ticker == \"AAPL\""` pruned 8 291 of 8 292 row groups, reducing load
time from ~62 s (full scan) to ~312 ms with ~60 MB peak RSS.

For point lookups on a single value (e.g. one ticker), SQLite with a B-tree
index on `(ticker, date)` is still faster (~42 ms, ~26 MB RSS) because it seeks
directly without reading all row-group metadata. For full-column analytics across
all tickers, Parquet with pruning is superior due to columnar compression.

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
- **Temporal types in Parquet** (`Timestamp(s|ms|µs|ns)`, `Date32`, `Date64`) are
  converted to Hayashi strings formatted as ISO 8601 (`YYYY-MM-DD` when the time
  component is midnight, otherwise `YYYY-MM-DDTHH:MM:SS`). To extract components
  vectorially use `generate df ano = substr(date, 0, 4)`; to convert a single ISO
  date string to a Unix timestamp use the scalar builtin `date("YYYY-MM-DD")` or
  `datetime("YYYY-MM-DDTHH:MM:SS")`.
- JSON expects an array of objects (one object per row).
