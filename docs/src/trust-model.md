# Trust Model

Hayashi scripts can read local files, download remote files, run SQL, and execute other Hayashi scripts through `source`, `import`, plugins, and packages. Treat a Hayashi script with the same care as an R, Python, Stata, or shell script: run code only from sources you trust, and prefer read-only credentials when connecting to external systems.

## Remote data

`load "https://..." as df` downloads a remote file and then parses it as data.

Hayashi's remote loader only accepts HTTP(S) URLs, rejects localhost and non-public network targets after DNS resolution, disables redirects, applies a request timeout, and caps downloaded responses. These checks reduce accidental access to internal services and unbounded downloads, but they do not make the file trusted.

Use remote `load` only for data sources you trust. Treat downloaded rows, column names, and formats as untrusted input until validated.

## Raw SQL

`query=` is raw SQL. For SQLite and ODBC loads, Hayashi sends the query string to the configured database engine:

```hayashi
load "survey.db" as df, query="SELECT * FROM resp WHERE year >= 2000"
load "odbc://DSN=Warehouse" as df, query="SELECT * FROM panel"
```

Hayashi does not parse or restrict that SQL before execution. The database decides what the query can read or execute, subject to the connection's permissions, driver behaviour, and database configuration.

Prefer `table=` when you only need a whole table, and reserve `query=` for SQL you have reviewed. Do not build `query=` strings from untrusted input.

## ODBC

ODBC connections can point at production databases. A DSN may carry credentials, connect through a system driver, or resolve to a service managed outside the Hayashi project.

Use read-only database users for analysis scripts where possible. If a script must connect through ODBC, review the DSN, driver, server, database, and SQL before running it.

ODBC is also the main exception to Hayashi's "single binary" or "zero system dependencies" story. The default build does not require a system ODBC stack, but ODBC support is optional and requires an ODBC driver manager plus the relevant database driver, such as `unixodbc-dev` on Linux.

## Packages and imports

`source("file.hay")`, `import("module")`, auto-loaded plugins in `~/.hayashi/plugins/`, and installed packages all execute Hayashi code in your session. Remote imports also download code before running it:

```hayashi
import("https://example.com/utils.hay")
```

Install packages and import remote modules only from repositories and URLs you trust. Review package code before using it in analyses that touch confidential data, credentials, production databases, or published results.

Native or binary plugins, where available, expand the trust boundary further because they run code outside ordinary Hayashi script evaluation. Do not load unknown binaries.

## Practical checklist

- Read unfamiliar `.hay` scripts before running them.
- Use read-only credentials and development DSNs for database-backed analysis.
- Keep `query=` strings literal and reviewed; avoid constructing them from untrusted values.
- Pin or archive remote data used for reproducible work.
- Treat installed packages, remote imports, and auto-loaded plugins as executable code.
