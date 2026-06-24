// Demonstra os formatos suportados pelo load

// ── CSV (padrão) ────────────────────────────────────────
load "exemplos/data/sample_semicolon.csv" as csv_df, sep=";"
print(csv_df)

// ── TSV (tab-separated) ─────────────────────────────────
load "exemplos/data/sample.tsv" as tsv_df
print(tsv_df)

// ── JSON (array of objects) ──────────────────────────────
load "exemplos/data/sample.json" as json_df
print(json_df)

// ── SQLite (tabela inteira) ──────────────────────────────
load "exemplos/data/sample.db" as db_df
print(db_df)

// ── SQLite (com query) ───────────────────────────────────
load "exemplos/data/sample.db" as soja, query="SELECT * FROM precos WHERE produto = 'Soja'"
print(soja)

// ── SQLite (com table) ───────────────────────────────────
load "exemplos/data/sample.db" as precos, table=precos
print(precos)

// ── Excel (quando disponível) ────────────────────────────
// load "exemplos/data/sample.xlsx" as xl_df
// load "exemplos/data/sample.xlsx" as xl_plan2, sheet=Plan2

// ── URL (baixa automaticamente) ──────────────────────────
// load "https://raw.githubusercontent.com/.../data.csv" as remote
