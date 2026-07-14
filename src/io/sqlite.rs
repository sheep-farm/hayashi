use crate::lang::error::{HayashiError, Result};
use crate::lang::predicate::RowPredicate;
use greeners::DataFrame;
use rusqlite::Connection;
use std::collections::HashMap;

pub fn load_sqlite(
    path: &str,
    table: Option<&str>,
    query: Option<&str>,
    columns: Option<&[String]>,
    predicate: Option<&RowPredicate>,
) -> Result<(DataFrame, usize)> {
    let conn = Connection::open(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot open '{path}': {e}")))?;

    let sql = if let Some(q) = query {
        // Caminho legado: query= explícita. columns=/where= já rejeitados
        // em exec_load antes de chegar aqui.
        q.to_string()
    } else {
        // Montar SELECT a partir de table= (ou primeira tabela) + columns= +
        // where=. Identifiers de tabela/columns são escapados; literais do
        // where são escapados por RowPredicate::to_sql.
        let tbl = match table {
            Some(t) => t.to_string(),
            None => first_table(&conn)?,
        };

        // Validar columns= e where= contra o schema real da tabela.
        // O SQLite, em modo compatível, trata `"xxx"` como string literal
        // quando a coluna não existe — então a validação explícita evita
        // resultados silenciosamente errados.
        let table_cols = table_columns(&conn, &tbl)?;
        if let Some(cols) = columns {
            for c in cols {
                if !table_cols.contains(c) {
                    return Err(HayashiError::Runtime(format!(
                        "load sqlite: column '{c}' not found in table '{tbl}' — available: {}",
                        table_cols.join(", ")
                    )));
                }
            }
        }
        if let Some(p) = predicate {
            for c in p.referenced_columns() {
                if !table_cols.contains(&c) {
                    return Err(HayashiError::Runtime(format!(
                        "load sqlite: where references unknown column '{c}' in table '{tbl}' — available: {}",
                        table_cols.join(", ")
                    )));
                }
            }
        }

        let cols_clause = match columns {
            Some(cols) if !cols.is_empty() => cols
                .iter()
                .map(|c| quote_sqlite_identifier(c))
                .collect::<Result<Vec<_>>>()?
                .join(", "),
            _ => "*".to_string(),
        };
        let mut s = format!(
            "SELECT {cols_clause} FROM {}",
            quote_sqlite_identifier(&tbl)?
        );
        if let Some(p) = predicate {
            s.push_str(&format!(" WHERE {}", p.to_sql()));
        }
        s
    };

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| HayashiError::Runtime(format!("SQL error: {e}")))?;

    let col_count = stmt.column_count();
    let col_names: Vec<String> = (0..col_count)
        .map(|i| stmt.column_name(i).unwrap_or("unnamed").to_string())
        .collect();

    let mut raw: Vec<Vec<rusqlite::types::Value>> = Vec::new();

    let rows = stmt
        .query_map([], |row| {
            let mut vals = Vec::with_capacity(col_count);
            for i in 0..col_count {
                vals.push(row.get::<_, rusqlite::types::Value>(i)?);
            }
            Ok(vals)
        })
        .map_err(|e| HayashiError::Runtime(format!("query error: {e}")))?;

    for row in rows {
        let r = row.map_err(|e| HayashiError::Runtime(format!("row error: {e}")))?;
        raw.push(r);
    }

    let n_rows = raw.len();

    let mut is_numeric: Vec<bool> = vec![true; col_count];
    for row in &raw {
        for (i, val) in row.iter().enumerate() {
            match val {
                rusqlite::types::Value::Text(_) | rusqlite::types::Value::Blob(_) => {
                    is_numeric[i] = false;
                }
                _ => {}
            }
        }
    }

    let mut float_cols: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut str_cols: HashMap<usize, Vec<String>> = HashMap::new();

    for i in 0..col_count {
        if is_numeric[i] {
            let vals: Vec<f64> = raw
                .iter()
                .map(|row| match &row[i] {
                    rusqlite::types::Value::Integer(v) => *v as f64,
                    rusqlite::types::Value::Real(v) => *v,
                    rusqlite::types::Value::Null => f64::NAN,
                    _ => f64::NAN,
                })
                .collect();
            float_cols.insert(i, vals);
        } else {
            let vals: Vec<String> = raw
                .iter()
                .map(|row| match &row[i] {
                    rusqlite::types::Value::Text(s) => s.clone(),
                    rusqlite::types::Value::Integer(v) => format!("{v}"),
                    rusqlite::types::Value::Real(v) => format!("{v}"),
                    rusqlite::types::Value::Null => String::new(),
                    rusqlite::types::Value::Blob(b) => format!("<blob {} bytes>", b.len()),
                })
                .collect();
            str_cols.insert(i, vals);
        }
    }

    let mut builder = DataFrame::builder();
    for (i, name) in col_names.iter().enumerate() {
        if is_numeric[i] {
            let vals = float_cols.remove(&i).unwrap();
            builder = builder.add_column(name, vals);
        } else {
            let vals = str_cols.remove(&i).unwrap();
            builder = builder.add_string(name, vals);
        }
    }

    let df = builder
        .build()
        .map_err(|e| HayashiError::Runtime(format!("DataFrame build error: {e}")))?;

    Ok((df, n_rows))
}

fn first_table(conn: &Connection) -> Result<String> {
    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 1")
        .map_err(|e| HayashiError::Runtime(format!("cannot list tables: {e}")))?;

    let name: String = stmt
        .query_row([], |row| row.get(0))
        .map_err(|_| HayashiError::Runtime("database has no tables".into()))?;

    Ok(name)
}

/// Lista os nomes das colunas de uma tabela via `PRAGMA table_info`.
fn table_columns(conn: &Connection, table: &str) -> Result<Vec<String>> {
    let quoted = quote_sqlite_identifier(table)?;
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({quoted})"))
        .map_err(|e| HayashiError::Runtime(format!("cannot read schema of '{table}': {e}")))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| HayashiError::Runtime(format!("schema query error: {e}")))?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| HayashiError::Runtime(format!("schema row error: {e}")))?);
    }
    if out.is_empty() {
        return Err(HayashiError::Runtime(format!(
            "table '{table}' not found or has no columns"
        )));
    }
    Ok(out)
}

fn quote_sqlite_identifier(name: &str) -> Result<String> {
    if name.contains('\0') {
        return Err(HayashiError::Runtime(
            "SQLite identifier contains NUL byte".into(),
        ));
    }

    Ok(format!("\"{}\"", name.replace('"', "\"\"")))
}

pub fn write_sqlite(df: &greeners::DataFrame, path: &str, table: &str) -> Result<()> {
    let mut conn = Connection::open(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot open '{path}': {e}")))?;

    let col_names = df.column_names();
    let quoted_table = quote_sqlite_identifier(table)?;

    let col_defs: Vec<String> = col_names
        .iter()
        .map(|name| {
            let dtype = match df.get_column(name) {
                Ok(greeners::Column::Float(_)) => "REAL",
                Ok(greeners::Column::Int(_)) => "INTEGER",
                Ok(greeners::Column::Bool(_)) => "INTEGER",
                _ => "TEXT",
            };
            Ok(format!("{} {}", quote_sqlite_identifier(name)?, dtype))
        })
        .collect::<Result<Vec<_>>>()?;

    conn.execute(&format!("DROP TABLE IF EXISTS {quoted_table}"), [])
        .map_err(|e| HayashiError::Runtime(format!("SQL error: {e}")))?;

    conn.execute(
        &format!("CREATE TABLE {quoted_table} ({})", col_defs.join(", ")),
        [],
    )
    .map_err(|e| HayashiError::Runtime(format!("SQL error: {e}")))?;

    let placeholders: Vec<&str> = vec!["?"; col_names.len()];
    let insert_sql = format!(
        "INSERT INTO {quoted_table} VALUES ({})",
        placeholders.join(", ")
    );

    let n_rows = df.n_rows();
    let tx = conn
        .transaction()
        .map_err(|e| HayashiError::Runtime(format!("transaction error: {e}")))?;

    {
        let mut stmt = tx
            .prepare(&insert_sql)
            .map_err(|e| HayashiError::Runtime(format!("prepare error: {e}")))?;

        for row in 0..n_rows {
            let vals: Vec<String> = col_names
                .iter()
                .map(|name| crate::io::dsv::col_value_at(df, name, row))
                .collect();

            let params: Vec<&dyn rusqlite::types::ToSql> = vals
                .iter()
                .map(|v| v as &dyn rusqlite::types::ToSql)
                .collect();

            stmt.execute(params.as_slice())
                .map_err(|e| HayashiError::Runtime(format!("insert error: {e}")))?;
        }
    }

    tx.commit()
        .map_err(|e| HayashiError::Runtime(format!("commit error: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::NamedTempFile;

    #[test]
    fn load_table_name_with_embedded_quote() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_string_lossy().to_string();
        let conn = Connection::open(&path).unwrap();
        conn.execute("CREATE TABLE \"quoted\"\"table\" (\"value\" INTEGER)", [])
            .unwrap();
        conn.execute("INSERT INTO \"quoted\"\"table\" VALUES (42)", [])
            .unwrap();

        let (_df, rows) = load_sqlite(&path, Some("quoted\"table"), None, None, None).unwrap();

        assert_eq!(rows, 1);
    }

    #[test]
    fn write_column_name_with_embedded_quote() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_string_lossy().to_string();
        let df = greeners::DataFrame::builder()
            .add_column("quoted\"column", vec![1.0, 2.0])
            .build()
            .unwrap();

        write_sqlite(&df, &path, "data").unwrap();

        let conn = Connection::open(&path).unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM \"data\" WHERE \"quoted\"\"column\" IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn write_table_name_with_embedded_quote() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_string_lossy().to_string();
        let df = greeners::DataFrame::builder()
            .add_column("value", vec![1.0, 2.0])
            .build()
            .unwrap();

        write_sqlite(&df, &path, "quoted\"table").unwrap();

        let conn = Connection::open(&path).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM \"quoted\"\"table\"", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn reject_nul_in_identifier() {
        let err = quote_sqlite_identifier("bad\0name").unwrap_err();

        assert!(err.to_string().contains("NUL"));
    }

    #[test]
    fn write_rejects_nul_in_table_name() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_string_lossy().to_string();
        let df = greeners::DataFrame::builder()
            .add_column("value", vec![1.0])
            .build()
            .unwrap();

        let err = write_sqlite(&df, &path, "bad\0name").unwrap_err();

        assert!(err.to_string().contains("NUL"));
    }
}
