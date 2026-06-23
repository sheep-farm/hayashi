use crate::lang::error::{HayashiError, Result};
use greeners::DataFrame;
use rusqlite::Connection;
use std::collections::HashMap;

pub fn load_sqlite(
    path: &str,
    table: Option<&str>,
    query: Option<&str>,
) -> Result<(DataFrame, usize)> {
    let conn = Connection::open(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot open '{path}': {e}")))?;

    let sql = if let Some(q) = query {
        q.to_string()
    } else if let Some(t) = table {
        format!("SELECT * FROM \"{t}\"")
    } else {
        let tbl = first_table(&conn)?;
        format!("SELECT * FROM \"{tbl}\"")
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

pub fn write_sqlite(df: &greeners::DataFrame, path: &str, table: &str) -> Result<()> {
    let mut conn = Connection::open(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot open '{path}': {e}")))?;

    let col_names = df.column_names();

    let col_defs: Vec<String> = col_names
        .iter()
        .map(|name| {
            let dtype = match df.get_column(name) {
                Ok(greeners::Column::Float(_)) => "REAL",
                Ok(greeners::Column::Int(_)) => "INTEGER",
                Ok(greeners::Column::Bool(_)) => "INTEGER",
                _ => "TEXT",
            };
            format!("\"{}\" {}", name, dtype)
        })
        .collect();

    conn.execute(&format!("DROP TABLE IF EXISTS \"{}\"", table), [])
        .map_err(|e| HayashiError::Runtime(format!("SQL error: {e}")))?;

    conn.execute(
        &format!("CREATE TABLE \"{}\" ({})", table, col_defs.join(", ")),
        [],
    )
    .map_err(|e| HayashiError::Runtime(format!("SQL error: {e}")))?;

    let placeholders: Vec<&str> = vec!["?"; col_names.len()];
    let insert_sql = format!(
        "INSERT INTO \"{}\" VALUES ({})",
        table,
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
