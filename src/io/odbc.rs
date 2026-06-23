use crate::lang::error::{HayashiError, Result};
use greeners::DataFrame;
use odbc_api::{buffers::TextRowSet, ConnectionOptions, Cursor, Environment, ResultSetMetadata};

pub fn load_odbc(conn_str: &str, query: &str) -> Result<(DataFrame, usize)> {
    let env = Environment::new()
        .map_err(|e| HayashiError::Runtime(format!("ODBC environment error: {e}")))?;

    let conn = env
        .connect_with_connection_string(conn_str, ConnectionOptions::default())
        .map_err(|e| HayashiError::Runtime(format!("ODBC connection error: {e}")))?;

    let mut stmt = conn
        .execute(query, (), None)
        .map_err(|e| HayashiError::Runtime(format!("ODBC query error: {e}")))?
        .ok_or_else(|| HayashiError::Runtime("ODBC query returned no result set".into()))?;

    let n_cols = stmt
        .num_result_cols()
        .map_err(|e| HayashiError::Runtime(format!("ODBC metadata error: {e}")))?
        as usize;

    let col_names: Vec<String> = (1..=n_cols as u16)
        .map(|i| stmt.col_name(i).unwrap_or_else(|_| format!("col_{i}")))
        .collect();

    let mut all_rows: Vec<Vec<String>> = vec![Vec::new(); n_cols];
    let batch_size = 1000;

    let mut buffers = TextRowSet::for_cursor(batch_size, &mut stmt, Some(4096))
        .map_err(|e| HayashiError::Runtime(format!("ODBC buffer error: {e}")))?;

    let mut cursor = stmt
        .bind_buffer(&mut buffers)
        .map_err(|e| HayashiError::Runtime(format!("ODBC bind error: {e}")))?;

    let mut n_rows: usize = 0;
    while let Some(batch) = cursor
        .fetch()
        .map_err(|e| HayashiError::Runtime(format!("ODBC fetch error: {e}")))?
    {
        let rows_in_batch = batch.num_rows();
        for row_idx in 0..rows_in_batch {
            for col_idx in 0..n_cols {
                let val = batch
                    .at(col_idx, row_idx)
                    .map(|bytes| String::from_utf8_lossy(bytes).to_string())
                    .unwrap_or_default();
                all_rows[col_idx].push(val);
            }
        }
        n_rows += rows_in_batch;
    }

    let mut builder = DataFrame::builder();
    for (i, name) in col_names.iter().enumerate() {
        let vals = &all_rows[i];
        if is_numeric_column(vals) {
            let floats: Vec<f64> = vals
                .iter()
                .map(|s| {
                    if s.is_empty() {
                        f64::NAN
                    } else {
                        s.parse::<f64>().unwrap_or(f64::NAN)
                    }
                })
                .collect();
            builder = builder.add_column(name, floats);
        } else {
            builder = builder.add_string(name, vals.clone());
        }
    }

    let df = builder
        .build()
        .map_err(|e| HayashiError::Runtime(format!("DataFrame build error: {e}")))?;

    Ok((df, n_rows))
}

fn is_numeric_column(vals: &[String]) -> bool {
    if vals.is_empty() {
        return true;
    }
    let mut num_count = 0;
    for v in vals {
        let t = v.trim();
        if t.is_empty() || t == "NA" || t == "." || t == "NaN" {
            num_count += 1;
            continue;
        }
        if t.parse::<f64>().is_ok() {
            num_count += 1;
        }
    }
    num_count * 100 / vals.len() >= 90
}
