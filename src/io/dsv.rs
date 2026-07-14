use crate::lang::error::{HayashiError, Result};
use crate::lang::predicate::{RowAccess, RowPredicate};
use greeners::DataFrame;

pub fn load_dsv(
    path: &str,
    delimiter: u8,
    columns: Option<&[String]>,
    predicate: Option<&RowPredicate>,
) -> Result<(DataFrame, usize)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(delimiter)
        .from_path(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot read '{path}': {e}")))?;

    let headers = reader
        .headers()
        .map_err(|e| HayashiError::Runtime(format!("header error: {e}")))?
        .clone();

    let all_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();

    // Validar columns= e computar colunas referenciadas pelo where=.
    let pred_cols: Vec<String> = predicate
        .map(|p| p.referenced_columns())
        .unwrap_or_default();
    for c in &pred_cols {
        if !all_names.iter().any(|n| n == c) {
            return Err(HayashiError::Runtime(format!(
                "load: where references unknown column '{c}' — available: {}",
                all_names.join(", ")
            )));
        }
    }
    let keep_cols: Vec<String> = match columns {
        Some(cols) if !cols.is_empty() => {
            for c in cols {
                if !all_names.iter().any(|n| n == c) {
                    return Err(HayashiError::Runtime(format!(
                        "load: column '{c}' not found — available: {}",
                        all_names.join(", ")
                    )));
                }
            }
            cols.to_vec()
        }
        _ => all_names.clone(),
    };

    // Índices das colunas que vamos retornar e das colunas que o predicado lê.
    let keep_idx: Vec<usize> = keep_cols
        .iter()
        .map(|c| all_names.iter().position(|n| n == c).unwrap())
        .collect();
    let pred_idx: Vec<usize> = pred_cols
        .iter()
        .map(|c| all_names.iter().position(|n| n == c).unwrap())
        .collect();

    // raw_columns só para as colunas que vamos retornar (keep_cols).
    let mut raw_columns: Vec<Vec<String>> = vec![Vec::new(); keep_cols.len()];

    // Buffer reutilizado para avaliar o predicado contra cada linha.
    // Para cada coluna referenciada pelo predicado guardamos o índice
    // (em all_names) e o nome; o DsvRow expõe esses campos.
    let pred_layout: Vec<(usize, String)> = pred_idx
        .iter()
        .copied()
        .zip(pred_cols.iter().cloned())
        .collect();

    let mut row_buf: Vec<String> = Vec::with_capacity(all_names.len());

    for result in reader.records() {
        let record = result.map_err(|e| HayashiError::Runtime(format!("record error: {e}")))?;
        row_buf.clear();
        for field in record.iter() {
            row_buf.push(field.to_string());
        }
        // where= ?
        if let Some(pred) = predicate {
            let row = DsvRow {
                fields: &row_buf,
                layout: &pred_layout,
            };
            if !pred.evaluate(&row) {
                continue;
            }
        }
        // Projeção: só as colunas pedidas.
        for (out_i, &src_i) in keep_idx.iter().enumerate() {
            if src_i < row_buf.len() {
                raw_columns[out_i].push(row_buf[src_i].clone());
            } else {
                raw_columns[out_i].push(String::new());
            }
        }
    }

    let n_rows = raw_columns.first().map_or(0, |c| c.len());

    let mut builder = DataFrame::builder();
    for (i, name) in keep_cols.iter().enumerate() {
        let vals = &raw_columns[i];
        if is_numeric_column(vals) {
            let floats: Vec<f64> = vals
                .iter()
                .map(|s| s.parse::<f64>().unwrap_or(f64::NAN))
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

/// Linha de um CSV/TSV para avaliação do `where`.
struct DsvRow<'a> {
    fields: &'a [String],
    /// (índice em `fields`, nome da coluna) — só colunas referenciadas.
    layout: &'a [(usize, String)],
}

impl<'a> RowAccess for DsvRow<'a> {
    fn get_f64(&self, col: &str) -> Option<f64> {
        let (idx, _) = self.layout.iter().find(|(_, n)| n == col)?;
        let s = self.fields.get(*idx)?;
        if s.is_empty() {
            Some(f64::NAN)
        } else {
            Some(s.parse::<f64>().unwrap_or(f64::NAN))
        }
    }

    fn get_str(&self, col: &str) -> Option<&str> {
        let (idx, _) = self.layout.iter().find(|(_, n)| n == col)?;
        self.fields.get(*idx).map(|s| s.as_str())
    }
}

pub fn write_dsv(df: &DataFrame, path: &str, delimiter: u8) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .delimiter(delimiter)
        .from_path(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot write '{path}': {e}")))?;

    let col_names = df.column_names();
    writer
        .write_record(&col_names)
        .map_err(|e| HayashiError::Runtime(format!("write header error: {e}")))?;

    let n_rows = df.n_rows();
    for row in 0..n_rows {
        let record: Vec<String> = col_names
            .iter()
            .map(|name| col_value_at(df, name, row))
            .collect();
        writer
            .write_record(&record)
            .map_err(|e| HayashiError::Runtime(format!("write row error: {e}")))?;
    }

    writer
        .flush()
        .map_err(|e| HayashiError::Runtime(format!("flush error: {e}")))?;
    Ok(())
}

pub(crate) fn col_value_at(df: &DataFrame, col: &str, row: usize) -> String {
    use greeners::Column;
    match df.get_column(col) {
        Ok(Column::Float(arr)) => {
            let v = arr[row];
            if v.is_nan() {
                String::new()
            } else {
                format!("{v}")
            }
        }
        Ok(Column::Int(arr)) => format!("{}", arr[row]),
        Ok(Column::Bool(arr)) => format!("{}", arr[row]),
        Ok(Column::String(arr)) => arr[row].clone(),
        Ok(Column::Categorical(cat)) => cat.get_string(row).unwrap_or("").to_string(),
        Ok(Column::DateTime(arr)) => format!("{}", arr[row]),
        Err(_) => String::new(),
    }
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
