use crate::lang::error::{HayashiError, Result};
use crate::lang::predicate::{RowAccess, RowPredicate};
use greeners::ColumnType;
use greeners::DataFrame;
use greeners::TypeInferenceConfig;

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

pub fn load_dsv(
    path: &str,
    _delimiter: u8,
    columns: Option<&[String]>,
    predicate: Option<&RowPredicate>,
    types: Option<&[String]>,
    na: Option<&[String]>,
) -> Result<(DataFrame, usize)> {
    // Build Greeners TypeInferenceConfig from Hayashi options
    let mut config = TypeInferenceConfig::default();

    // Apply na= values
    if let Some(na_values) = na {
        config.null_values = na_values.to_vec();
    }

    // Apply types= overrides (column_name -> type)
    if let Some(type_list) = types {
        for t in type_list {
            // Format: "colname:type" or just "type" (for positional)
            if let Some((col, typ)) = t.split_once(':') {
                let col_type = match typ.trim().to_lowercase().as_str() {
                    "int" => ColumnType::Int,
                    "float" => ColumnType::Float,
                    "bool" => ColumnType::Bool,
                    "string" => ColumnType::String,
                    "categorical" => ColumnType::Categorical,
                    "datetime" | "date" => ColumnType::DateTime,
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "load: unknown type '{}' in types= — use: int, float, bool, string, categorical, datetime",
                            typ
                        )))
                    }
                };
                config.column_types.insert(col.trim().to_string(), col_type);
            } else {
                // If no colon, treat as positional list (not implemented yet)
                return Err(HayashiError::Runtime(
                    "load: types= must be in format 'colname:type' (e.g., types=[ticker:string, price:float])".into()
                ));
            }
        }
    }

    // Note: encoding is not yet supported in Greeners CSV reader
    // TODO: handle encoding option

    // If no predicate, use Greeners directly for better performance
    if predicate.is_none() {
        let df = DataFrame::from_csv_with_config(path, config, columns, Some(_delimiter), None)
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let n_rows = df.n_rows();
        return Ok((df, n_rows));
    }

    // Otherwise, use Hayashi's parser with predicate support, but apply the config for type inference
    // This is a fallback - in the future we should convert predicates to Greeners format
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(_delimiter)
        .from_path(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot read '{path}': {e}")))?;

    let headers = reader
        .headers()
        .map_err(|e| HayashiError::Runtime(format!("header error: {e}")))?
        .clone();

    let all_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();

    // Validar columns= e computar colunas referenciadas pelo where=.
    let pred_cols: Vec<String> = predicate
        .as_ref()
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
        if let Some(pred) = predicate.as_ref() {
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

    // Apply type inference config to each column
    let mut builder = DataFrame::builder();
    for (i, name) in keep_cols.iter().enumerate() {
        let vals = &raw_columns[i];

        // Check for explicit type override
        if let Some(override_type) = config.column_types.get(name) {
            // Use Greeners' type override logic
            let col = greeners::DataFrame::create_column_with_type(vals, override_type, &config);
            builder = match col.as_ref() {
                greeners::Column::Int(arr) => builder.add_int(name, arr.to_vec()),
                greeners::Column::Float(arr) => builder.add_column(name, arr.to_vec()),
                greeners::Column::Bool(arr) => builder.add_bool(name, arr.to_vec()),
                greeners::Column::String(arr) => builder.add_string(name, arr.to_vec()),
                greeners::Column::Categorical(cat) => {
                    builder.add_categorical(name, cat.to_strings())
                }
                greeners::Column::DateTime(arr) => builder.add_datetime(name, arr.to_vec()),
            };
        } else {
            // Use config-based inference
            if config.enable_float {
                let mut float_parse_ok = true;
                let mut finite_count = 0;
                let mut floats: Vec<f64> = Vec::with_capacity(vals.len());
                for s in vals {
                    let t = s.trim();
                    let is_null = config.null_values.iter().any(|nv| nv == t);
                    if is_null {
                        floats.push(f64::NAN);
                    } else if let Ok(v) = t.parse::<f64>() {
                        floats.push(v);
                        if v.is_finite() {
                            finite_count += 1;
                        }
                    } else {
                        float_parse_ok = false;
                        break;
                    }
                }
                if float_parse_ok && (!config.require_finite_for_float || finite_count > 0) {
                    builder = builder.add_column(name, floats);
                } else {
                    builder = builder.add_string(name, vals.clone());
                }
            } else {
                builder = builder.add_string(name, vals.clone());
            }
        }
    }

    let df = builder
        .build()
        .map_err(|e| HayashiError::Runtime(format!("DataFrame build error: {e}")))?;

    Ok((df, n_rows))
}

pub fn write_dsv(df: &DataFrame, path: &str, delimiter: u8) -> Result<()> {
    write_dsv_with_append(df, path, delimiter, false)
}

pub fn write_dsv_with_append(
    df: &DataFrame,
    path: &str,
    delimiter: u8,
    append: bool,
) -> Result<()> {
    use std::fs::OpenOptions;

    let file = if append {
        OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| HayashiError::Runtime(format!("cannot open '{path}' for append: {e}")))?
    } else {
        std::fs::File::create(path)
            .map_err(|e| HayashiError::Runtime(format!("cannot create '{path}': {e}")))?
    };

    let mut writer = csv::WriterBuilder::new()
        .delimiter(delimiter)
        .from_writer(file);

    let col_names = df.column_names();

    // Write headers only if not appending
    if !append {
        writer
            .write_record(&col_names)
            .map_err(|e| HayashiError::Runtime(format!("write header error: {e}")))?;
    }

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
