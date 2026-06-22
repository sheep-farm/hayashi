use greeners::DataFrame;
use crate::lang::error::{HayashiError, Result};

pub fn load_dsv(path: &str, delimiter: u8) -> Result<(DataFrame, usize)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(delimiter)
        .from_path(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot read '{path}': {e}")))?;

    let headers = reader.headers()
        .map_err(|e| HayashiError::Runtime(format!("header error: {e}")))?
        .clone();

    let col_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
    let mut raw_columns: Vec<Vec<String>> = vec![Vec::new(); col_names.len()];

    for result in reader.records() {
        let record = result.map_err(|e| HayashiError::Runtime(format!("record error: {e}")))?;
        for (i, field) in record.iter().enumerate() {
            if i < raw_columns.len() {
                raw_columns[i].push(field.to_string());
            }
        }
    }

    let n_rows = raw_columns.first().map_or(0, |c| c.len());

    let mut builder = DataFrame::builder();
    for (i, name) in col_names.iter().enumerate() {
        let vals = &raw_columns[i];
        if is_numeric_column(vals) {
            let floats: Vec<f64> = vals.iter().map(|s| {
                s.parse::<f64>().unwrap_or(f64::NAN)
            }).collect();
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

pub fn write_dsv(df: &DataFrame, path: &str, delimiter: u8) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .delimiter(delimiter)
        .from_path(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot write '{path}': {e}")))?;

    let col_names = df.column_names();
    writer.write_record(&col_names)
        .map_err(|e| HayashiError::Runtime(format!("write header error: {e}")))?;

    let n_rows = df.n_rows();
    for row in 0..n_rows {
        let record: Vec<String> = col_names.iter().map(|name| {
            col_value_at(df, name, row)
        }).collect();
        writer.write_record(&record)
            .map_err(|e| HayashiError::Runtime(format!("write row error: {e}")))?;
    }

    writer.flush().map_err(|e| HayashiError::Runtime(format!("flush error: {e}")))?;
    Ok(())
}

pub(crate) fn col_value_at(df: &DataFrame, col: &str, row: usize) -> String {
    use greeners::Column;
    match df.get_column(col) {
        Ok(Column::Float(arr)) => {
            let v = arr[row];
            if v.is_nan() { String::new() } else { format!("{v}") }
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
    if vals.is_empty() { return true; }
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
