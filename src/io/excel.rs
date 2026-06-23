use crate::lang::error::{HayashiError, Result};
use calamine::{open_workbook_auto, Data, Reader};
use greeners::DataFrame;

pub fn load_excel(path: &str, sheet: Option<&str>) -> Result<(DataFrame, usize)> {
    let mut workbook = open_workbook_auto(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot open '{path}': {e}")))?;

    let sheet_names = workbook.sheet_names().to_vec();
    let sheet_name = match sheet {
        Some(s) => {
            if !sheet_names.contains(&s.to_string()) {
                return Err(HayashiError::Runtime(format!(
                    "sheet '{s}' not found — available: {}",
                    sheet_names.join(", ")
                )));
            }
            s.to_string()
        }
        None => sheet_names
            .first()
            .ok_or_else(|| HayashiError::Runtime("workbook has no sheets".into()))?
            .clone(),
    };

    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|e| HayashiError::Runtime(format!("cannot read sheet '{sheet_name}': {e}")))?;

    let mut rows_iter = range.rows();
    let header_row = rows_iter
        .next()
        .ok_or_else(|| HayashiError::Runtime("sheet is empty".into()))?;

    let headers: Vec<String> = header_row
        .iter()
        .map(|c| match c {
            Data::String(s) => s.clone(),
            Data::Float(f) => format!("{f}"),
            Data::Int(i) => format!("{i}"),
            _ => "unnamed".into(),
        })
        .collect();

    let data_rows: Vec<&[Data]> = rows_iter.collect();
    let n_rows = data_rows.len();

    let mut is_numeric: Vec<bool> = vec![true; headers.len()];
    for row in &data_rows {
        for (col_idx, cell) in row.iter().enumerate() {
            if col_idx < headers.len() {
                match cell {
                    Data::String(_)
                    | Data::Bool(_)
                    | Data::DateTimeIso(_)
                    | Data::DurationIso(_) => {
                        is_numeric[col_idx] = false;
                    }
                    _ => {}
                }
            }
        }
    }

    let mut builder = DataFrame::builder();
    for (col_idx, name) in headers.iter().enumerate() {
        if is_numeric[col_idx] {
            let vals: Vec<f64> = data_rows
                .iter()
                .map(|row| {
                    if col_idx < row.len() {
                        match &row[col_idx] {
                            Data::Float(f) => *f,
                            Data::Int(i) => *i as f64,
                            Data::Empty => f64::NAN,
                            _ => f64::NAN,
                        }
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            builder = builder.add_column(name, vals);
        } else {
            let vals: Vec<String> = data_rows
                .iter()
                .map(|row| {
                    if col_idx < row.len() {
                        match &row[col_idx] {
                            Data::String(s) => s.clone(),
                            Data::Float(f) => format!("{f}"),
                            Data::Int(i) => format!("{i}"),
                            Data::Bool(b) => format!("{b}"),
                            Data::DateTimeIso(s) => s.clone(),
                            Data::DurationIso(s) => s.clone(),
                            Data::Empty => String::new(),
                            _ => String::new(),
                        }
                    } else {
                        String::new()
                    }
                })
                .collect();
            builder = builder.add_string(name, vals);
        }
    }

    let df = builder
        .build()
        .map_err(|e| HayashiError::Runtime(format!("DataFrame build error: {e}")))?;

    Ok((df, n_rows))
}

pub fn write_excel(df: &DataFrame, path: &str) -> Result<()> {
    use rust_xlsxwriter::{Format, Workbook};

    let mut workbook = Workbook::new();
    let worksheet = workbook.add_worksheet();

    let col_names = df.column_names();
    let bold = Format::new().set_bold();

    for (c, name) in col_names.iter().enumerate() {
        worksheet
            .write_string_with_format(0, c as u16, name, &bold)
            .map_err(|e| HayashiError::Runtime(format!("xlsx write error: {e}")))?;
    }

    let n_rows = df.n_rows();
    for row in 0..n_rows {
        for (c, name) in col_names.iter().enumerate() {
            let val = crate::io::dsv::col_value_at(df, name, row);
            let r = (row + 1) as u32;
            let col = c as u16;
            if let Ok(num) = val.parse::<f64>() {
                worksheet
                    .write_number(r, col, num)
                    .map_err(|e| HayashiError::Runtime(format!("xlsx write error: {e}")))?;
            } else {
                worksheet
                    .write_string(r, col, &val)
                    .map_err(|e| HayashiError::Runtime(format!("xlsx write error: {e}")))?;
            }
        }
    }

    workbook
        .save(path)
        .map_err(|e| HayashiError::Runtime(format!("xlsx save error: {e}")))?;

    Ok(())
}
