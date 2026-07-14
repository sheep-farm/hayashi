use crate::lang::error::{HayashiError, Result};
use crate::lang::predicate::{RowAccess, RowPredicate};
use calamine::{open_workbook_auto, Data, Reader};
use greeners::DataFrame;

pub fn load_excel(
    path: &str,
    sheet: Option<&str>,
    columns: Option<&[String]>,
    predicate: Option<&RowPredicate>,
) -> Result<(DataFrame, usize)> {
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

    let all_headers: Vec<String> = header_row
        .iter()
        .map(|c| match c {
            Data::String(s) => s.clone(),
            Data::Float(f) => format!("{f}"),
            Data::Int(i) => format!("{i}"),
            _ => "unnamed".into(),
        })
        .collect();

    // Validar columns= e where=.
    let pred_cols: Vec<String> = predicate
        .map(|p| p.referenced_columns())
        .unwrap_or_default();
    for c in &pred_cols {
        if !all_headers.contains(c) {
            return Err(HayashiError::Runtime(format!(
                "load excel: where references unknown column '{c}' — available: {}",
                all_headers.join(", ")
            )));
        }
    }
    let keep_cols: Vec<String> = match columns {
        Some(cols) if !cols.is_empty() => {
            for c in cols {
                if !all_headers.contains(c) {
                    return Err(HayashiError::Runtime(format!(
                        "load excel: column '{c}' not found — available: {}",
                        all_headers.join(", ")
                    )));
                }
            }
            cols.to_vec()
        }
        _ => all_headers.clone(),
    };

    let keep_idx: Vec<usize> = keep_cols
        .iter()
        .map(|c| all_headers.iter().position(|n| n == c).unwrap())
        .collect();
    let pred_idx: Vec<usize> = pred_cols
        .iter()
        .map(|c| all_headers.iter().position(|n| n == c).unwrap())
        .collect();

    // Primeiro coletamos apenas as linhas que passam no where= (se houver).
    // calamine já materializou o range todo em RAM, então a economia aqui é
    // apenas nos acumuladores finais (não no parse da planilha).
    let pred_layout: Vec<(usize, String)> = pred_idx
        .iter()
        .copied()
        .zip(pred_cols.iter().cloned())
        .collect();
    let filtered_rows: Vec<&[Data]> = match predicate {
        Some(pred) => rows_iter
            .filter(|row| {
                let r = ExcelRow {
                    row,
                    layout: &pred_layout,
                };
                pred.evaluate(&r)
            })
            .collect(),
        None => rows_iter.collect(),
    };
    let data_rows = filtered_rows;
    let n_rows = data_rows.len();

    let mut is_numeric: Vec<bool> = vec![true; keep_cols.len()];
    for row in &data_rows {
        for (out_i, &col_idx) in keep_idx.iter().enumerate() {
            if col_idx < row.len() {
                match &row[col_idx] {
                    Data::String(_)
                    | Data::Bool(_)
                    | Data::DateTimeIso(_)
                    | Data::DurationIso(_) => {
                        is_numeric[out_i] = false;
                    }
                    _ => {}
                }
            }
        }
    }

    let mut builder = DataFrame::builder();
    for (out_i, name) in keep_cols.iter().enumerate() {
        let col_idx = keep_idx[out_i];
        if is_numeric[out_i] {
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

/// Linha de uma planilha Excel para avaliação do `where`.
struct ExcelRow<'a> {
    row: &'a [Data],
    /// (índice em `row`, nome da coluna) — só colunas referenciadas.
    layout: &'a [(usize, String)],
}

impl<'a> RowAccess for ExcelRow<'a> {
    fn get_f64(&self, col: &str) -> Option<f64> {
        let (idx, _) = self.layout.iter().find(|(_, n)| n == col)?;
        let cell = self.row.get(*idx)?;
        Some(match cell {
            Data::Float(f) => *f,
            Data::Int(i) => *i as f64,
            Data::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            Data::Empty => f64::NAN,
            // Strings: tentar parse (igual ao DSV). Falha → NaN (null).
            Data::String(s) => s.parse::<f64>().unwrap_or(f64::NAN),
            Data::DateTimeIso(s) | Data::DurationIso(s) => s.parse::<f64>().unwrap_or(f64::NAN),
            _ => f64::NAN,
        })
    }

    fn get_str(&self, col: &str) -> Option<&str> {
        let (idx, _) = self.layout.iter().find(|(_, n)| n == col)?;
        let cell = self.row.get(*idx)?;
        match cell {
            Data::String(s) => Some(s.as_str()),
            Data::DateTimeIso(s) | Data::DurationIso(s) => Some(s.as_str()),
            Data::Empty => Some(""),
            // Numéricos em coluna string: o trait exige &str com lifetime da
            // linha, mas formatar requer owned. Nesses casos retornamos None
            // e o predicado cai no caminho numérico.
            _ => None,
        }
    }
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
