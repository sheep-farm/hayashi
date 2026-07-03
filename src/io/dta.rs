use crate::lang::error::{HayashiError, Result};
use dta::stata::dta::dta_reader::DtaReader;
use dta::stata::dta::value::Value as DtaValue;
use dta::stata::dta::variable_type::VariableType;
use greeners::DataFrame;
use std::collections::HashMap;

/// Lê um arquivo .dta e converte para DataFrame do Greeners.
///
/// Colunas numéricas (byte, int, long, float, double) são importadas
/// como Float. Colunas de string são importadas como String. Valores
/// missing do Stata são convertidos para NaN.
pub fn load_dta(path: &str) -> Result<(DataFrame, usize)> {
    let header_reader = DtaReader::default()
        .from_path(path)
        .map_err(|e| HayashiError::Runtime(format!("cannot open '{path}': {e}")))?;

    let schema_reader = header_reader
        .read_header()
        .map_err(|e| HayashiError::Runtime(format!("dta header error: {e}")))?;

    let char_reader = schema_reader
        .read_schema()
        .map_err(|e| HayashiError::Runtime(format!("dta schema error: {e}")))?;

    let mut record_reader = char_reader
        .seek_records()
        .map_err(|e| HayashiError::Runtime(format!("dta seek error: {e}")))?;

    let variables: Vec<(String, VariableType)> = record_reader
        .schema()
        .variables()
        .iter()
        .map(|v| (v.name().to_string(), v.variable_type()))
        .collect();

    let _n_vars = variables.len();

    // acumuladores por coluna
    let mut float_cols: HashMap<String, Vec<f64>> = HashMap::new();
    let mut str_cols: HashMap<String, Vec<String>> = HashMap::new();
    let mut col_order: Vec<(String, bool)> = Vec::new(); // (nome, is_numeric)

    for (name, vtype) in &variables {
        match vtype {
            VariableType::FixedString(_) | VariableType::LongString => {
                str_cols.insert(name.clone(), Vec::new());
                col_order.push((name.clone(), false));
            }
            _ => {
                float_cols.insert(name.clone(), Vec::new());
                col_order.push((name.clone(), true));
            }
        }
    }

    let mut n_rows: usize = 0;

    loop {
        match record_reader
            .read_record()
            .map_err(|e| HayashiError::Runtime(format!("dta record error: {e}")))?
        {
            None => break,
            Some(record) => {
                n_rows += 1;
                for (i, value) in record.values().iter().enumerate() {
                    let name = &variables[i].0;
                    match value {
                        DtaValue::Double(d) => {
                            let v = (*d).present().unwrap_or(f64::NAN);
                            float_cols.get_mut(name).unwrap().push(v);
                        }
                        DtaValue::Float(f) => {
                            let v = (*f).present().map(|x| x as f64).unwrap_or(f64::NAN);
                            float_cols.get_mut(name).unwrap().push(v);
                        }
                        DtaValue::Long(l) => {
                            let v = (*l).present().map(|x| x as f64).unwrap_or(f64::NAN);
                            float_cols.get_mut(name).unwrap().push(v);
                        }
                        DtaValue::Int(i_val) => {
                            let v = (*i_val).present().map(|x| x as f64).unwrap_or(f64::NAN);
                            float_cols.get_mut(name).unwrap().push(v);
                        }
                        DtaValue::Byte(b) => {
                            let v = (*b).present().map(|x| x as f64).unwrap_or(f64::NAN);
                            float_cols.get_mut(name).unwrap().push(v);
                        }
                        DtaValue::String(s) => {
                            str_cols.get_mut(name).unwrap().push(s.to_string());
                        }
                        DtaValue::LongStringRef(_) => {
                            str_cols.get_mut(name).unwrap().push(String::new());
                        }
                    }
                }
            }
        }
    }

    // Monta o DataFrame na ordem original das colunas
    let mut builder = DataFrame::builder();
    for (name, is_numeric) in &col_order {
        if *is_numeric {
            let vals = float_cols.remove(name).unwrap();
            builder = builder.add_column(name, vals);
        } else {
            let vals = str_cols.remove(name).unwrap();
            builder = builder.add_string(name, vals);
        }
    }

    let df = builder
        .build()
        .map_err(|e| HayashiError::Runtime(format!("DataFrame build error: {e}")))?;

    Ok((df, n_rows))
}
