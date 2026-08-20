use crate::lang::error::{HayashiError, Result};
use crate::lang::predicate::{RowAccess, RowPredicate};
use dta::stata::dta::dta_reader::DtaReader;
use dta::stata::dta::value::Value as DtaValue;
use dta::stata::dta::variable_type::VariableType;
use greeners::DataFrame;
use std::collections::HashMap;

/// Reads a .dta file and converts it to a Greeners DataFrame.
///
/// Numeric columns (byte, int, long, float, double) are imported
/// as Float. String columns are imported as String. Stata missing
/// values are converted to NaN.
///
/// Quando `columns` ou `predicate` são fornecidos, apenas as colunas
/// pedidas (e as linhas que satisfazem o predicado) são materializadas,
/// economizando RAM em arquivos grandes.
pub fn load_dta(
    path: &str,
    columns: Option<&[String]>,
    predicate: Option<&RowPredicate>,
) -> Result<(DataFrame, usize)> {
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

    let all_names: Vec<String> = variables.iter().map(|(n, _)| n.clone()).collect();

    // Validar columns= e where=.
    let pred_cols: Vec<String> = predicate
        .map(|p| p.referenced_columns())
        .unwrap_or_default();
    for c in &pred_cols {
        if !all_names.contains(c) {
            return Err(HayashiError::Runtime(format!(
                "load dta: where references unknown column '{c}' — available: {}",
                all_names.join(", ")
            )));
        }
    }
    let keep_cols: Vec<String> = match columns {
        Some(cols) if !cols.is_empty() => {
            for c in cols {
                if !all_names.contains(c) {
                    return Err(HayashiError::Runtime(format!(
                        "load dta: column '{c}' not found — available: {}",
                        all_names.join(", ")
                    )));
                }
            }
            cols.to_vec()
        }
        _ => all_names.clone(),
    };

    // Índices (em variables) das colunas que vamos retornar e das colunas
    // referenciadas pelo predicado.
    let keep_idx: Vec<usize> = keep_cols
        .iter()
        .map(|c| all_names.iter().position(|n| n == c).unwrap())
        .collect();
    let pred_idx: Vec<usize> = pred_cols
        .iter()
        .map(|c| all_names.iter().position(|n| n == c).unwrap())
        .collect();

    // column accumulators — só para as colunas que vamos retornar.
    let mut float_cols: HashMap<String, Vec<f64>> = HashMap::new();
    let mut str_cols: HashMap<String, Vec<String>> = HashMap::new();
    let mut col_order: Vec<(String, bool)> = Vec::new(); // (name, is_numeric)

    for &i in &keep_idx {
        let (name, vtype) = &variables[i];
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
                let values = record.values();
                // where= ?
                if let Some(pred) = predicate {
                    let row = DtaRowRef {
                        values,
                        variables: &variables,
                        pred_idx: &pred_idx,
                    };
                    if !pred.evaluate(&row) {
                        continue;
                    }
                }
                n_rows += 1;
                for &i in &keep_idx {
                    let name = &variables[i].0;
                    let value = &values[i];
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

    // Build the DataFrame in the requested column order
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

/// Linha de um .dta para avaliação do `where`. Expõe apenas as colunas
/// referenciadas pelo predicado.
struct DtaRowRef<'a> {
    values: &'a [DtaValue<'a>],
    variables: &'a [(String, VariableType)],
    pred_idx: &'a [usize],
}

impl<'a> RowAccess for DtaRowRef<'a> {
    fn get_f64(&self, col: &str) -> Option<f64> {
        let pos = self
            .pred_idx
            .iter()
            .position(|i| self.variables[*i].0 == col)?;
        let i = self.pred_idx[pos];
        let v = &self.values[i];
        Some(match v {
            DtaValue::Double(d) => d.present().unwrap_or(f64::NAN),
            DtaValue::Float(f) => f.present().map(|x| x as f64).unwrap_or(f64::NAN),
            DtaValue::Long(l) => l.present().map(|x| x as f64).unwrap_or(f64::NAN),
            DtaValue::Int(iv) => iv.present().map(|x| x as f64).unwrap_or(f64::NAN),
            DtaValue::Byte(b) => b.present().map(|x| x as f64).unwrap_or(f64::NAN),
            // Strings em coluna numérica: NaN (vai ser tratado como null).
            DtaValue::String(_) | DtaValue::LongStringRef(_) => f64::NAN,
        })
    }

    fn get_str(&self, col: &str) -> Option<&str> {
        let pos = self
            .pred_idx
            .iter()
            .position(|i| self.variables[*i].0 == col)?;
        let i = self.pred_idx[pos];
        match &self.values[i] {
            DtaValue::String(s) => Some(&s[..]),
            DtaValue::LongStringRef(_) => Some(""),
            // Numéricos em coluna string: o trait exige &str; o predicado
            // cai no caminho numérico via get_f64.
            _ => None,
        }
    }
}
