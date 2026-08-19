// ── Resolução de covariância ──────────────────────────────────────────────────
use super::super::*;
use std::sync::Arc;

/// Resolves a simple covariance option.
pub(in crate::lang::interpreter) fn resolve_cov(opt_val: Option<&Value>) -> Result<CovarianceType> {
    match opt_val {
        None => Ok(CovarianceType::NonRobust),
        Some(Value::Str(s)) => match s.as_str() {
            "nonrobust" | "ols" => Ok(CovarianceType::NonRobust),
            "robust" => Ok(CovarianceType::HC1),
            "HC1" => Ok(CovarianceType::HC1),
            "HC2" => Ok(CovarianceType::HC2),
            "HC3" => Ok(CovarianceType::HC3),
            "HC4" => Ok(CovarianceType::HC4),
            other => Err(HayashiError::Type(format!(
                "unknown covariance type '{other}'"
            ))),
        },
        _ => Err(HayashiError::Type("cov= must be a string".into())),
    }
}

/// Converts an ID column (int, float, or string) into compact IDs.
pub(in crate::lang::interpreter) fn col_to_cluster_ids(
    df: &DataFrame,
    col: &str,
) -> Result<Vec<usize>> {
    let mut map: HashMap<i64, usize> = HashMap::new();
    let mut next = 0usize;
    if let Ok(arr) = df.get_int(col) {
        Ok(arr
            .iter()
            .map(|&v| {
                *map.entry(v).or_insert_with(|| {
                    let id = next;
                    next += 1;
                    id
                })
            })
            .collect())
    } else if let Ok(arr) = df.get(col) {
        Ok(arr
            .iter()
            .map(|&v| {
                let key = v as i64;
                *map.entry(key).or_insert_with(|| {
                    let id = next;
                    next += 1;
                    id
                })
            })
            .collect())
    } else if let Ok(arr) = df.get_string(col) {
        let mut smap: HashMap<String, usize> = HashMap::new();
        Ok(arr
            .iter()
            .map(|v| {
                *smap.entry(v.clone()).or_insert_with(|| {
                    let id = next;
                    next += 1;
                    id
                })
            })
            .collect())
    } else {
        Err(HayashiError::Runtime(format!(
            "cluster column '{col}' not found"
        )))
    }
}

/// Resolves the full covariance, including cluster, Newey-West, and robust.
pub(in crate::lang::interpreter) fn resolve_cov_full(
    opt_map: &HashMap<String, Value>,
    df: &DataFrame,
) -> Result<CovarianceType> {
    if let Some(Value::Str(cluster_col)) = opt_map.get("cluster") {
        let ids = col_to_cluster_ids(df, cluster_col)?;
        if let Some(Value::Str(cluster2_col)) = opt_map.get("cluster2") {
            let ids2 = col_to_cluster_ids(df, cluster2_col)?;
            Ok(CovarianceType::ClusteredTwoWay(ids, ids2))
        } else {
            Ok(CovarianceType::Clustered(ids))
        }
    } else if let Some(Value::Str(nw)) = opt_map.get("nw") {
        let lags: usize = nw
            .parse()
            .unwrap_or_else(|_| (df.n_rows() as f64).powf(0.25) as usize);
        Ok(CovarianceType::NeweyWest(lags))
    } else if let Some(Value::Int(nw)) = opt_map.get("nw") {
        Ok(CovarianceType::NeweyWest(*nw as usize))
    } else {
        resolve_cov(opt_map.get("cov"))
    }
}

/// Filters DataFrame by numeric mask (values != 0 are kept).
pub(in crate::lang::interpreter) fn filter_df_by_mask(
    df: &DataFrame,
    mask: &[f64],
) -> Result<Arc<DataFrame>> {
    let keep: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, &m)| m != 0.0)
        .map(|(i, _)| i)
        .collect();
    df.iloc(Some(&keep), None)
        .map(Arc::new)
        .map_err(|e| HayashiError::Runtime(e.to_string()))
}

/// Sorts a DataFrame by a single column (ascending).
pub(in crate::lang::interpreter) fn sort_df_by(df: &DataFrame, col: &str) -> Result<DataFrame> {
    use greeners::Column;
    let n = df.n_rows();

    let mut idx: Vec<usize> = (0..n).collect();
    match df.get_column(col) {
        Ok(Column::Float(arr)) => {
            let v = arr.to_vec();
            idx.sort_by(|&a, &b| nan_last_cmp(&v[a], &v[b]));
        }
        Ok(Column::Int(arr)) => {
            let v: Vec<f64> = arr.iter().map(|&x| x as f64).collect();
            idx.sort_by(|&a, &b| nan_last_cmp(&v[a], &v[b]));
        }
        _ => {
            if let Ok(arr) = df.get_string(col) {
                let v = arr.to_vec();
                idx.sort_by(|&a, &b| v[a].cmp(&v[b]));
            } else {
                return Err(HayashiError::Runtime(format!("column '{col}' not found")));
            }
        }
    }

    let mut builder = DataFrame::builder();
    for name in &df.column_names() {
        match df.get_column(name) {
            Ok(Column::Float(arr)) => {
                builder = builder.add_column(name, idx.iter().map(|&i| arr[i]).collect::<Vec<_>>());
            }
            Ok(Column::Int(arr)) => {
                builder = builder
                    .add_column(name, idx.iter().map(|&i| arr[i] as f64).collect::<Vec<_>>());
            }
            _ => {
                if let Ok(arr) = df.get_string(name) {
                    let v = arr.to_vec();
                    builder = builder.add_string(name, idx.iter().map(|&i| v[i].clone()).collect());
                }
            }
        }
    }
    builder
        .build()
        .map_err(|e| HayashiError::Runtime(e.to_string()))
}
