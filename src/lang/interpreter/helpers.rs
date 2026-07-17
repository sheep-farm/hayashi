use super::*;
use greeners::Column;
use std::cmp::Ordering;
use std::sync::Arc;

// ── Conversão de tipos e valores ─────────────────────────────────────────────

/// Comparator for `f64` that treats `NaN` as greater than any finite value
/// (matching Stata's convention where missing sorts last in ascending order).
/// This avoids panics from `partial_cmp(...).unwrap()` when data contains NaN.
pub(super) fn nan_last_cmp(a: &f64, b: &f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (true, true) => Ordering::Equal,
    }
}

/// Builds a rendered diagnostic value.
pub(super) fn diag(rendered: String) -> Value {
    Value::DiagResult(Rc::new(DiagResult {
        rendered,
        fields: HashMap::new(),
    }))
}

/// Builds a rendered diagnostic value with structured fields for DAP/debug.
pub(super) fn diag_with(rendered: String, fields: HashMap<String, Value>) -> Value {
    Value::DiagResult(Rc::new(DiagResult { rendered, fields }))
}

/// Converts `Value` to boolean permissively.
pub(super) fn value_as_bool(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Int(i) => *i != 0,
        Value::Float(f) => *f != 0.0 && !f.is_nan(),
        Value::Nil => false,
        _ => true,
    }
}

/// Extracts estimated coefficients from a model result.
pub(super) fn extract_params(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::OlsResult(m) => Some(m.result.params.to_vec()),
        Value::BinaryResult(m) => Some(m.result.params.to_vec()),
        Value::PenalizedResult(m) => Some(m.params.to_vec()),
        Value::PoissonResult(r) => Some(r.params.to_vec()),
        Value::NegBinResult(r) => Some(r.params.to_vec()),
        Value::QuantileResult(r) => Some(r.params.to_vec()),
        Value::PanelResult(r) => Some(r.params.to_vec()),
        Value::TobitResult(r) => Some(r.params.to_vec()),
        _ => None,
    }
}

/// Extracts standard errors from a model result.
pub(super) fn extract_se(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::OlsResult(m) => Some(m.result.std_errors.to_vec()),
        Value::BinaryResult(m) => Some(m.result.std_errors.to_vec()),
        Value::PenalizedResult(m) => Some(m.std_errors.to_vec()),
        Value::PoissonResult(r) => Some(r.std_errors.to_vec()),
        Value::NegBinResult(r) => Some(r.std_errors.to_vec()),
        Value::QuantileResult(r) => Some(r.std_errors.to_vec()),
        Value::PanelResult(r) => Some(r.std_errors.to_vec()),
        Value::TobitResult(r) => Some(r.std_errors.to_vec()),
        _ => None,
    }
}

/// Extracts coefficient names from a model result.
pub(super) fn extract_var_names(v: &Value) -> Vec<String> {
    match v {
        Value::OlsResult(m) => m.result.variable_names.clone().unwrap_or_default(),
        Value::BinaryResult(m) => m.coef_names.clone(),
        Value::PenalizedResult(m) => m.variable_names.clone(),
        Value::PoissonResult(r) => r.variable_names.clone().unwrap_or_default(),
        Value::NegBinResult(r) => r.variable_names.clone().unwrap_or_default(),
        Value::QuantileResult(r) => r.variable_names.clone().unwrap_or_default(),
        Value::PanelResult(r) => r.variable_names.clone().unwrap_or_default(),
        Value::TobitResult(r) => r.variable_names.clone().unwrap_or_default(),
        _ => vec![],
    }
}

/// Converts `Value` to `f64`.
pub(super) fn value_as_f64(v: &Value) -> Result<f64> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        _ => Err(HayashiError::Type("expected numeric value".into())),
    }
}

// ── Avaliação de operadores ───────────────────────────────────────────────────

/// Evaluates a scalar binary operator.
pub(super) fn eval_scalar_binop(op: &BinOp, l: Value, r: Value) -> Result<Value> {
    // Comparisons (works with any comparable type)
    match op {
        BinOp::Eq => {
            let eq = match (&l, &r) {
                (Value::Nil, Value::Nil) => true,
                (Value::Nil, _) | (_, Value::Nil) => false,
                (Value::Str(a), Value::Str(b)) => a == b,
                (Value::Bool(a), Value::Bool(b)) => a == b,
                _ => {
                    let a = value_as_f64(&l)?;
                    let b = value_as_f64(&r)?;
                    (a - b).abs() < f64::EPSILON
                }
            };
            return Ok(Value::Bool(eq));
        }
        BinOp::Ne => {
            let ne = match (&l, &r) {
                (Value::Nil, Value::Nil) => false,
                (Value::Nil, _) | (_, Value::Nil) => true,
                (Value::Str(a), Value::Str(b)) => a != b,
                (Value::Bool(a), Value::Bool(b)) => a != b,
                _ => {
                    let a = value_as_f64(&l)?;
                    let b = value_as_f64(&r)?;
                    (a - b).abs() >= f64::EPSILON
                }
            };
            return Ok(Value::Bool(ne));
        }
        _ => {}
    }

    // Arithmetic and numeric comparisons
    match (&l, &r) {
        // Int × Int → Int (for Add/Sub/Mul); Div/Pow → Float
        (Value::Int(a), Value::Int(b)) => match op {
            BinOp::Add => Ok(Value::Int(a + b)),
            BinOp::Sub => Ok(Value::Int(a - b)),
            BinOp::Mul => Ok(Value::Int(a * b)),
            BinOp::Div => Ok(Value::Float(*a as f64 / *b as f64)),
            BinOp::Mod => Ok(Value::Int(a % b)),
            BinOp::Pow => Ok(Value::Float((*a as f64).powf(*b as f64))),
            BinOp::Gt => Ok(Value::Bool(a > b)),
            BinOp::Lt => Ok(Value::Bool(a < b)),
            BinOp::GtEq => Ok(Value::Bool(a >= b)),
            BinOp::LtEq => Ok(Value::Bool(a <= b)),
            BinOp::And | BinOp::Or | BinOp::Eq | BinOp::Ne | BinOp::In => unreachable!(),
        },
        // Any Float → Float
        _ => {
            // String concatenation
            if let (BinOp::Add, Value::Str(a), Value::Str(b)) = (op, &l, &r) {
                return Ok(Value::Str(format!("{a}{b}")));
            }
            let a = value_as_f64(&l)?;
            let b = value_as_f64(&r)?;
            match op {
                BinOp::Add => Ok(Value::Float(a + b)),
                BinOp::Sub => Ok(Value::Float(a - b)),
                BinOp::Mul => Ok(Value::Float(a * b)),
                BinOp::Div => Ok(Value::Float(a / b)),
                BinOp::Mod => Ok(Value::Float(a % b)),
                BinOp::Pow => Ok(Value::Float(a.powf(b))),
                BinOp::Gt => Ok(Value::Bool(a > b)),
                BinOp::Lt => Ok(Value::Bool(a < b)),
                BinOp::GtEq => Ok(Value::Bool(a >= b)),
                BinOp::LtEq => Ok(Value::Bool(a <= b)),
                BinOp::And | BinOp::Or | BinOp::Eq | BinOp::Ne | BinOp::In => unreachable!(),
            }
        }
    }
}

// ── Manipulação de DataFrame ─────────────────────────────────────────────────

/// Extracts a column as Array1<f64>; accepts Float, Int, Bool, Categorical, etc.
pub(super) fn get_col_f64(df: &DataFrame, name: &str) -> Result<ndarray::Array1<f64>> {
    let col = df
        .get_column(name)
        .map_err(|_| HayashiError::Runtime(format!("column '{name}' not found")))?;
    Ok(col.to_float())
}

/// Rebuilds X from the model's variable name list.
/// `_cons`/`const`/`Intercept` → column of 1s; others → columns from df.
pub(super) fn build_x_from_varnames(
    df: &DataFrame,
    names: &[String],
) -> Result<ndarray::Array2<f64>> {
    let n = df.n_rows();
    let k = names.len();
    let mut x = ndarray::Array2::<f64>::zeros((n, k));
    for (j, name) in names.iter().enumerate() {
        match name.as_str() {
            "_cons" | "const" | "Intercept" | "(Intercept)" => {
                x.column_mut(j).fill(1.0);
            }
            other => {
                let col = get_col_f64(df, other).map_err(|_| {
                    HayashiError::Runtime(format!(
                        "predict: column '{other}' not found in DataFrame"
                    ))
                })?;
                x.column_mut(j).assign(&col);
            }
        }
    }
    Ok(x)
}

// ── Resolução de covariância ──────────────────────────────────────────────────

/// Resolves a simple covariance option.
pub(super) fn resolve_cov(opt_val: Option<&Value>) -> Result<CovarianceType> {
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
pub(super) fn col_to_cluster_ids(df: &DataFrame, col: &str) -> Result<Vec<usize>> {
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
pub(super) fn resolve_cov_full(
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
pub(super) fn filter_df_by_mask(df: &DataFrame, mask: &[f64]) -> Result<Arc<DataFrame>> {
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
pub(super) fn sort_df_by(df: &DataFrame, col: &str) -> Result<DataFrame> {
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

// ── Extração de coeficientes e metadados de modelos ──────────────────────────

/// Generates coefficient names from the formula and observed categories.
pub(super) fn coef_names_from_formula(
    formula_ast: &Formula,
    df: &DataFrame,
    n_cols: usize,
) -> Vec<String> {
    let mut names: Vec<String> = vec!["_cons".into()];
    for term in &formula_ast.rhs {
        match term {
            RhsTerm::Categorical(e) => {
                // Para C(Var(v)) simples extraímos os níveis do df
                if let Expr::Var(v) = e.as_ref() {
                    let raw = col_to_strings(df, v).unwrap_or_default();
                    let mut unique: Vec<String> = raw
                        .into_iter()
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    sort_maybe_numeric_strings(&mut unique);
                    for val in unique.into_iter().skip(1) {
                        names.push(format!("{v}={val}"));
                    }
                } else {
                    names.push(term.display_name());
                }
            }
            other => names.push(other.display_name()),
        }
    }
    names.truncate(n_cols);
    while names.len() < n_cols {
        names.push(format!("x{}", names.len() + 1));
    }
    names
}

// ── Tabulação e tabelas de frequência ────────────────────────────────────────

/// Extracts a column as Vec<String> (for tabulate, categories, etc.).
pub(super) fn col_to_strings(df: &DataFrame, name: &str) -> Result<Vec<String>> {
    use greeners::Column;
    match df.get_column(name) {
        Ok(Column::Int(arr)) => Ok(arr.iter().map(|x| x.to_string()).collect()),
        Ok(Column::Float(arr)) => Ok(arr
            .iter()
            .map(|x| {
                if x.is_nan() {
                    ".".to_string()
                } else if x.fract() == 0.0 && x.abs() < 1e14 {
                    format!("{}", *x as i64)
                } else {
                    format!("{:.4}", x)
                }
            })
            .collect()),
        Ok(Column::String(arr)) => Ok(arr.to_vec()),
        Ok(Column::Categorical(cat)) => Ok((0..df.n_rows())
            .map(|row| cat.get_string(row).unwrap_or("").to_string())
            .collect()),
        _ => df.get_string(name).map(|arr| arr.to_vec()).map_err(|_| {
            HayashiError::Runtime(format!(
                "column '{name}' not found or has unsupported type for tabulate"
            ))
        }),
    }
}

/// Frequency table (univariate).
pub(super) fn tabulate_one(df: &DataFrame, var: &str) -> Result<DataFrame> {
    let vals = col_to_strings(df, var)?;
    let n = vals.len();

    let mut counts: HashMap<String, usize> = HashMap::new();
    for v in &vals {
        *counts.entry(v.clone()).or_insert(0) += 1;
    }

    let mut unique: Vec<String> = counts.keys().cloned().collect();
    sort_maybe_numeric_strings(&mut unique);

    let label_w = var
        .len()
        .max(12)
        .max(unique.iter().map(|s| s.len()).max().unwrap_or(0))
        + 2;
    let sep = format!("{}-{}", "-".repeat(label_w), "-".repeat(36));

    println!(
        "\n{:>lw$} | {:>10}  {:>10}  {:>10}",
        var,
        "Freq.",
        "Percent",
        "Cum.",
        lw = label_w
    );
    println!("{sep}");

    let mut value_vec = Vec::new();
    let mut freq_vec = Vec::new();
    let mut pct_vec = Vec::new();
    let mut cum_vec = Vec::new();

    let mut cum = 0.0_f64;
    for key in &unique {
        let freq = counts[key];
        let pct = freq as f64 / n as f64 * 100.0;
        cum += pct;
        println!(
            "{:>lw$} | {:>10}  {:>10.2}  {:>10.2}",
            key,
            freq,
            pct,
            cum,
            lw = label_w
        );
        value_vec.push(key.clone());
        freq_vec.push(freq);
        pct_vec.push(pct);
        cum_vec.push(cum);
    }
    println!("{sep}");
    println!(
        "{:>lw$} | {:>10}  {:>10.2}",
        "Total",
        n,
        100.0_f64,
        lw = label_w
    );
    println!();

    use indexmap::IndexMap;
    let mut columns: IndexMap<String, Column> = IndexMap::new();
    columns.insert(var.to_string(), Column::String(Array1::from(value_vec)));
    columns.insert(
        "freq".to_string(),
        Column::Int(Array1::from(
            freq_vec.iter().map(|&v| v as i64).collect::<Vec<_>>(),
        )),
    );
    columns.insert("percent".to_string(), Column::Float(Array1::from(pct_vec)));
    columns.insert(
        "cum_percent".to_string(),
        Column::Float(Array1::from(cum_vec)),
    );
    DataFrame::from_columns(columns).map_err(|e| HayashiError::Runtime(e.to_string()))
}

/// Cross-tabulation (bivariate, optional chi2).
pub(super) fn tabulate_two(
    df: &DataFrame,
    row_var: &str,
    col_var: &str,
    do_chi2: bool,
) -> Result<(DataFrame, Option<HashMap<String, Value>>)> {
    let rows = col_to_strings(df, row_var)?;
    let cols = col_to_strings(df, col_var)?;

    if rows.len() != cols.len() {
        return Err(HayashiError::Runtime(
            "columns have different lengths".into(),
        ));
    }

    let mut row_set: Vec<String> = rows
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    sort_maybe_numeric_strings(&mut row_set);
    let mut col_set: Vec<String> = cols
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    sort_maybe_numeric_strings(&mut col_set);

    let mut cell: HashMap<(String, String), usize> = HashMap::new();
    for (r, c) in rows.iter().zip(cols.iter()) {
        *cell.entry((r.clone(), c.clone())).or_insert(0) += 1;
    }

    let n = rows.len();
    let col_totals: Vec<usize> = col_set
        .iter()
        .map(|c| {
            row_set
                .iter()
                .map(|r| *cell.get(&(r.clone(), c.clone())).unwrap_or(&0))
                .sum()
        })
        .collect();
    let row_totals: Vec<usize> = row_set
        .iter()
        .map(|r| {
            col_set
                .iter()
                .map(|c| *cell.get(&(r.clone(), c.clone())).unwrap_or(&0))
                .sum()
        })
        .collect();

    let cell_w = 10usize;
    let row_lw = row_var
        .len()
        .max(12)
        .max(row_set.iter().map(|s| s.len()).max().unwrap_or(0))
        + 2;
    let col_head_w = col_set.len() * (cell_w + 1) + 1;
    let total_w = cell_w + 2;

    println!(
        "\n{:>rw$} | {:^chw$}| {:>tw$}",
        "",
        col_var,
        "Total",
        rw = row_lw,
        chw = col_head_w,
        tw = total_w
    );

    print!("{:>rw$} |", row_var, rw = row_lw);
    for cv in &col_set {
        print!(" {:>cw$}", cv, cw = cell_w);
    }
    println!(" | {:>cw$}", "Total", cw = cell_w);

    let sep = format!(
        "{}-{}-{}",
        "-".repeat(row_lw),
        "-".repeat(col_head_w),
        "-".repeat(total_w)
    );
    println!("{sep}");

    for (i, rv) in row_set.iter().enumerate() {
        print!("{:>rw$} |", rv, rw = row_lw);
        for cv in &col_set {
            let cnt = *cell.get(&(rv.clone(), cv.clone())).unwrap_or(&0);
            print!(" {:>cw$}", cnt, cw = cell_w);
        }
        println!(" | {:>cw$}", row_totals[i], cw = cell_w);
    }

    println!("{sep}");
    print!("{:>rw$} |", "Total", rw = row_lw);
    for ct in &col_totals {
        print!(" {:>cw$}", ct, cw = cell_w);
    }
    println!(" | {:>cw$}", n, cw = cell_w);
    println!();

    let mut chi2_map = None;
    if do_chi2 {
        let mut stat = 0.0_f64;
        for (i, rv) in row_set.iter().enumerate() {
            for (j, cv) in col_set.iter().enumerate() {
                let obs = *cell.get(&(rv.clone(), cv.clone())).unwrap_or(&0) as f64;
                let exp = row_totals[i] as f64 * col_totals[j] as f64 / n as f64;
                if exp > 0.0 {
                    stat += (obs - exp).powi(2) / exp;
                }
            }
        }
        let df = (row_set.len() - 1) * (col_set.len() - 1);
        let p = chi2_pvalue(stat, df as f64);
        println!("  Pearson chi2({df}) = {stat:.4}   Pr = {p:.4}");
        println!();
        let mut map = HashMap::new();
        map.insert("chi2".into(), Value::Float(stat));
        map.insert("df".into(), Value::Int(df as i64));
        map.insert("p_value".into(), Value::Float(p));
        chi2_map = Some(map);
    }

    // Build long-format DataFrame
    let mut row_vec = Vec::new();
    let mut col_vec = Vec::new();
    let mut freq_vec = Vec::new();
    let mut row_total_vec = Vec::new();
    let mut col_total_vec = Vec::new();
    for rv in &row_set {
        let rt = *row_totals
            .iter()
            .zip(row_set.iter())
            .find(|(_, r)| *r == rv)
            .map(|(t, _)| t)
            .unwrap_or(&0);
        for cv in &col_set {
            let ct = *col_totals
                .iter()
                .zip(col_set.iter())
                .find(|(_, c)| *c == cv)
                .map(|(t, _)| t)
                .unwrap_or(&0);
            let cnt = *cell.get(&(rv.clone(), cv.clone())).unwrap_or(&0);
            row_vec.push(rv.clone());
            col_vec.push(cv.clone());
            freq_vec.push(cnt as i64);
            row_total_vec.push(rt as i64);
            col_total_vec.push(ct as i64);
        }
    }

    use indexmap::IndexMap;
    let mut columns: IndexMap<String, Column> = IndexMap::new();
    columns.insert(row_var.to_string(), Column::String(Array1::from(row_vec)));
    columns.insert(col_var.to_string(), Column::String(Array1::from(col_vec)));
    columns.insert("freq".to_string(), Column::Int(Array1::from(freq_vec)));
    columns.insert(
        "row_total".to_string(),
        Column::Int(Array1::from(row_total_vec)),
    );
    columns.insert(
        "col_total".to_string(),
        Column::Int(Array1::from(col_total_vec)),
    );
    let df = DataFrame::from_columns(columns).map_err(|e| HayashiError::Runtime(e.to_string()))?;
    Ok((df, chi2_map))
}

// ── Ordenação de strings ──────────────────────────────────────────────────────

/// Converts a finite numeric string, if possible.
pub(super) fn finite_numeric_string(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Sorts strings numerically if all are finite numeric; otherwise alphabetically.
pub(super) fn sort_maybe_numeric_strings(values: &mut [String]) {
    if values
        .iter()
        .all(|value| finite_numeric_string(value).is_some())
    {
        values.sort_by(
            |a, b| match (finite_numeric_string(a), finite_numeric_string(b)) {
                (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
                _ => a.cmp(b),
            },
        );
    } else {
        values.sort();
    }
}

// ── Visualização ASCII ────────────────────────────────────────────────────────

/// ASCII histogram.
pub(super) fn ascii_histogram(data: &[f64], bins: usize, title: &str, var: &str, width: usize) {
    if data.is_empty() {
        println!("  (no data)");
        return;
    }
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-15 {
        println!("  (zero variance)");
        return;
    }
    let step = (max - min) / bins as f64;
    let mut counts = vec![0usize; bins];
    for &v in data {
        let idx = ((v - min) / step).floor() as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }
    let max_count = *counts.iter().max().unwrap_or(&1);
    let bar_w = width.max(10);
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let sd = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    println!();
    println!("{:=^width$}", format!(" {title} "), width = bar_w + 34);
    println!("  Variable: {var}   n={n}   μ={mean:.4}   σ={sd:.4}   [{min:.4}, {max:.4}]");
    println!("{:-^width$}", "", width = bar_w + 34);
    for (i, &cnt) in counts.iter().enumerate() {
        let lo = min + i as f64 * step;
        let hi = lo + step;
        let bar_len = (cnt * bar_w).checked_div(max_count).unwrap_or(0);
        let bar: String = "█".repeat(bar_len);
        println!(
            "  [{:>10.4},{:>10.4})  {:>5}  {:<width$}",
            lo,
            hi,
            cnt,
            bar,
            width = bar_w
        );
    }
    println!("{:-^width$}", "", width = bar_w + 34);
    println!();
}

pub(super) fn ascii_scatter(
    xs: &[f64],
    ys: &[f64],
    title: &str,
    xlab: &str,
    ylab: &str,
    w: usize,
    h: usize,
) {
    if xs.is_empty() {
        println!("  (no data)");
        return;
    }
    let xmin = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ymin = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let ymax = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let xrng = (xmax - xmin).max(1e-15);
    let yrng = (ymax - ymin).max(1e-15);
    let mut grid = vec![vec![' '; w]; h];
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        if x.is_nan() || y.is_nan() {
            continue;
        }
        let col = ((x - xmin) / xrng * (w - 1) as f64).round() as usize;
        let row = h - 1 - ((y - ymin) / yrng * (h - 1) as f64).round() as usize;
        let col = col.min(w - 1);
        let row = row.min(h - 1);
        grid[row][col] = '·';
    }
    println!();
    println!("{:=^width$}", format!(" {title} "), width = w + 18);
    println!("  {:<10}  {:>10.4} ┐", ylab, ymax);
    for (i, row) in grid.iter().enumerate() {
        let y_val = ymax - (i as f64 / (h - 1) as f64) * yrng;
        let prefix = if i == 0 || i == h / 2 || i == h - 1 {
            format!("  {:>10.4} │", y_val)
        } else {
            "             │".to_string()
        };
        let line: String = row.iter().collect();
        println!("{prefix}{line}");
    }
    println!("             └{}", "─".repeat(w));
    let mid_x = xmin + xrng / 2.0;
    println!(
        "              {:<10.4}{:^width$.4}{:>10.4}",
        xmin,
        mid_x,
        xmax,
        width = w - 20
    );
    println!("              {:^width$}", xlab, width = w);
    println!("  n={}", xs.len());
    println!();
}

pub(super) fn ascii_lineplot(
    xs: &[f64],
    ys: &[f64],
    title: &str,
    xlab: &str,
    ylab: &str,
    w: usize,
    h: usize,
) {
    if xs.is_empty() {
        println!("  (no data)");
        return;
    }
    let xmin = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ymin = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let ymax = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let xrng = (xmax - xmin).max(1e-15);
    let yrng = (ymax - ymin).max(1e-15);
    let mut pairs: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys.iter())
        .filter(|(&x, &y)| !x.is_nan() && !y.is_nan())
        .map(|(&x, &y)| (x, y))
        .collect();
    pairs.sort_by(|a, b| nan_last_cmp(&a.0, &b.0));
    let mut grid = vec![vec![' '; w]; h];
    let mut prev_col: Option<(usize, usize)> = None;
    for &(x, y) in &pairs {
        let col = ((x - xmin) / xrng * (w - 1) as f64).round() as usize;
        let row = h - 1 - ((y - ymin) / yrng * (h - 1) as f64).round() as usize;
        let col = col.min(w - 1);
        let row = row.min(h - 1);
        if let Some((pr, pc)) = prev_col {
            if pc < col {
                (pc..=col).for_each(|c| {
                    let t = (c - pc) as f64 / (col - pc).max(1) as f64;
                    let r =
                        ((pr as f64 + t * (row as f64 - pr as f64)).round() as usize).min(h - 1);
                    if grid[r][c] == ' ' {
                        grid[r][c] = '─';
                    }
                });
            }
        }
        grid[row][col] = '●';
        prev_col = Some((row, col));
    }
    println!();
    println!("{:=^width$}", format!(" {title} "), width = w + 18);
    println!("  {:<10}  {:>10.4} ┐", ylab, ymax);
    for (i, row) in grid.iter().enumerate() {
        let y_val = ymax - (i as f64 / (h - 1) as f64) * yrng;
        let prefix = if i == 0 || i == h / 2 || i == h - 1 {
            format!("  {:>10.4} │", y_val)
        } else {
            "             │".to_string()
        };
        let line: String = row.iter().collect();
        println!("{prefix}{line}");
    }
    println!("             └{}", "─".repeat(w));
    let mid_x = xmin + xrng / 2.0;
    println!(
        "              {:<10.4}{:^width$.4}{:>10.4}",
        xmin,
        mid_x,
        xmax,
        width = w - 20
    );
    println!("              {:^width$}", xlab, width = w);
    println!("  n={}", pairs.len());
    println!();
}

pub(super) fn ascii_boxplot(data: &[f64], title: &str, var: &str, w: usize) {
    if data.is_empty() {
        println!("  (no data)");
        return;
    }
    let mut sorted = data.to_vec();
    sorted.retain(|v| !v.is_nan());
    sorted.sort_by(nan_last_cmp);
    let n = sorted.len();
    if n < 4 {
        println!("  (too few data for boxplot)");
        return;
    }
    let q = |p: f64| -> f64 {
        let idx = p * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil().min((n - 1) as f64) as usize;
        sorted[lo] + (idx - lo as f64) * (sorted[hi] - sorted[lo])
    };
    let mn = sorted[0];
    let q1 = q(0.25);
    let med = q(0.50);
    let q3 = q(0.75);
    let mx = sorted[n - 1];
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let iqr = q3 - q1;
    let fence_lo = q1 - 1.5 * iqr;
    let fence_hi = q3 + 1.5 * iqr;
    let whisker_lo = sorted
        .iter()
        .cloned()
        .filter(|&v| v >= fence_lo)
        .fold(f64::INFINITY, f64::min);
    let whisker_hi = sorted
        .iter()
        .cloned()
        .filter(|&v| v <= fence_hi)
        .fold(f64::NEG_INFINITY, f64::max);
    let outliers: Vec<f64> = sorted
        .iter()
        .cloned()
        .filter(|&v| v < fence_lo || v > fence_hi)
        .collect();

    let rng = (mx - mn).max(1e-15);
    let to_col =
        |v: f64| -> usize { (((v - mn) / rng * (w - 1) as f64).round() as usize).min(w - 1) };
    let c_wlo = to_col(whisker_lo);
    let c_q1 = to_col(q1);
    let c_med = to_col(med);
    let c_q3 = to_col(q3);
    let c_whi = to_col(whisker_hi);

    let mut line = vec![' '; w];
    line[c_wlo..=c_whi].fill('─');
    line[c_q1..=c_q3].fill('█');
    line[c_wlo] = '├';
    line[c_whi] = '┤';
    line[c_q1] = '▐';
    line[c_q3] = '▌';
    line[c_med] = '|';
    for &v in &outliers {
        let c = to_col(v);
        line[c] = '○';
    }

    println!();
    println!("{:=^width$}", format!(" {title} "), width = w + 18);
    println!("  Variable: {var}   n={n}");
    println!();
    println!("             {}", line.iter().collect::<String>());
    println!();
    println!(
        "  Min:    {:>12.4}   Q1:  {:>12.4}   Median:  {:>12.4}",
        whisker_lo, q1, med
    );
    println!(
        "  Mean:   {:>12.4}   Q3:  {:>12.4}   Max:     {:>12.4}",
        mean, q3, whisker_hi
    );
    println!("  IQR:    {:>12.4}   Outliers: {}", iqr, outliers.len());
    if !outliers.is_empty() && outliers.len() <= 10 {
        let out_str: Vec<String> = outliers.iter().map(|v| format!("{:.3}", v)).collect();
        println!("  Values: [{}]", out_str.join(", "));
    }
    println!();
}

/// ACF / PACF as ASCII bars.
pub(super) fn ascii_acf(data: &[f64], max_lag: usize, title: &str, width: usize, partial: bool) {
    let n = data.len();
    if n < 4 {
        println!("(insufficient data for ACF)");
        return;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-15 {
        println!("(zero variance)");
        return;
    }

    let max_lag = max_lag.min(n / 2);
    let acf: Vec<f64> = (0..=max_lag)
        .map(|k| {
            let s: f64 = (0..n - k)
                .map(|i| (data[i] - mean) * (data[i + k] - mean))
                .sum();
            s / (n as f64 * var)
        })
        .collect();

    let values: Vec<f64> = if partial {
        let mut pacf = vec![0.0f64; max_lag + 1];
        pacf[0] = 1.0;
        if max_lag >= 1 {
            pacf[1] = acf[1];
        }
        let mut phi: Vec<Vec<f64>> = vec![vec![0.0; max_lag + 1]; max_lag + 1];
        phi[1][1] = acf[1];
        for k in 2..=max_lag {
            let num: f64 = acf[k] - (1..k).map(|j| phi[k - 1][j] * acf[k - j]).sum::<f64>();
            let den: f64 = 1.0 - (1..k).map(|j| phi[k - 1][j] * acf[j]).sum::<f64>();
            let phi_kk = if den.abs() < 1e-15 { 0.0 } else { num / den };
            phi[k][k] = phi_kk;
            for j in 1..k {
                phi[k][j] = phi[k - 1][j] - phi_kk * phi[k - 1][k - j];
            }
            pacf[k] = phi_kk;
        }
        pacf
    } else {
        acf.clone()
    };

    let ci = 1.96 / (n as f64).sqrt();
    println!("\n{:=<width$}", "");
    println!(" {title}");
    println!("{:=<width$}", "");
    let half = width / 2;
    for (lag, v) in values.iter().enumerate().skip(1) {
        let bar_len = ((v.abs() * half as f64).round() as usize).min(half);
        let in_ci = v.abs() <= ci;
        let bar_char = if in_ci { '─' } else { '█' };
        let bar: String = std::iter::repeat_n(bar_char, bar_len).collect();
        let (left, right) = if *v >= 0.0 {
            (format!("{:<half$}", " "), bar.to_string())
        } else {
            let pad = half - bar_len;
            (format!("{:>half$}", bar), " ".repeat(pad))
        };
        println!("{:3} |{}|{} {:6.3}", lag, left, right, v);
    }
    println!("{:=<width$}", "");
    println!("  CI ±{:.3} (95%)  │ ── inside  █ outside", ci);
    println!();
}

/// Normal QQ-plot ASCII.
pub(super) fn ascii_qqplot(data: &[f64], title: &str, var: &str, w: usize, h: usize) {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n < 4 {
        println!("(insufficient data for QQ-plot)");
        return;
    }
    let theoretical: Vec<f64> = (1..=n)
        .map(|i| {
            let p = (i as f64 - 0.375) / (n as f64 + 0.25);
            let q = p - 0.5;
            let r = if q.abs() <= 0.425 {
                let a = [
                    3.3871328_f64,
                    133.14166789,
                    1971.5909503,
                    13731.693765,
                    45921.953931,
                    67265.770927,
                    33430.575583,
                    2509.0809287,
                ];
                let b = [
                    1.0_f64,
                    42.313330701,
                    687.18700749,
                    5394.1960214,
                    21213.794301,
                    39307.895800,
                    28729.085735,
                    5226.4952788,
                ];
                let q2 = q * q;
                let num = a
                    .iter()
                    .enumerate()
                    .fold(0.0, |s, (i, &c)| s + c * q2.powi(i as i32));
                let den = b
                    .iter()
                    .enumerate()
                    .fold(0.0, |s, (i, &c)| s + c * q2.powi(i as i32));
                q * num / den
            } else {
                let pp = if q < 0.0 { p } else { 1.0 - p };
                let r = (-pp.ln()).sqrt();
                let c = if r <= 5.0 {
                    [
                        1.42343711_f64,
                        4.63033784,
                        5.76082150,
                        1.42343711,
                        1.63155402,
                        0.07027109,
                    ]
                } else {
                    [
                        6.65790464_f64,
                        5.46378491,
                        1.78482653,
                        0.05697114,
                        0.18127138,
                        0.00778070,
                    ]
                };
                let num = c[0] + r * (c[1] + r * c[2]);
                let den = 1.0 + r * (c[3] + r * (c[4] + r * c[5]));
                if q < 0.0 {
                    -(num / den)
                } else {
                    num / den
                }
            };
            r
        })
        .collect();
    let mean_s = sorted.iter().sum::<f64>() / n as f64;
    let std_s = (sorted.iter().map(|x| (x - mean_s).powi(2)).sum::<f64>() / n as f64)
        .sqrt()
        .max(1e-15);
    let empirical: Vec<f64> = sorted.iter().map(|x| (x - mean_s) / std_s).collect();
    println!("\n{:=<w$}", "");
    println!(" {title}  (normalized)");
    println!("{:=<w$}", "");
    ascii_scatter(
        &theoretical,
        &empirical,
        title,
        "theoretical quantile",
        var,
        w,
        h,
    );
    println!("  (ideal line: points along the diagonal)");
}

/// Correlation matrix as text heatmap.
pub(super) fn ascii_corrplot(cols: &[Vec<f64>], names: &[String]) {
    let n = cols[0].len();
    let means: Vec<f64> = cols
        .iter()
        .map(|c| c.iter().sum::<f64>() / n as f64)
        .collect();
    let corr: Vec<Vec<f64>> = cols
        .iter()
        .enumerate()
        .map(|(i, col_i)| {
            let xi: Vec<f64> = col_i.iter().map(|x| x - means[i]).collect();
            let di = xi.iter().map(|a| a * a).sum::<f64>().sqrt();
            cols.iter()
                .enumerate()
                .map(|(j, col_j)| {
                    let xj: Vec<f64> = col_j.iter().map(|x| x - means[j]).collect();
                    let num: f64 = xi.iter().zip(&xj).map(|(a, b)| a * b).sum();
                    let dj = xj.iter().map(|b| b * b).sum::<f64>().sqrt();
                    if di * dj < 1e-15 {
                        0.0
                    } else {
                        num / (di * dj)
                    }
                })
                .collect()
        })
        .collect();
    let nw = names.iter().map(|n| n.len()).max().unwrap_or(4).max(4);
    println!("\n{:=<80}", "");
    println!(" Correlation Matrix");
    println!("{:=<80}", "");
    print!("{:>nw$}", "");
    for n in names {
        print!(" {:>7}", &n[..n.len().min(7)]);
    }
    println!();
    for (name, row) in names.iter().zip(&corr) {
        let name_disp = &name[..name.len().min(nw)];
        print!("{:>nw$}", name_disp);
        for v in row {
            let shade = if v.abs() >= 0.9 {
                "████"
            } else if v.abs() >= 0.7 {
                "▓▓▓▓"
            } else if v.abs() >= 0.5 {
                "▒▒▒▒"
            } else if v.abs() >= 0.3 {
                "░░░░"
            } else {
                "    "
            };
            let sign = if *v < 0.0 { "-" } else { "+" };
            print!(" {sign}{shade}");
        }
        print!("   ");
        for v in row {
            print!(" {:>6.3}", v);
        }
        println!();
    }
    println!("{:=<80}", "");
    println!("  Scale: ████ |r|≥0.9  ▓▓▓▓ ≥0.7  ▒▒▒▒ ≥0.5  ░░░░ ≥0.3  (+neg=-)");
    println!();
}

// ── Funções estatísticas ──────────────────────────────────────────────────────

/// Φ(x) normal CDF — Abramowitz & Stegun 26.2.17 (error < 7.5e-8).
pub(super) fn norm_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let phi = 1.0 - greeners::norm_pdf(x) * poly;
    if x >= 0.0 {
        phi
    } else {
        1.0 - phi
    }
}
