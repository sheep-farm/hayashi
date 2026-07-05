use super::*;

/// Helpers puros e estáticos compartilhados pelo interpretador e seus submódulos.

/// Constrói um valor de diagnóstico renderizado.
pub(super) fn diag(rendered: String) -> Value {
    Value::DiagResult(Rc::new(DiagResult { rendered }))
}

/// Converte `Value` para booleano de forma permissiva.
pub(super) fn value_as_bool(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Int(i) => *i != 0,
        Value::Float(f) => *f != 0.0 && !f.is_nan(),
        Value::Nil => false,
        _ => true,
    }
}

/// Extrai coeficientes estimados de um resultado de modelo.
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

/// Extrai erros-padrão de um resultado de modelo.
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

/// Extrai nomes de coeficientes de um resultado de modelo.
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

/// Converte `Value` para `f64`.
pub(super) fn value_as_f64(v: &Value) -> Result<f64> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        _ => Err(HayashiError::Type("expected numeric value".into())),
    }
}

/// Avalia operador binário escalar.
pub(super) fn eval_scalar_binop(op: &BinOp, l: Value, r: Value) -> Result<Value> {
    // Comparações (funciona com qualquer tipo comparável)
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

    // Aritmética e comparações numéricas
    match (&l, &r) {
        // Int × Int → Int (para Add/Sub/Mul); Div/Pow → Float
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
        // Qualquer Float → Float
        _ => {
            // Concatenação de strings
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

/// Extrai coluna como Array1<f64>; aceita Float, Int, Bool, Categorical, etc.
pub(super) fn get_col_f64(df: &DataFrame, name: &str) -> Result<ndarray::Array1<f64>> {
    let col = df
        .get_column(name)
        .map_err(|_| HayashiError::Runtime(format!("column '{name}' not found")))?;
    Ok(col.to_float())
}

/// Reconstrói X a partir da lista de nomes de variáveis do modelo.
/// `_cons`/`const`/`Intercept` → coluna de 1s; demais → colunas do df.
pub(super) fn build_x_from_varnames(df: &DataFrame, names: &[String]) -> Result<ndarray::Array2<f64>> {
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
                        "predict: column '{other}' not found no DataFrame"
                    ))
                })?;
                x.column_mut(j).assign(&col);
            }
        }
    }
    Ok(x)
}

/// Resolve opção de covariância simples.
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

/// Converte coluna de IDs (int, float ou string) em IDs compactos.
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

/// Resolve covariância completa, incluindo cluster, Newey-West e robusta.
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

/// Filtra DataFrame por máscara numérica (valores != 0 são mantidos).
pub(super) fn filter_df_by_mask(df: &DataFrame, mask: &[f64]) -> Result<Rc<DataFrame>> {
    let keep: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, &m)| m != 0.0)
        .map(|(i, _)| i)
        .collect();
    df.iloc(Some(&keep), None)
        .map(Rc::new)
        .map_err(|e| HayashiError::Runtime(e.to_string()))
}

/// Ordena um DataFrame por uma única coluna (ascendente).
pub(super) fn sort_df_by(df: &DataFrame, col: &str) -> Result<DataFrame> {
    use greeners::Column;
    let n = df.n_rows();

    let mut idx: Vec<usize> = (0..n).collect();
    match df.get_column(col) {
        Ok(Column::Float(arr)) => {
            let v = arr.to_vec();
            idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
        }
        Ok(Column::Int(arr)) => {
            let v: Vec<f64> = arr.iter().map(|&x| x as f64).collect();
            idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
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
                builder =
                    builder.add_column(name, idx.iter().map(|&i| arr[i]).collect::<Vec<_>>());
            }
            Ok(Column::Int(arr)) => {
                builder = builder
                    .add_column(name, idx.iter().map(|&i| arr[i] as f64).collect::<Vec<_>>());
            }
            _ => {
                if let Ok(arr) = df.get_string(name) {
                    let v = arr.to_vec();
                    builder =
                        builder.add_string(name, idx.iter().map(|&i| v[i].clone()).collect());
                }
            }
        }
    }
    builder
        .build()
        .map_err(|e| HayashiError::Runtime(e.to_string()))
}

/// Gera nomes de coeficientes a partir da fórmula e das categorias observadas.
pub(super) fn coef_names_from_formula(
    formula_ast: &Formula,
    df: &DataFrame,
    n_cols: usize,
) -> Vec<String> {
    let mut names: Vec<String> = vec!["_cons".into()];
    for term in &formula_ast.rhs {
        match term {
            RhsTerm::Var(v) => names.push(v.clone()),
            RhsTerm::Transform(fn_, v) => names.push(format!("{fn_}({v})")),
            RhsTerm::Interaction(a, b) => names.push(format!("{a}:{b}")),
            RhsTerm::Categorical(v) => {
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
            }
        }
    }
    names.truncate(n_cols);
    while names.len() < n_cols {
        names.push(format!("x{}", names.len() + 1));
    }
    names
}

/// Extrai coluna como Vec<String> (para tabulate, categorias etc.).
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

/// Tabela de frequências (uni-variada).
pub(super) fn tabulate_one(df: &DataFrame, var: &str) -> Result<()> {
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
    Ok(())
}

/// Tabela cruzada (bi-variada, opcional chi2).
pub(super) fn tabulate_two(df: &DataFrame, row_var: &str, col_var: &str, do_chi2: bool) -> Result<()> {
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
    }

    Ok(())
}

/// Converte string numérica finita, se possível.
pub(super) fn finite_numeric_string(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Ordena strings numericamente se todas forem numéricas finitas; senão alfabeticamente.
pub(super) fn sort_maybe_numeric_strings(values: &mut [String]) {
    if values
        .iter()
        .all(|value| finite_numeric_string(value).is_some())
    {
        values.sort_by(|a, b| {
            match (finite_numeric_string(a), finite_numeric_string(b)) {
                (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
                _ => a.cmp(b),
            }
        });
    } else {
        values.sort();
    }
}

/// Φ(x) normal CDF — Abramowitz & Stegun 26.2.17 (erro < 7.5e-8).
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
