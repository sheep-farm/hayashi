// ── Tabulação e tabelas de frequência ────────────────────────────────────────
use super::super::*;
use greeners::Column;

/// Extracts a column as Vec<String> (for tabulate, categories, etc.).
pub(in crate::lang::interpreter) fn col_to_strings(
    df: &DataFrame,
    name: &str,
) -> Result<Vec<String>> {
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
pub(in crate::lang::interpreter) fn tabulate_one(df: &DataFrame, var: &str) -> Result<DataFrame> {
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
pub(in crate::lang::interpreter) fn tabulate_two(
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
