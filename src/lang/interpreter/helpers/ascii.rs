// ── Visualização ASCII ────────────────────────────────────────────────────────
use super::super::*;

/// ASCII histogram.
pub(in crate::lang::interpreter) fn ascii_histogram(
    data: &[f64],
    bins: usize,
    title: &str,
    var: &str,
    width: usize,
) {
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

pub(in crate::lang::interpreter) fn ascii_scatter(
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

pub(in crate::lang::interpreter) fn ascii_lineplot(
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

pub(in crate::lang::interpreter) fn ascii_boxplot(data: &[f64], title: &str, var: &str, w: usize) {
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
pub(in crate::lang::interpreter) fn ascii_acf(
    data: &[f64],
    max_lag: usize,
    title: &str,
    width: usize,
    partial: bool,
) {
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
pub(in crate::lang::interpreter) fn ascii_qqplot(
    data: &[f64],
    title: &str,
    var: &str,
    w: usize,
    h: usize,
) {
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
pub(in crate::lang::interpreter) fn ascii_corrplot(cols: &[Vec<f64>], names: &[String]) {
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
