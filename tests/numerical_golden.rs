use std::collections::HashMap;
use std::process::Command;

const TOL: f64 = 1e-8;

#[derive(Debug)]
struct CoefRow {
    coef: f64,
    std_err: f64,
}

fn tmp_csv(name: &str) -> String {
    let mut path = std::env::temp_dir();
    path.push(format!("hayashi_{name}_{}.csv", std::process::id()));
    path.to_string_lossy().replace('\\', "/")
}

fn run_inline(src: &str) -> (bool, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_hay"))
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .expect("stdin is piped for hay subprocess")
                .write_all(src.as_bytes())?;
            child.wait_with_output()
        })
        .expect("failed to execute hay");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output.status.success(), format!("{stdout}{stderr}"))
}

fn ols_golden_script(model_expr: &str, output_path: &str) -> String {
    format!(
        r#"input df
Y X firm
10 2 1
12 3 1
8 1 1
15 5 2
11 2 2
14 4 2
9 1 3
13 4 3
end

let m = {model_expr}
export(m, "csv", "{output_path}")
"#
    )
}

fn run_model_csv(test_name: &str, model_expr: &str) -> HashMap<String, CoefRow> {
    let output_path = tmp_csv(test_name);
    let script = ols_golden_script(model_expr, &output_path);
    let (ok, out) = run_inline(&script);
    assert!(ok, "{test_name} failed:\n{out}");

    let csv = std::fs::read_to_string(&output_path)
        .unwrap_or_else(|e| panic!("{test_name}: could not read {output_path}: {e}"));
    let _ = std::fs::remove_file(&output_path);

    parse_model_csv(&csv)
}

fn parse_model_csv(csv: &str) -> HashMap<String, CoefRow> {
    let mut rows = HashMap::new();
    for (line_no, line) in csv.lines().enumerate() {
        if line_no == 0 || line.trim().is_empty() {
            continue;
        }

        let fields: Vec<_> = line.split(',').collect();
        assert!(
            fields.len() >= 3,
            "expected at least 3 CSV fields on line {}: {line}",
            line_no + 1
        );

        rows.insert(
            fields[0].to_string(),
            CoefRow {
                coef: fields[1]
                    .parse()
                    .unwrap_or_else(|e| panic!("invalid coefficient on line {}: {e}", line_no + 1)),
                std_err: fields[2].parse().unwrap_or_else(|e| {
                    panic!("invalid standard error on line {}: {e}", line_no + 1)
                }),
            },
        );
    }
    rows
}

fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "{label}: expected {expected}, got {actual}, diff {diff}, tol {tol}"
    );
}

fn row<'a>(rows: &'a HashMap<String, CoefRow>, name: &str) -> &'a CoefRow {
    rows.get(name)
        .unwrap_or_else(|| panic!("missing coefficient row '{name}'; rows: {rows:?}"))
}

#[test]
fn ols_nonrobust_golden_matches_statsmodels() {
    // Reference: statsmodels OLS with classical covariance on the in-test data.
    let rows = run_model_csv("ols_nonrobust_golden", "ols(Y ~ X, df)");

    assert_close(
        row(&rows, "const").coef,
        7.064516129032259,
        TOL,
        "const coef",
    );
    assert_close(row(&rows, "X").coef, 1.6129032258064515, TOL, "X coef");
    assert_close(
        row(&rows, "const").std_err,
        0.41394358182624,
        TOL,
        "const classical SE",
    );
    assert_close(
        row(&rows, "X").std_err,
        0.1343010322236215,
        TOL,
        "X classical SE",
    );
}

#[test]
fn ols_hc1_golden_matches_statsmodels() {
    // Reference: statsmodels OLS with cov_type="HC1" on the in-test data.
    let rows = run_model_csv("ols_hc1_golden", "ols(Y ~ X, df, cov=robust)");

    assert_close(
        row(&rows, "const").coef,
        7.064516129032259,
        TOL,
        "const coef",
    );
    assert_close(row(&rows, "X").coef, 1.6129032258064515, TOL, "X coef");
    assert_close(
        row(&rows, "const").std_err,
        0.45006215415299705,
        TOL,
        "const HC1 SE",
    );
    assert_close(
        row(&rows, "X").std_err,
        0.12733717017122537,
        TOL,
        "X HC1 SE",
    );
}

#[test]
fn ols_clustered_golden_matches_statsmodels() {
    // Reference: statsmodels OLS clustered covariance with small-sample correction.
    let rows = run_model_csv("ols_clustered_golden", "ols(Y ~ X, df, cluster=firm)");

    assert_close(
        row(&rows, "const").coef,
        7.064516129032259,
        TOL,
        "const coef",
    );
    assert_close(row(&rows, "X").coef, 1.6129032258064515, TOL, "X coef");
    assert_close(
        row(&rows, "const").std_err,
        0.5869675263055752,
        TOL,
        "const clustered SE",
    );
    assert_close(
        row(&rows, "X").std_err,
        0.160766242643499,
        TOL,
        "X clustered SE",
    );
}
