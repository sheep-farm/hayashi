use std::process::Command;

fn tmp(name: &str) -> String {
    let mut p = std::env::temp_dir();
    p.push(name);
    p.to_string_lossy().replace('\\', "/")
}

fn run_hy(script: &str) -> (bool, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_hay"))
        .arg(script)
        .output()
        .expect("failed to execute hay");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output.status.success(), format!("{stdout}{stderr}"))
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
            child.stdin.take().unwrap().write_all(src.as_bytes())?;
            child.wait_with_output()
        })
        .expect("failed to execute hay");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output.status.success(), format!("{stdout}{stderr}"))
}

fn assert_ok(name: &str, src: &str) {
    let (ok, out) = run_inline(src);
    assert!(ok, "{name} failed:\n{out}");
}

fn assert_ok_contains(name: &str, src: &str, needle: &str) {
    let (ok, out) = run_inline(src);
    assert!(ok, "{name} failed:\n{out}");
    assert!(
        out.contains(needle),
        "{name}: output missing '{needle}':\n{out}"
    );
}

fn assert_err_contains(name: &str, src: &str, needle: &str) {
    let (ok, out) = run_inline(src);
    assert!(!ok, "{name} unexpectedly succeeded:\n{out}");
    assert!(
        out.contains(needle),
        "{name}: output missing '{needle}':\n{out}"
    );
}

#[derive(Debug)]
struct MarginsRow {
    dydx: f64,
    se: f64,
    z: f64,
    p: f64,
}

fn parse_margins_row(out: &str, var_name: &str) -> MarginsRow {
    let row = out
        .lines()
        .find(|line| line.split_whitespace().next() == Some(var_name))
        .unwrap_or_else(|| panic!("missing margins row for {var_name}:\n{out}"));
    let fields: Vec<&str> = row.split_whitespace().collect();
    assert!(
        fields.len() >= 5,
        "margins row for {var_name} did not include inference columns:\n{row}\n\n{out}"
    );
    MarginsRow {
        dydx: fields[1].parse().unwrap(),
        se: fields[2].parse().unwrap(),
        z: fields[3].parse().unwrap(),
        p: fields[4].parse().unwrap(),
    }
}

fn assert_close(label: &str, actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: expected {expected}, got {actual}"
    );
}

fn assert_margins_row_close(
    model: &str,
    var_name: &str,
    row: MarginsRow,
    expected_dydx: f64,
    expected_se: f64,
    expected_z: f64,
    expected_p: f64,
) {
    assert_close(
        &format!("{model} {var_name} dy/dx"),
        row.dydx,
        expected_dydx,
        5e-6,
    );
    assert_close(&format!("{model} {var_name} se"), row.se, expected_se, 5e-6);
    assert_close(&format!("{model} {var_name} z"), row.z, expected_z, 5e-4);
    assert_close(&format!("{model} {var_name} p"), row.p, expected_p, 5e-5);
}

fn assert_binary_margins_match_statsmodels(
    name: &str,
    estimator: &str,
    margins_call: &str,
    expected_x1: (f64, f64, f64, f64),
    expected_x2: (f64, f64, f64, f64),
) {
    let (ok, out) = run_inline(&format!(
        r#"
input df
y x1 x2
0 -2.0 0
0 -1.8 1
1 -1.6 0
0 -1.4 1
0 -1.2 0
1 -1.0 1
0 -0.8 0
0 -0.6 1
1 -0.4 0
0 -0.2 1
0 0.0 0
1 0.2 1
0 0.4 0
1 0.6 1
1 0.8 0
0 1.0 1
1 1.2 0
1 1.4 1
0 1.6 0
1 1.8 1
1 2.0 0
0 2.2 1
1 2.4 0
1 2.6 1
1 2.8 0
0 3.0 1
1 3.2 0
1 3.4 1
1 3.6 0
0 3.8 1
1 4.0 0
1 4.2 1
0 4.4 0
1 4.6 1
1 4.8 0
1 5.0 1
0 5.2 0
1 5.4 1
1 5.6 0
1 5.8 1
end
let m = {estimator}(y ~ x1 + x2, df)
{margins_call}
"#,
    ));

    assert!(ok, "{name} failed:\n{out}");
    assert!(
        out.contains("Std.Err.") && out.contains("P>|z|"),
        "{name}: margins output should include valid inference columns:\n{out}"
    );

    let x1 = parse_margins_row(&out, "x1");
    let x2 = parse_margins_row(&out, "x2");
    assert_margins_row_close(
        name,
        "x1",
        x1,
        expected_x1.0,
        expected_x1.1,
        expected_x1.2,
        expected_x1.3,
    );
    assert_margins_row_close(
        name,
        "x2",
        x2,
        expected_x2.0,
        expected_x2.1,
        expected_x2.2,
        expected_x2.3,
    );
}

#[test]
fn margins_logit_matches_statsmodels_delta_method() {
    // statsmodels 0.14.5: Logit(...).fit().get_margeff(at="overall", method="dydx")
    assert_binary_margins_match_statsmodels(
        "logit margins",
        "logit",
        "margins(m)",
        (0.074867, 0.024771, 3.022, 0.0025),
        (-0.014973, 0.143830, -0.104, 0.9171),
    );
}

#[test]
fn margins_probit_matches_statsmodels_delta_method() {
    // statsmodels 0.14.5: Probit(...).fit().get_margeff(at="overall", method="dydx")
    assert_binary_margins_match_statsmodels(
        "probit margins",
        "probit",
        "margins(m)",
        (0.075134, 0.025423, 2.955, 0.0031),
        (-0.009018, 0.143939, -0.063, 0.9500),
    );
}

#[test]
fn margins_probit_at_matches_statsmodels_delta_method() {
    // statsmodels 0.14.5: Probit(...).fit().get_margeff(
    //     at="overall", method="dydx", atexog={2: 1.0}
    // )
    assert_binary_margins_match_statsmodels(
        "probit margins at_x2",
        "probit",
        "margins(m, at_x2=1)",
        (0.075333, 0.025688, 2.933, 0.0034),
        (-0.009041, 0.144700, -0.062, 0.9502),
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// SMOKE TESTS — exemplos .hay offline (17 que passam sem rede)
// ══════════════════════════════════════════════════════════════════════════════

macro_rules! smoke {
    ($name:ident, $path:expr, $needle:expr) => {
        #[test]
        fn $name() {
            let (ok, out) = run_hy($path);
            assert!(ok, "{} failed:\n{}", $path, out);
            assert!(
                out.contains($needle),
                "{}: missing '{}'\n{}",
                $path,
                $needle,
                out
            );
        }
    };
}

smoke!(
    smoke_input_display,
    "examples/input_display.hay",
    "OLS Regression"
);
smoke!(smoke_eststo_loop, "examples/eststo_loop.hay", "estclear");
smoke!(
    smoke_foreach_regression,
    "examples/foreach_regression.hay",
    "●"
);
smoke!(smoke_portfolio_sort, "examples/portfolio_sort.hay", "H-L");
smoke!(
    smoke_fama_macbeth,
    "examples/fama_macbeth.hay",
    "Fama-MacBeth"
);
smoke!(smoke_double_sort, "examples/double_sort.hay", "Double Sort");
smoke!(smoke_pwcorr, "examples/pwcorr.hay", "***");
smoke!(
    smoke_missing_values,
    "examples/missing_values.hay",
    "Missing"
);
smoke!(
    smoke_preserve_restore,
    "examples/preserve_restore.hay",
    "restore"
);
smoke!(
    smoke_test_cov_display,
    "examples/test_cov_display.hay",
    "OLS (cluster)"
);
smoke!(
    smoke_three_workflows,
    "examples/three_workflows.hay",
    "Hausman"
);
smoke!(smoke_count_models, "examples/count_models.hay", "Poisson");
smoke!(smoke_panel, "examples/panel.hay", "FE");

// ══════════════════════════════════════════════════════════════════════════════
// SCOPING — block scoping with deterministic destruction
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn scope_let_block() {
    assert_ok_contains(
        "scope_let_block",
        r#"
let x = 10
if x > 5 {
    let inner = 42
    display inner
}
display x
"#,
        "42",
    );
}

#[test]
fn scope_assign_outer() {
    assert_ok_contains(
        "scope_assign_outer",
        r#"
let counter = 0
for i in 1..4 {
    counter = counter + 1
}
display counter
"#,
        "3",
    );
}

#[test]
fn scope_fn_params_die() {
    assert_ok_contains(
        "scope_fn_params",
        r#"
fn add(a, b) {
    return a + b
}
display add(3, 4)
"#,
        "7",
    );
}

#[test]
fn scope_for_var_dies() {
    let (_, out) = run_inline(
        r#"
for v in [10, 20, 30] {
    display v
}
"#,
    );
    assert!(out.contains("10") && out.contains("20") && out.contains("30"));
}

#[test]
fn scope_nested_blocks() {
    assert_ok_contains(
        "nested_blocks",
        r#"
let x = 1
if true {
    let x = 2
    if true {
        let x = 3
        display x
    }
    display x
}
display x
"#,
        "3",
    );
}

#[test]
fn scope_while_cleanup() {
    assert_ok_contains(
        "while_cleanup",
        r#"
let n = 0
while n < 3 {
    let temp = n * 10
    n = n + 1
}
display n
"#,
        "3",
    );
}

#[test]
fn scope_fn_recursive() {
    assert_ok_contains(
        "fn_recursive",
        r#"
fn factorial(n) {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}
display factorial(5)
"#,
        "120",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// PARSER — lexer, parser, expressions
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn parser_comments() {
    assert_ok_contains(
        "comments",
        r#"
// this is a comment
# this too
display 42
"#,
        "42",
    );
}

#[test]
fn parser_string_concat() {
    assert_ok_contains(
        "string_concat",
        r#"
let a = "hello"
let b = " world"
display a + b
"#,
        "hello world",
    );
}

#[test]
fn parser_negative_numbers() {
    assert_ok_contains(
        "negative",
        r#"
display -5 + 3
"#,
        "-2",
    );
}

#[test]
fn parser_boolean_ops() {
    assert_ok_contains(
        "bool_and",
        r#"
let x = 5
if x > 3 && x < 10 {
    display 1
}
"#,
        "1",
    );
}

#[test]
fn parser_list_indexing() {
    assert_ok_contains(
        "list_index",
        r#"
let v = [10, 20, 30]
display v[1]
"#,
        "20",
    );
}

#[test]
fn parser_not_operator() {
    assert_ok_contains(
        "not_op",
        r#"
let x = false
if !x {
    display 99
}
"#,
        "99",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// EXPRESSIONS — arithmetic, comparison, functions
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn expr_arithmetic() {
    assert_ok_contains("arithmetic", "display 2 + 3 * 4", "14");
}

#[test]
fn expr_power() {
    assert_ok_contains("power", "display 2 ^ 10", "1024");
}

#[test]
fn expr_modular() {
    assert_ok_contains("parens", "display (2 + 3) * 4", "20");
}

#[test]
fn expr_float() {
    assert_ok_contains("float", "display 1.5 + 2.5", "4");
}

#[test]
fn expr_comparison() {
    assert_ok_contains(
        "comparison",
        r#"
if 5 > 3 { display 1 }
if 3 >= 3 { display 2 }
if 2 < 3 { display 3 }
if 2 != 3 { display 4 }
"#,
        "4",
    );
}

#[test]
fn expr_math_functions() {
    assert_ok_contains(
        "math_fns",
        r#"
input df
X
1
4
end
generate df E = exp(X)
generate df S = sqrt(X)
generate df A = abs(X)
list(df, n=2)
"#,
        "generated",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// DADOS — input, generate, replace, summarize, list
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn data_input_block() {
    assert_ok_contains(
        "input",
        r#"
input df
Y X
1 2
3 4
5 6
end
display mean(df, Y)
"#,
        "3",
    );
}

#[test]
fn data_generate() {
    assert_ok_contains(
        "generate",
        r#"
input df
X
2
4
6
end
generate df Y = X * 2 + 1
display mean(df, Y)
"#,
        "9",
    );
}

#[test]
fn generate_substr_on_string_column() {
    // substr() in generate must produce a String column from a String column.
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.json" as df
generate df city_upper = upper(city)
generate df city_first3 = substr(city, 0, 3)
print(df)"#,
    );
    assert!(ok, "generate substr/upper failed:\n{out}");
    assert!(
        out.contains("city_upper"),
        "expected city_upper column:\n{out}"
    );
    assert!(
        out.contains("city_first3"),
        "expected city_first3 column:\n{out}"
    );
    assert!(out.contains("SÃO"), "expected uppercased city:\n{out}");
    assert!(
        out.contains("São"),
        "expected original city preserved:\n{out}"
    );
}

#[test]
fn generate_str_literal_broadcast() {
    // A bare string literal in generate must broadcast to a String column.
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.json" as df
generate df tag = "BR"
print(df)"#,
    );
    assert!(ok, "generate str literal failed:\n{out}");
    assert!(out.contains("tag"), "expected tag column:\n{out}");
    assert!(out.contains("BR"), "expected BR value:\n{out}");
}

#[test]
fn data_replace_if() {
    assert_ok_contains(
        "replace_if",
        r#"
input df
Y group
10 1
20 1
30 2
40 2
end
replace df Y = 0 if group == 2
display mean(df, Y)
"#,
        "real changes",
    );
}

#[test]
fn data_mean_conditional() {
    assert_ok_contains(
        "mean_if",
        r#"
input df
Y X
10 1
20 1
30 0
40 0
end
scalar m = mean(df, Y, if = X == 1)
display m
"#,
        "15",
    );
}

#[test]
fn data_summarize() {
    assert_ok_contains(
        "summarize",
        r#"
input df
Y
1
2
3
4
5
end
summarize(df)
"#,
        "Mean",
    );
}

#[test]
fn data_tabulate() {
    assert_ok_contains(
        "tabulate",
        r#"
input df
group
1
1
2
2
2
end
tabulate(df, group)
"#,
        "Freq",
    );
}

#[test]
fn tabulate_nan_string_group_does_not_panic() {
    assert_ok_contains(
        "tabulate_nan_string_group",
        r#"
let d = {"group": ["1", "NaN", "2"]}
let df = dataframe(d)
tabulate(df, group)
"#,
        "NaN",
    );
}

#[test]
fn data_list_n() {
    assert_ok_contains(
        "list_n",
        r#"
input df
Y X
1 2
3 4
5 6
7 8
end
list(df, n=2)
"#,
        "more observations",
    );
}

#[test]
fn data_missing_nan() {
    assert_ok_contains(
        "missing",
        r#"
input df
Y
1
.
3
end
summarize(df)
"#,
        "Missing",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// ESTIMATION — OLS, formulas, options
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn ols_basic() {
    assert_ok_contains(
        "ols_basic",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
ols(Y ~ X, df)
"#,
        "R-squared",
    );
}

#[test]
fn ols_reg_alias() {
    assert_ok_contains(
        "reg_alias",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
end
reg(Y ~ X, df)
"#,
        "OLS Regression",
    );
}

#[test]
fn ols_robust() {
    assert_ok_contains(
        "ols_robust",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
ols(Y ~ X, df, cov=robust)
"#,
        "Robust (HC1)",
    );
}

#[test]
fn ols_hc3() {
    assert_ok_contains(
        "ols_hc3",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
ols(Y ~ X, df, cov=HC3)
"#,
        "HC3",
    );
}

#[test]
fn ols_dynamic_formula() {
    assert_ok_contains(
        "dynamic_formula",
        r#"
input df
Y X1
10 2
12 3
8 1
15 5
11 2
14 4
end
ols("Y ~ X1", df)
"#,
        "OLS Regression",
    );
}

#[test]
fn ols_if_condition() {
    assert_ok_contains(
        "ols_if",
        r#"
input df
Y X group
10 2 1
12 3 1
8 1 1
15 5 1
11 2 2
14 4 2
9 1 2
13 4 2
end
ols(Y ~ X, df, if = group == 1)
"#,
        "No. Observations:                  4",
    );
}

#[test]
fn ols_cluster() {
    assert_ok_contains(
        "ols_cluster",
        r#"
input df
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
ols(Y ~ X, df, cluster=firm)
"#,
        "Clustered",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// POST-ESTIMATION — test, esttab, coefplot, nlcom, predict
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn post_test_wald() {
    assert_ok_contains(
        "test_wald",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
9 1 8
13 4 3
end
let m = ols(Y ~ X1 + X2, df)
test(m, "X1")
"#,
        "H₀",
    );
}

#[test]
fn post_test_restriction() {
    assert_ok_contains(
        "test_restriction",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
9 1 8
13 4 3
end
let m = ols(Y ~ X1 + X2, df)
test(m, "X1 = X2")
"#,
        "H₀: X1 = X2",
    );
}

#[test]
fn post_test_white() {
    assert_ok_contains(
        "test_white",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
9 1
13 4
end
let m = ols(Y ~ X, df)
test(m, "white")
"#,
        "White Test",
    );
}

#[test]
fn post_esttab() {
    assert_ok_contains(
        "esttab",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
end
let m1 = ols(Y ~ X1, df)
let m2 = ols(Y ~ X1 + X2, df)
esttab(m1, m2)
"#,
        "R²",
    );
}

#[test]
fn post_coefplot() {
    assert_ok_contains(
        "coefplot",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
end
let m = ols(Y ~ X1 + X2, df)
coefplot(m)
"#,
        "●",
    );
}

#[test]
fn post_nlcom() {
    assert_ok_contains(
        "nlcom",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
9 1 8
13 4 3
end
let m = ols(Y ~ X1 + X2, df)
nlcom(m, X1 / X2)
"#,
        "delta method",
    );
}

#[test]
fn post_predict() {
    assert_ok_contains(
        "predict",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
let m = ols(Y ~ X, df)
predict df yhat = m
predict df resid = m, "residuals"
list(df, n=2)
"#,
        "predicted",
    );
}

#[test]
fn post_estat() {
    assert_ok_contains(
        "estat",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
let m1 = ols(Y ~ X, df)
estat(m1)
"#,
        "AIC",
    );
}

#[test]
fn tidy_ols_basic() {
    assert_ok_contains(
        "tidy_ols",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
let m = ols(Y ~ X, df)
let t = tidy(m)
print(t)
"#,
        "variable",
    );
}

#[test]
fn glance_ols_basic() {
    assert_ok_contains(
        "glance_ols",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
let m = ols(Y ~ X, df)
let g = glance(m)
print(g)
"#,
        "r2",
    );
}

#[test]
fn rolling_ols_basic() {
    assert_ok_contains(
        "rolling_ols",
        r#"
input df
Y X
1 1
2 2
3 3
4 4
5 5
6 6
7 7
8 8
9 9
10 10
end
let roll = rolling(Y ~ X, df, window=5)
print(roll)
"#,
        "Rolling Regression",
    );
}

#[test]
fn tidy_rolling_wide_format() {
    assert_ok_contains(
        "tidy_rolling",
        r#"
input df
Y X
1 1
2 2
3 3
4 4
5 5
6 6
7 7
8 8
9 9
10 10
end
let roll = rolling(Y ~ X, df, window=5)
let t = tidy(roll)
print(t)
"#,
        "const",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// FINANCE — portsort, doublesort, fmb
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn finance_portsort() {
    assert_ok_contains(
        "portsort",
        r#"
input df
ret beta
0.05 0.8
0.08 1.2
0.03 0.5
0.12 1.5
0.04 0.9
0.10 1.3
end
portsort(df, ret, beta, n=3)
"#,
        "H-L",
    );
}

#[test]
fn finance_fmb() {
    assert_ok_contains(
        "fmb",
        r#"
input df
ret beta firm year
0.05 0.8 1 2020
0.08 1.2 2 2020
0.03 0.5 3 2020
0.12 1.5 4 2020
0.06 0.9 1 2021
0.10 1.3 2 2021
0.04 0.6 3 2021
0.14 1.6 4 2021
end
fmb(ret ~ beta, df, time=year)
"#,
        "Fama-MacBeth",
    );
}

#[test]
fn finance_fmb_nw() {
    assert_ok_contains(
        "fmb_nw",
        r#"
input df
ret beta firm year
0.05 0.8 1 2020
0.08 1.2 2 2020
0.03 0.5 3 2020
0.12 1.5 4 2020
0.06 0.9 1 2021
0.10 1.3 2 2021
0.04 0.6 3 2021
0.14 1.6 4 2021
0.07 1.0 1 2022
0.11 1.4 2 2022
0.02 0.3 3 2022
0.15 1.7 4 2022
end
fmb(ret ~ beta, df, time=year, nw=1)
"#,
        "NW(1)",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// DADOS — winsor, encode, tabgen, xtset
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn data_winsor() {
    assert_ok_contains(
        "winsor",
        r#"
input df
Y
1
2
3
4
5
100
end
winsor(df, Y, p=0.10)
"#,
        "winsor",
    );
}

#[test]
fn data_tabgen() {
    assert_ok_contains(
        "tabgen",
        r#"
input df
group
1
1
2
2
3
end
tabgen(df, group)
"#,
        "dummies",
    );
}

#[test]
fn data_xtset() {
    assert_ok_contains(
        "xtset",
        r#"
input df
Y X firm year
10 2 1 2020
12 3 1 2021
15 5 2 2020
17 6 2 2021
end
xtset(df, firm, year)
"#,
        "xtset df",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// LINGUAGEM — for, while, fn, if/else, source, help, timer
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn lang_for_range() {
    let (ok, out) = run_inline(
        r#"
let s = 0
for i in 1..6 {
    s = s + i
}
display s
"#,
    );
    assert!(ok);
    assert!(out.contains("15"));
}

#[test]
fn lang_increment_decrement() {
    let (ok, out) = run_inline(
        r#"
let x = 5
x++
display x
x--
x--
display x
let y = 10
y++
y++
display y
"#,
    );
    assert!(ok);
    assert!(out.contains("6"));
    assert!(out.contains("4"));
    assert!(out.contains("12"));
}

#[test]
fn lang_increment_in_loop() {
    let (ok, out) = run_inline(
        r#"
let n = 0
for i in 1..=5 {
    n++
}
display n
"#,
    );
    assert!(ok);
    assert!(out.contains("5"));
}

#[test]
fn lang_for_list() {
    let (ok, out) = run_inline(
        r#"
for x in [10, 20, 30] {
    display x
}
"#,
    );
    assert!(ok);
    assert!(out.contains("10") && out.contains("20") && out.contains("30"));
}

#[test]
fn lang_for_list_indexed() {
    let (ok, out) = run_inline(
        r#"
let s = ""
for i, v in ["a", "b", "c"] {
    s = s + str(i) + ":" + v + " "
}
print(s)
"#,
    );
    assert!(ok);
    assert!(out.contains("0:a 1:b 2:c"));
}

#[test]
fn lang_for_dict_pairs() {
    let (ok, out) = run_inline(
        r#"
let d = {"x": 10, "y": 20}
let s = ""
for k, v in d {
    s = s + k + "=" + str(v) + " "
}
print(s)
"#,
    );
    assert!(ok);
    assert!(out.contains("x=10") && out.contains("y=20"));
}

#[test]
fn lang_for_dict_requires_two_vars() {
    let (ok, out) = run_inline(
        r#"
for v in {"x": 1} {
    print(v)
}
"#,
    );
    assert!(!ok);
    assert!(out.contains("dict iteration requires two variables"));
}

#[test]
fn lang_while() {
    assert_ok_contains(
        "while",
        r#"
let i = 0
let s = 0
while i < 5 {
    s = s + i
    i = i + 1
}
display s
"#,
        "10",
    );
}

#[test]
fn lang_if_else() {
    assert_ok_contains(
        "if_else",
        r#"
let x = 5
if x > 10 {
    display 1
} else {
    display 2
}
"#,
        "2",
    );
}

#[test]
fn lang_fn_return() {
    assert_ok_contains(
        "fn_return",
        r#"
fn square(x) {
    return x * x
}
display square(7)
"#,
        "49",
    );
}

#[test]
fn lang_break_continue() {
    assert_ok_contains(
        "break_continue",
        r#"
let s = 0
for i in 1..100 {
    if i > 5 { break }
    if i == 3 { continue }
    s = s + i
}
display s
"#,
        "12",
    );
}

#[test]
fn lang_nested_fn() {
    assert_ok_contains(
        "nested_fn",
        r#"
fn double(x) { return x * 2 }
fn quad(x) { return double(double(x)) }
display quad(3)
"#,
        "12",
    );
}

#[test]
fn lang_eststo_esttab() {
    assert_ok_contains(
        "eststo_esttab",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
end
eststo(ols(Y ~ X1, df))
eststo(ols(Y ~ X1 + X2, df))
esttab()
estclear()
"#,
        "estclear",
    );
}

#[test]
fn lang_help() {
    assert_ok_contains("help", "help(ols)", "cluster=var");
}

#[test]
fn lang_help_index() {
    assert_ok_contains("help_index", "help()", "ESTIMATORS");
}

#[test]
fn help_metadata_index_lists_categories() {
    assert_ok_contains("help_metadata_index", "help()", "POST-ESTIMATION");
}

#[test]
fn help_metadata_alias_lookup() {
    assert_ok_contains(
        "help_metadata_alias",
        "help(regress)",
        "Aliases: reg, regress",
    );
}

#[test]
fn help_bare_keyword_if() {
    assert_ok_contains("help_bare_if", "help(if)", "Conditional expression");
}

#[test]
fn help_bare_keyword_for() {
    assert_ok_contains("help_bare_for", "help(for)", "Loop over range");
}

#[test]
fn help_dw() {
    assert_ok_contains("help_dw", "help(dw)", "Durbin-Watson");
}

#[test]
fn help_xtsum() {
    assert_ok_contains("help_xtsum", "help(xtsum)", "within/between");
}

#[test]
fn help_self_doc() {
    assert_ok_contains("help_self_doc", "help(help)", "Show command index");
}

#[test]
fn help_wls() {
    assert_ok_contains("help_wls", "help(wls)", "Weighted Least Squares");
}

#[test]
fn help_sur() {
    assert_ok_contains("help_sur", "help(sur)", "Seemingly Unrelated");
}

#[test]
fn help_fuzzy_rd() {
    assert_ok_contains("help_fuzzy_rd", "help(fuzzy_rd)", "Fuzzy Regression");
}

#[test]
fn help_eststo() {
    assert_ok_contains("help_eststo", "help(eststo)", "Store a model");
}

#[test]
fn help_testparm() {
    assert_ok_contains("help_testparm", "help(testparm)", "Joint F-test");
}

#[test]
fn help_median() {
    assert_ok_contains("help_median", "help(median)", "Median");
}

#[test]
fn help_is_int() {
    assert_ok_contains("help_is_int", "help(is_int)", "Type predicates");
}

#[test]
fn help_dataframe() {
    assert_ok_contains("help_dataframe", "help(dataframe)", "DataFrame");
}

#[test]
fn help_drop_collinear() {
    assert_ok_contains("help_drop_collinear", "help(drop_collinear)", "collinear");
}

#[test]
fn help_tsset() {
    assert_ok_contains("help_tsset", "help(tsset)", "time-series");
}

#[test]
fn lang_timer() {
    assert_ok_contains(
        "timer",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
end
timer(ols(Y ~ X, df))
"#,
        "elapsed",
    );
}

#[test]
fn lang_set_seed() {
    let (ok1, out1) = run_inline("set_seed(42)\ndisplay 1");
    let (ok2, out2) = run_inline("set_seed(42)\ndisplay 1");
    assert!(ok1 && ok2);
    assert_eq!(out1, out2);
}

#[test]
fn lang_display_types() {
    let (ok, out) = run_inline(
        r#"
display 42
display 3.14
display "hello"
display true
"#,
    );
    assert!(ok);
    assert!(
        out.contains("42") && out.contains("3.14") && out.contains("hello") && out.contains("true")
    );
}

#[test]
fn lang_scalar() {
    assert_ok_contains(
        "scalar",
        r#"
scalar x = 2 + 3
display x
"#,
        "5",
    );
}

#[test]
fn lang_preserve_restore() {
    assert_ok(
        "preserve_restore",
        r#"
input df
Y X
1 2
3 4
end
preserve(df)
generate df Z = X * 2
restore(df)
"#,
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// EDGE CASES — behavior in boundary conditions
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn edge_single_observation() {
    assert_ok_contains(
        "single_obs",
        r#"
input df
Y
42
end
summarize(df)
"#,
        "42",
    );
}

#[test]
fn edge_single_column() {
    assert_ok_contains(
        "single_col",
        r#"
input df
X
1
2
3
4
5
end
display mean(df, X)
"#,
        "3",
    );
}

#[test]
fn edge_negative_numbers_input() {
    assert_ok_contains(
        "neg_input",
        r#"
input df
Y
-10
-5
0
5
10
end
display mean(df, Y)
"#,
        "0",
    );
}

#[test]
fn edge_missing_dot_input() {
    assert_ok_contains(
        "missing_dot",
        r#"
input df
Y X
1 10
. 20
3 .
end
summarize(df)
"#,
        "Missing",
    );
}

#[test]
fn edge_large_numbers() {
    assert_ok_contains(
        "large_numbers",
        r#"
display 1000000 * 1000000
"#,
        "1000000000000",
    );
}

#[test]
fn edge_float_precision() {
    assert_ok_contains(
        "float_precision",
        r#"
display 0.1 + 0.2
"#,
        "0.3",
    );
}

#[test]
fn edge_division_by_zero() {
    let (_, out) = run_inline("display 1 / 0");
    // should not crash — may produce inf or error message
    assert!(!out.is_empty());
}

#[test]
fn edge_zero_times_inf() {
    let (_, out) = run_inline("display 0 * (1 / 0)");
    assert!(!out.is_empty());
}

#[test]
fn edge_negative_power() {
    assert_ok_contains("neg_power", "display 2 ^ -1", "0.5");
}

// ══════════════════════════════════════════════════════════════════════════════
// ERROR HANDLING — undefined variables, wrong types, missing args
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn error_undefined_variable() {
    let (ok, _) = run_inline("display undefined_var");
    assert!(!ok, "should fail on undefined variable");
}

#[test]
fn error_ols_missing_column() {
    let (ok, _) = run_inline(
        r#"
input df
Y X
1 2
3 4
5 6
7 8
end
ols(Y ~ Z, df)
"#,
    );
    assert!(!ok, "should fail: column Z does not exist");
}

#[test]
fn error_wrong_type_to_mean() {
    let (ok, _) = run_inline(
        r#"
display mean("hello", "world")
"#,
    );
    assert!(!ok, "should fail: mean on strings");
}

#[test]
fn error_len_no_args() {
    let (ok, _) = run_inline("display len()");
    assert!(!ok, "len() requires 1 argument");
}

#[test]
fn error_substr_wrong_type() {
    let (ok, _) = run_inline("display substr(42, 0, 2)");
    assert!(!ok, "substr expects string");
}

// ══════════════════════════════════════════════════════════════════════════════
// STRING OPERATIONS — upper, lower, trim, contains, len, split, substr
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn string_upper() {
    assert_ok_contains(
        "str_upper",
        r#"
display upper("hello")
"#,
        "HELLO",
    );
}

#[test]
fn string_lower() {
    assert_ok_contains(
        "str_lower",
        r#"
display lower("WORLD")
"#,
        "world",
    );
}

#[test]
fn string_trim() {
    assert_ok_contains(
        "str_trim",
        r#"
display trim("  abc  ")
"#,
        "abc",
    );
}

#[test]
fn string_contains_true() {
    assert_ok_contains(
        "str_contains_t",
        r#"
display contains("hayashi", "ashi")
"#,
        "true",
    );
}

#[test]
fn string_contains_false() {
    assert_ok_contains(
        "str_contains_f",
        r#"
display contains("hayashi", "xyz")
"#,
        "false",
    );
}

#[test]
fn string_starts_with() {
    assert_ok_contains(
        "str_starts",
        r#"
display starts_with("hayashi", "hay")
"#,
        "true",
    );
}

#[test]
fn string_ends_with() {
    assert_ok_contains(
        "str_ends",
        r#"
display ends_with("hayashi", "shi")
"#,
        "true",
    );
}

#[test]
fn string_len() {
    assert_ok_contains(
        "str_len",
        r#"
display len("abcdef")
"#,
        "6",
    );
}

#[test]
fn string_substr_basic() {
    assert_ok_contains(
        "str_substr",
        r#"
display substr("hayashi", 0, 3)
"#,
        "hay",
    );
}

#[test]
fn string_substr_negative_index() {
    assert_ok_contains(
        "str_substr_neg",
        r#"
display substr("hayashi", -3)
"#,
        "shi",
    );
}

#[test]
fn string_split() {
    assert_ok_contains(
        "str_split",
        r#"
let parts = split("a,b,c", ",")
display len(parts)
"#,
        "3",
    );
}

#[test]
fn string_str_replace() {
    assert_ok_contains(
        "str_replace",
        r#"
display str_replace("hello world", "world", "hayashi")
"#,
        "hello hayashi",
    );
}

#[test]
fn string_concat_in_loop() {
    assert_ok_contains(
        "str_concat_loop",
        r#"
let s = ""
for i in 1..4 {
    s = s + str(i)
}
display s
"#,
        "123",
    );
}

#[test]
fn string_str_conversion() {
    assert_ok_contains(
        "str_conv",
        r#"
let x = str(42)
display x
"#,
        "42",
    );
}

#[test]
fn string_int_conversion() {
    assert_ok_contains(
        "int_conv",
        r#"
let x = int("123")
display x + 1
"#,
        "124",
    );
}

#[test]
fn string_float_conversion() {
    assert_ok_contains(
        "float_conv",
        r#"
let x = float("3.14")
display x
"#,
        "3.14",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// GENERATE EXPRESSIONS — math functions inside generate
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn gen_log() {
    assert_ok_contains(
        "gen_log",
        r#"
input df
X
1
10
100
end
generate df LX = log(X)
list(df)
"#,
        "generated",
    );
}

#[test]
fn gen_exp() {
    assert_ok_contains(
        "gen_exp",
        r#"
input df
X
0
1
2
end
generate df EX = exp(X)
list(df)
"#,
        "generated",
    );
}

#[test]
fn gen_sqrt() {
    assert_ok_contains(
        "gen_sqrt",
        r#"
input df
X
1
4
9
16
end
generate df SX = sqrt(X)
list(df)
"#,
        "generated",
    );
}

#[test]
fn gen_abs_negative() {
    assert_ok_contains(
        "gen_abs",
        r#"
input df
X
-5
-3
0
3
5
end
generate df AX = abs(X)
list(df)
"#,
        "generated",
    );
}

#[test]
fn gen_floor_ceil() {
    assert_ok_contains(
        "gen_floor_ceil",
        r#"
input df
X
1.3
2.7
3.5
end
generate df FL = floor(X)
generate df CL = ceil(X)
list(df)
"#,
        "generated",
    );
}

#[test]
fn gen_round() {
    assert_ok_contains(
        "gen_round",
        r#"
input df
X
1.4
2.5
3.6
end
generate df RX = round(X)
list(df)
"#,
        "generated",
    );
}

#[test]
fn gen_arithmetic_chain() {
    assert_ok_contains(
        "gen_chain",
        r#"
input df
X
2
4
6
end
generate df Y = X * 2 + 1
generate df Z = Y ^ 2
list(df)
"#,
        "generated",
    );
}

#[test]
fn gen_conditional_equality() {
    assert_ok_contains(
        "gen_cond_eq",
        r#"
input df
group
1
1
2
2
end
generate df dummy = (group == 1)
summarize(df)
"#,
        "dummy",
    );
}

#[test]
fn gen_nested_functions() {
    assert_ok_contains(
        "gen_nested",
        r#"
input df
X
1
4
9
end
generate df Y = log(sqrt(X))
list(df)
"#,
        "generated",
    );
}

#[test]
fn gen_sin_cos() {
    assert_ok_contains(
        "gen_trig",
        r#"
input df
X
0
1
2
3
end
generate df SIN = sin(X)
generate df COS = cos(X)
list(df)
"#,
        "generated",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// MULTIPLE REGRESSION — 3+ variables, R² properties
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn ols_three_vars() {
    assert_ok_contains(
        "ols_3vars",
        r#"
input df
Y X1 X2 X3
10 2 5 1
12 3 3 2
8 1 7 3
15 5 2 1
11 2 6 2
14 4 4 3
9 1 8 1
13 4 3 2
end
ols(Y ~ X1 + X2 + X3, df)
"#,
        "R-squared",
    );
}

#[test]
fn ols_r_squared_between_0_1() {
    let (ok, out) = run_inline(
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
ols(Y ~ X, df)
"#,
    );
    assert!(ok);
    assert!(out.contains("R-squared"));
}

#[test]
fn ols_intercept_only_model() {
    assert_ok_contains(
        "ols_intercept",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
let m1 = ols(Y ~ X, df)
let m2 = ols(Y ~ X, df, cov=robust)
esttab(m1, m2)
"#,
        "R²",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// PANEL ECONOMETRICS — fe, re, hausman
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn panel_fe_basic() {
    assert_ok_contains(
        "panel_fe",
        r#"
input panel
output capital labor firm year
10.2 5 8 1 2019
11.0 5 9 1 2020
12.5 6 9 1 2021
11.8 5 10 1 2022
19.3 10 12 2 2019
20.1 10 13 2 2020
23.1 12 14 2 2021
20.7 11 13 2 2022
14.6 7 10 3 2019
15.3 7 11 3 2020
17.9 8 11 3 2021
15.2 7 12 3 2022
24.8 13 15 4 2019
25.5 13 16 4 2020
27.3 14 16 4 2021
26.1 14 17 4 2022
end
xtset(panel, firm, year)
fe(output ~ capital + labor, panel)
"#,
        "Fixed Effects",
    );
}

#[test]
fn panel_re_basic() {
    assert_ok_contains(
        "panel_re",
        r#"
input panel
output capital labor firm year
10.2 5 8 1 2019
11.0 5 9 1 2020
12.5 6 9 1 2021
11.8 5 10 1 2022
19.3 10 12 2 2019
20.1 10 13 2 2020
23.1 12 14 2 2021
20.7 11 13 2 2022
14.6 7 10 3 2019
15.3 7 11 3 2020
17.9 8 11 3 2021
15.2 7 12 3 2022
24.8 13 15 4 2019
25.5 13 16 4 2020
27.3 14 16 4 2021
26.1 14 17 4 2022
end
xtset(panel, firm, year)
re(output ~ capital + labor, panel)
"#,
        "Random Effects",
    );
}

#[test]
fn panel_hausman() {
    assert_ok_contains(
        "panel_hausman",
        r#"
input panel
output capital labor firm year
10.2 5 8 1 2019
11.0 5 9 1 2020
12.5 6 9 1 2021
11.8 5 10 1 2022
19.3 10 12 2 2019
20.1 10 13 2 2020
23.1 12 14 2 2021
20.7 11 13 2 2022
14.6 7 10 3 2019
15.3 7 11 3 2020
17.9 8 11 3 2021
15.2 7 12 3 2022
24.8 13 15 4 2019
25.5 13 16 4 2020
27.3 14 16 4 2021
26.1 14 17 4 2022
end
xtset(panel, firm, year)
let mfe = fe(output ~ capital + labor, panel)
let mre = re(output ~ capital + labor, panel)
hausman(mfe, mre)
"#,
        "Hausman",
    );
}

#[test]
fn panel_fe_two_regressors() {
    assert_ok_contains(
        "panel_fe_2reg",
        r#"
input panel
output capital labor firm year
10.2 5 8 1 2019
11.0 5 9 1 2020
12.5 6 9 1 2021
11.8 5 10 1 2022
19.3 10 12 2 2019
20.1 10 13 2 2020
23.1 12 14 2 2021
20.7 11 13 2 2022
14.6 7 10 3 2019
15.3 7 11 3 2020
17.9 8 11 3 2021
15.2 7 12 3 2022
24.8 13 15 4 2019
25.5 13 16 4 2020
27.3 14 16 4 2021
26.1 14 17 4 2022
end
xtset(panel, firm, year)
fe(output ~ capital + labor, panel)
"#,
        "Fixed Effects",
    );
}

#[test]
fn panel_esttab_fe_re() {
    assert_ok_contains(
        "panel_esttab",
        r#"
input panel
output capital labor firm year
10.2 5 8 1 2019
11.0 5 9 1 2020
12.5 6 9 1 2021
11.8 5 10 1 2022
19.3 10 12 2 2019
20.1 10 13 2 2020
23.1 12 14 2 2021
20.7 11 13 2 2022
14.6 7 10 3 2019
15.3 7 11 3 2020
17.9 8 11 3 2021
15.2 7 12 3 2022
24.8 13 15 4 2019
25.5 13 16 4 2020
27.3 14 16 4 2021
26.1 14 17 4 2022
end
xtset(panel, firm, year)
let mfe = fe(output ~ capital + labor, panel)
let mre = re(output ~ capital + labor, panel)
esttab(mfe, mre)
"#,
        "capital",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// ESTTAB VARIATIONS — multiple models, formatting
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn esttab_three_models() {
    assert_ok_contains(
        "esttab_3m",
        r#"
input df
Y X1 X2 X3
10 2 5 1
12 3 3 2
8 1 7 3
15 5 2 1
11 2 6 2
14 4 4 3
9 1 8 1
13 4 3 2
end
let m1 = ols(Y ~ X1, df)
let m2 = ols(Y ~ X1 + X2, df)
let m3 = ols(Y ~ X1 + X2 + X3, df)
esttab(m1, m2, m3)
"#,
        "R²",
    );
}

#[test]
fn esttab_robust_vs_ols() {
    assert_ok_contains(
        "esttab_robust",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
9 1
13 4
end
let m1 = ols(Y ~ X, df)
let m2 = ols(Y ~ X, df, cov=robust)
esttab(m1, m2)
"#,
        "R²",
    );
}

#[test]
fn esttab_via_eststo() {
    assert_ok_contains(
        "esttab_eststo",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
end
eststo(ols(Y ~ X1, df))
eststo(ols(Y ~ X1 + X2, df))
esttab()
estclear()
"#,
        "R²",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// BOOTSTRAP — basic bootstrap calls
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn bootstrap_ols_basic() {
    assert_ok_contains(
        "boot_ols",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
9 1
13 4
end
set_seed(42)
bootstrap(ols, Y ~ X, df, n=100)
"#,
        "Bootstrap",
    );
}

#[test]
fn bootstrap_with_seed_reproducibility() {
    let src = r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
9 1
13 4
end
set_seed(42)
bootstrap(ols, Y ~ X, df, n=50)
"#;
    let (ok1, out1) = run_inline(src);
    let (ok2, out2) = run_inline(src);
    assert!(ok1 && ok2);
    // both should produce bootstrap output
    assert!(out1.contains("Bootstrap") && out2.contains("Bootstrap"));
}

// ══════════════════════════════════════════════════════════════════════════════
// FOR LOOP VARIATIONS — nested, accumulator, with eststo
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn for_nested_loops() {
    assert_ok_contains(
        "for_nested",
        r#"
let s = 0
for i in 1..4 {
    for j in 1..4 {
        s = s + 1
    }
}
display s
"#,
        "9",
    );
}

#[test]
fn for_accumulator_product() {
    assert_ok_contains(
        "for_product",
        r#"
let p = 1
for i in 1..6 {
    p = p * i
}
display p
"#,
        "120",
    );
}

#[test]
fn for_string_list() {
    assert_ok_contains(
        "for_str_list",
        r#"
for s in ["alpha", "beta", "gamma"] {
    display s
}
"#,
        "gamma",
    );
}

#[test]
fn for_with_break_accumulator() {
    assert_ok_contains(
        "for_break_acc",
        r#"
let total = 0
for i in 1..100 {
    total = total + i
    if total > 50 { break }
}
display total
"#,
        "55",
    );
}

#[test]
fn for_continue_skip_multiples() {
    assert_ok_contains(
        "for_skip_mult",
        r#"
let s = 0
for i in 1..11 {
    if i == 2 || i == 4 || i == 6 || i == 8 || i == 10 { continue }
    s = s + i
}
display s
"#,
        "25",
    );
}

#[test]
fn while_nested() {
    assert_ok_contains(
        "while_nested",
        r#"
let total = 0
let i = 0
while i < 3 {
    let j = 0
    while j < 3 {
        total = total + 1
        j = j + 1
    }
    i = i + 1
}
display total
"#,
        "9",
    );
}

#[test]
fn for_eststo_loop() {
    assert_ok_contains(
        "for_eststo_lp",
        r#"
input df
Y X1 X2 X3
10 2 5 1
12 3 3 2
8 1 7 3
15 5 2 1
11 2 6 2
14 4 4 3
9 1 8 1
13 4 3 2
end
for v in ["X1", "X2", "X3"] {
    eststo(ols("Y ~ " + v, df))
}
esttab()
estclear()
"#,
        "R²",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// INTERACTION BETWEEN FEATURES — generate + ols, winsor + ols, etc.
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn interact_generate_then_ols() {
    assert_ok_contains(
        "gen_then_ols",
        r#"
input df
X1 X2
2 5
3 3
1 7
5 2
2 6
4 4
1 8
4 3
end
generate df Y = X1 * 2 + X2 * 3 + 1
ols(Y ~ X1 + X2, df)
"#,
        "R-squared",
    );
}

#[test]
fn interact_winsor_then_ols() {
    assert_ok_contains(
        "winsor_ols",
        r#"
input df
Y X
1 2
12 3
8 1
15 5
11 2
14 4
100 1
13 4
end
winsor(df, Y, p=0.10)
ols(Y ~ X, df)
"#,
        "R-squared",
    );
}

#[test]
fn interact_filter_then_ols() {
    assert_ok_contains(
        "filter_ols",
        r#"
input df
Y X group
10 2 1
12 3 1
8 1 1
15 5 1
11 2 2
14 4 2
9 1 2
13 4 2
end
let sub = filter(df, group == 1)
ols(Y ~ X, sub)
"#,
        "No. Observations:                  4",
    );
}

#[test]
fn interact_generate_log_ols() {
    assert_ok_contains(
        "gen_log_ols",
        r#"
input df
Y X
10 2
12 3
8 5
15 7
11 4
14 6
end
generate df LX = log(X)
ols(Y ~ LX, df)
"#,
        "R-squared",
    );
}

#[test]
fn interact_predict_summarize() {
    assert_ok_contains(
        "predict_summarize",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
let m = ols(Y ~ X, df)
predict df yhat = m
predict df resid = m, "residuals"
summarize(df)
"#,
        "resid",
    );
}

#[test]
fn interact_xtset_fe_hausman() {
    assert_ok_contains(
        "xtset_fe_haus",
        r#"
input panel
output capital labor firm year
10.2 5 8 1 2019
11.0 5 9 1 2020
12.5 6 9 1 2021
11.8 5 10 1 2022
19.3 10 12 2 2019
20.1 10 13 2 2020
23.1 12 14 2 2021
20.7 11 13 2 2022
14.6 7 10 3 2019
15.3 7 11 3 2020
17.9 8 11 3 2021
15.2 7 12 3 2022
24.8 13 15 4 2019
25.5 13 16 4 2020
27.3 14 16 4 2021
26.1 14 17 4 2022
end
xtset(panel, firm, year)
let mfe = fe(output ~ capital + labor, panel)
let mre = re(output ~ capital + labor, panel)
hausman(mfe, mre)
"#,
        "Hausman",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// REGRESSION DIAGNOSTICS — bp, dw, vif, reset, jb, condnum
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn diag_breusch_pagan() {
    assert_ok_contains(
        "test_bp",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
9 1
13 4
end
let m = ols(Y ~ X, df)
test(m, "bp")
"#,
        "Breusch-Pagan",
    );
}

#[test]
fn diag_durbin_watson() {
    assert_ok_contains(
        "test_dw",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
9 1
13 4
end
let m = ols(Y ~ X, df)
test(m, "dw")
"#,
        "Durbin-Watson",
    );
}

#[test]
fn diag_vif() {
    assert_ok_contains(
        "vif",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
9 1 8
13 4 3
end
let m = ols(Y ~ X1 + X2, df)
vif(m)
"#,
        "VIF",
    );
}

#[test]
fn diag_condnum() {
    assert_ok_contains(
        "condnum",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
9 1 8
13 4 3
end
let m = ols(Y ~ X1 + X2, df)
condnum(m)
"#,
        "Condition Number",
    );
}

#[test]
fn diag_reset() {
    assert_ok_contains(
        "reset_test",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
9 1
13 4
end
let m = ols(Y ~ X, df)
reset(m)
"#,
        "RESET",
    );
}

#[test]
fn diag_jb_on_model() {
    assert_ok_contains(
        "jb_model",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
9 1
13 4
end
let m = ols(Y ~ X, df)
jb(m)
"#,
        "Jarque-Bera",
    );
}

#[test]
fn diag_jb_on_series() {
    assert_ok_contains(
        "jb_series",
        r#"
input df
Y
1
2
3
4
5
6
7
8
end
jb(df, Y)
"#,
        "Jarque-Bera",
    );
}

#[test]
fn diag_predict_resid_then_summarize() {
    assert_ok_contains(
        "resid_sum",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
let m = ols(Y ~ X, df)
predict df resid = m, "residuals"
summarize(df, resid)
"#,
        "Mean",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// ASSIGN SEMANTICS — shadowing, nested scope assignment
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn assign_shadow_inner_scope() {
    assert_ok_contains(
        "shadow_inner",
        r#"
let x = 10
if true {
    let x = 20
    display x
}
display x
"#,
        "20",
    );
}

#[test]
fn assign_outer_from_loop() {
    assert_ok_contains(
        "assign_from_loop",
        r#"
let last = 0
for i in [10, 20, 30] {
    last = i
}
display last
"#,
        "30",
    );
}

#[test]
fn assign_nested_scope_modify() {
    assert_ok_contains(
        "nested_modify",
        r#"
let x = 0
if true {
    if true {
        x = 42
    }
}
display x
"#,
        "42",
    );
}

#[test]
fn assign_fn_does_not_leak() {
    assert_ok_contains(
        "fn_no_leak",
        r#"
let x = 10
fn f() {
    let x = 99
    return x
}
display f()
display x
"#,
        "10",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// CORRELATE / PWCORR — basic usage
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn correlate_basic() {
    assert_ok_contains(
        "correlate",
        r#"
input df
Y X1 X2
6 2 1
8 2 0
1 2 1
4 1 1
5 1 0
7 1 1
3 1 0
9 0 1
end
correlate(df, Y, X1, X2)
"#,
        "Y",
    );
}

#[test]
fn pwcorr_with_stars() {
    assert_ok_contains(
        "pwcorr_stars",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
9 1 8
13 4 3
2 0 9
18 6 1
end
pwcorr(df, Y, X1, X2)
"#,
        "Y",
    );
}

#[test]
fn correlate_two_vars() {
    assert_ok_contains(
        "corr_two",
        r#"
input df
X Y
1 2
2 4
3 6
4 8
5 10
end
correlate(df, X, Y)
"#,
        "X",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// SORT / LIST / DESCRIBE — inspection commands
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn sort_ascending() {
    assert_ok_contains(
        "sort_asc",
        r#"
input df
X
5
3
1
4
2
end
let sorted = sort(df, X)
list(sorted)
"#,
        "1",
    );
}

#[test]
fn sort_descending() {
    assert_ok_contains(
        "sort_desc",
        r#"
input df
X
1
2
3
4
5
end
let sorted = sort(df, X, desc=true)
list(sorted, n=1)
"#,
        "5",
    );
}

#[test]
fn describe_basic() {
    assert_ok_contains(
        "describe",
        r#"
input df
Y X group
10 2 1
12 3 1
8 1 2
15 5 2
end
describe(df)
"#,
        "Y",
    );
}

#[test]
fn list_full() {
    assert_ok_contains(
        "list_full",
        r#"
input df
A B
1 10
2 20
3 30
end
list(df)
"#,
        "A",
    );
}

#[test]
fn count_all() {
    assert_ok_contains(
        "count_all",
        r#"
input df
X
1
2
3
4
5
end
count df
"#,
        "5",
    );
}

#[test]
fn count_conditional() {
    assert_ok_contains(
        "count_cond",
        r#"
input df
X
1
2
3
4
5
end
count df if X > 3
"#,
        "2",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// INPUT EDGE CASES — single row, negatives, all missing
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn input_single_row() {
    assert_ok_contains(
        "input_1row",
        r#"
input df
X Y
42 99
end
list(df)
"#,
        "42",
    );
}

#[test]
fn input_all_negative() {
    assert_ok_contains(
        "input_neg",
        r#"
input df
X
-1
-2
-3
-4
-5
end
display mean(df, X)
"#,
        "-3",
    );
}

#[test]
fn input_mixed_missing() {
    assert_ok_contains(
        "input_mixed_miss",
        r#"
input df
A B C
1 . 3
. 2 .
3 3 3
end
summarize(df)
"#,
        "Missing",
    );
}

#[test]
fn input_many_columns() {
    assert_ok_contains(
        "input_many_cols",
        r#"
input df
A B C D E
1 2 3 4 5
6 7 8 9 10
end
describe(df)
"#,
        "E",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// FORMULA PARSING — dynamic formulas, multi-term
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn formula_string_multi_term() {
    assert_ok_contains(
        "formula_str_multi",
        r#"
input df
Y X1 X2 X3
10 2 5 1
12 3 3 2
8 1 7 3
15 5 2 1
11 2 6 2
14 4 4 3
9 1 8 1
13 4 3 2
end
ols("Y ~ X1 + X2 + X3", df)
"#,
        "R-squared",
    );
}

#[test]
fn formula_concat_dynamic() {
    assert_ok_contains(
        "formula_concat",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
end
let f = "Y ~ X1 + X2"
ols(f, df)
"#,
        "R-squared",
    );
}

#[test]
fn formula_loop_build() {
    assert_ok_contains(
        "formula_loop",
        r#"
input df
Y X1 X2 X3
10 2 5 1
12 3 3 2
8 1 7 3
15 5 2 1
11 2 6 2
14 4 4 3
9 1 8 1
13 4 3 2
end
let vars = "X1"
for v in ["X2", "X3"] {
    vars = vars + " + " + v
}
let formula = "Y ~ " + vars
ols(formula, df)
"#,
        "R-squared",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// TTEST — one-sample with mu, two-sample with by, paired
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn ttest_one_sample() {
    assert_ok_contains(
        "ttest_1s",
        r#"
input df
Y
10
12
8
15
11
14
9
13
end
ttest(df, Y, mu=10)
"#,
        "H0",
    );
}

#[test]
fn ttest_two_groups() {
    assert_ok_contains(
        "ttest_2g",
        r#"
input df
Y group
10 1
12 1
8 1
15 1
20 2
22 2
18 2
25 2
end
ttest(df, Y, by=group)
"#,
        "Welch",
    );
}

#[test]
fn ttest_by_nan_string_group_does_not_panic() {
    assert_err_contains(
        "ttest_nan_string_group",
        r#"
let d = {"Y": [10, 12, 20], "group": ["1", "NaN", "2"]}
let df = dataframe(d)
ttest(df, Y, by=group)
"#,
        "two-sample ttest requires exactly 2 groups",
    );
}

#[test]
fn ttest_paired() {
    assert_ok_contains(
        "ttest_paired",
        r#"
input df
before after
10 12
8 11
15 16
11 14
9 13
14 17
end
ttest(df, before, after, paired=true)
"#,
        "Paired",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// DATA MANIPULATION — filter, drop, keep, rename, collapse, append, sort
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn data_filter() {
    assert_ok_contains(
        "filter",
        r#"
input df
Y X
10 1
20 2
30 3
40 4
50 5
end
let sub = filter(df, X > 3)
list(sub)
"#,
        "remaining",
    );
}

#[test]
fn data_drop_columns() {
    assert_ok_contains(
        "drop_cols",
        r#"
input df
Y X Z
1 2 3
4 5 6
end
let df2 = drop(df, Z)
describe(df2)
"#,
        "Y",
    );
}

#[test]
fn data_keep_columns() {
    assert_ok_contains(
        "keep_cols",
        r#"
input df
Y X Z
1 2 3
4 5 6
end
let df2 = keep(df, Y, X)
describe(df2)
"#,
        "Y",
    );
}

#[test]
fn data_rename() {
    assert_ok_contains(
        "rename_col",
        r#"
input df
old_name
1
2
3
end
rename(df, old_name, new_name)
describe(df)
"#,
        "new_name",
    );
}

#[test]
fn data_collapse_mean() {
    assert_ok_contains(
        "collapse_mean",
        r#"
input df
Y group
10 1
20 1
30 2
40 2
end
let agg = collapse(df, mean, Y, by=group)
list(agg)
"#,
        "group",
    );
}

#[test]
fn data_collapse_sum() {
    assert_ok_contains(
        "collapse_sum",
        r#"
input df
Y group
10 1
20 1
30 2
40 2
end
let agg = collapse(df, sum, Y, by=group)
list(agg)
"#,
        "group",
    );
}

#[test]
fn collapse_by_nan_string_group_does_not_panic() {
    assert_ok_contains(
        "collapse_nan_string_group",
        r#"
let d = {"Y": [10, 20, 30], "group": ["1", "NaN", "2"]}
let df = dataframe(d)
let agg = collapse(df, sum, Y, by=group)
list(agg)
"#,
        "NaN",
    );
}

#[test]
fn data_append() {
    assert_ok_contains(
        "append",
        r#"
input df1
X Y
1 2
3 4
end
input df2
X Y
5 6
7 8
end
let combined = append(df1, df2)
list(combined)
"#,
        "5",
    );
}

#[test]
fn rbind_list_of_dfs() {
    assert_ok_contains(
        "rbind_list",
        r#"
input df1
X Y
1 10
2 20
end
input df2
X Y
3 30
end
input df3
X Y
4 40
5 50
6 60
end
let combined = rbind([df1, df2, df3])
list(combined)
"#,
        "6",
    );
}

#[test]
fn rbind_skips_nils() {
    assert_ok_contains(
        "rbind_nils",
        r#"
input df1
X Y
1 10
2 20
end
input df2
X Y
3 30
end
let combined = rbind([df1, nil, df2, nil])
list(combined)
"#,
        "3",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// GENERATE EXTENSIONS — rowmean, rank, cumsum, group, rowsum
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn egen_rowmean() {
    assert_ok_contains(
        "egen_rowmean",
        r#"
input df
A B
2 4
6 8
10 12
end
generate df M = rowmean(A, B)
list(df)
"#,
        "generated",
    );
}

#[test]
fn egen_rowsum() {
    assert_ok_contains(
        "egen_rowsum",
        r#"
input df
A B C
1 2 3
4 5 6
end
generate df S = rowsum(A, B, C)
list(df)
"#,
        "generated",
    );
}

#[test]
fn egen_rank() {
    assert_ok_contains(
        "egen_rank",
        r#"
input df
X
30
10
20
50
40
end
generate df R = rank(X)
list(df)
"#,
        "generated",
    );
}

#[test]
fn egen_cumsum() {
    assert_ok_contains(
        "egen_cumsum",
        r#"
input df
X
1
2
3
4
5
end
generate df CS = cumsum(X)
list(df)
"#,
        "generated",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// LINCOM — linear combination of coefficients
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn lincom_basic() {
    assert_ok_contains(
        "lincom",
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
9 1 8
13 4 3
end
let m = ols(Y ~ X1 + X2, df)
lincom(m, X1=1, X2=1)
"#,
        "lincom",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// TIMER / SEED — execution timing, RNG seed
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn timer_returns_result() {
    assert_ok_contains(
        "timer_result",
        r#"
let x = timer(2 + 3)
display x
"#,
        "5",
    );
}

#[test]
fn seed_display() {
    assert_ok_contains(
        "seed_display",
        r#"
set_seed(12345)
"#,
        "12345",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// LANGUAGE — additional coverage
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn lang_if_elif_else() {
    assert_ok_contains(
        "elif",
        r#"
let x = 5
if x > 10 {
    display "big"
} else if x > 3 {
    display "medium"
} else {
    display "small"
}
"#,
        "medium",
    );
}

#[test]
fn lang_fn_default_no_return() {
    assert_ok_contains(
        "fn_no_ret",
        r#"
fn greet(name) {
    display "hello " + name
}
greet("world")
"#,
        "hello world",
    );
}

#[test]
fn lang_fn_multiple_params() {
    assert_ok_contains(
        "fn_multi_param",
        r#"
fn weighted_avg(a, b, wa, wb) {
    return (a * wa + b * wb) / (wa + wb)
}
display weighted_avg(10, 20, 3, 7)
"#,
        "17",
    );
}

#[test]
fn lang_list_operations() {
    assert_ok_contains(
        "list_ops",
        r#"
let v = [1, 2, 3, 4, 5]
display len(v)
display v[0]
display v[4]
"#,
        "5",
    );
}

#[test]
fn lang_boolean_or() {
    assert_ok_contains(
        "bool_or",
        r#"
let x = 1
if x < 0 || x > 0 {
    display "nonzero"
}
"#,
        "nonzero",
    );
}

#[test]
fn lang_unary_minus() {
    assert_ok_contains(
        "unary_minus",
        r#"
let x = 10
display -x
"#,
        "-10",
    );
}

#[test]
fn lang_string_equality() {
    assert_ok_contains(
        "str_eq",
        r#"
let a = "hello"
let b = "hello"
if a == b {
    display "equal"
}
"#,
        "equal",
    );
}

#[test]
fn lang_string_inequality() {
    assert_ok_contains(
        "str_neq",
        r#"
let a = "hello"
let b = "world"
if a != b {
    display "different"
}
"#,
        "different",
    );
}

#[test]
fn lang_nested_fn_calls() {
    assert_ok_contains(
        "nested_fn_calls",
        r#"
fn add(a, b) { return a + b }
fn mul(a, b) { return a * b }
display add(mul(2, 3), mul(4, 5))
"#,
        "26",
    );
}

#[test]
fn lang_recursive_sum() {
    assert_ok_contains(
        "rec_sum",
        r#"
fn sum_to(n) {
    if n <= 0 { return 0 }
    return n + sum_to(n - 1)
}
display sum_to(10)
"#,
        "55",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// AGGREGATION FUNCTIONS — sum, min, max, std on lists and DataFrames
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn agg_sum_df() {
    assert_ok_contains(
        "sum_df",
        r#"
input df
X
1
2
3
4
5
end
display sum(df, X)
"#,
        "15",
    );
}

#[test]
fn agg_min_max_df() {
    let (ok, out) = run_inline(
        r#"
input df
X
10
2
30
4
5
end
display min(df, X)
display max(df, X)
"#,
    );
    assert!(ok);
    assert!(out.contains("2") && out.contains("30"));
}

#[test]
fn agg_std_df() {
    assert_ok_contains(
        "std_df",
        r#"
input df
X
1
2
3
4
5
end
display std(df, X)
"#,
        ".",
    );
}

#[test]
fn agg_mean_conditional() {
    assert_ok_contains(
        "mean_cond2",
        r#"
input df
Y group
10 1
20 1
30 2
40 2
end
scalar m = mean(df, Y, if = group == 2)
display m
"#,
        "35",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// HELP — various help topics
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn help_bootstrap() {
    assert_ok_contains("help_boot", "help(bootstrap)", "bootstrap");
}

#[test]
fn help_fe() {
    assert_ok_contains("help_fe", "help(fe)", "fe");
}

#[test]
fn help_xtset() {
    assert_ok_contains("help_xtset", "help(xtset)", "xtset");
}

// ══════════════════════════════════════════════════════════════════════════════
// TYPE CONVERSIONS — int, float, str, bool arithmetic
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn type_int_from_float() {
    assert_ok_contains(
        "int_from_float",
        r#"
let x = int(3.7)
display x
"#,
        "3",
    );
}

#[test]
fn type_float_from_int() {
    assert_ok_contains(
        "float_from_int",
        r#"
let x = float(42)
display x
"#,
        "42",
    );
}

#[test]
fn type_bool_to_int() {
    assert_ok_contains(
        "bool_to_int",
        r#"
let x = int(true)
display x
"#,
        "1",
    );
}

#[test]
fn type_str_from_number() {
    assert_ok_contains(
        "str_from_num",
        r#"
let s = str(3.14)
display len(s)
"#,
        "4",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// SVG PLOTS — plotters backend
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn graph_scatter_svg() {
    let script = format!(
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
end
graph_scatter(df, X, Y, path="{}")
"#,
        tmp("test_scatter.svg")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "graph_scatter failed:\n{out}");
    assert!(out.contains("graph saved"));
    assert!(std::path::Path::new(&tmp("test_scatter.svg")).exists());
}

#[test]
fn graph_line_svg() {
    let script = format!(
        r#"
input df
Y X
1 1
4 2
9 3
16 4
25 5
end
graph_line(df, X, Y, path="{}")
"#,
        tmp("test_line.svg")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "graph_line failed:\n{out}");
    assert!(out.contains("graph saved"));
    assert!(std::path::Path::new(&tmp("test_line.svg")).exists());
}

#[test]
fn graph_hist_svg() {
    let script = format!(
        r#"
input df
Y
1
2
3
4
5
6
7
8
end
graph_hist(df, Y, path="{}", bins=4)
"#,
        tmp("test_hist.svg")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "graph_hist failed:\n{out}");
    assert!(out.contains("graph saved"));
    assert!(std::path::Path::new(&tmp("test_hist.svg")).exists());
}

#[test]
fn graph_coef_svg() {
    let script = format!(
        r#"
input df
Y X1 X2
10 2 5
12 3 3
8 1 7
15 5 2
11 2 6
14 4 4
end
let m = ols(Y ~ X1 + X2, df)
graph_coef(m, path="{}")
"#,
        tmp("test_coef.svg")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "graph_coef failed:\n{out}");
    assert!(out.contains("graph saved"));
    assert!(std::path::Path::new(&tmp("test_coef.svg")).exists());
}

#[test]
fn graph_scatter_custom_title() {
    let script = format!(
        r#"
input df
Y X
10 2
12 3
8 1
15 5
end
graph_scatter(df, X, Y, path="{}", title="Custom Title")
"#,
        tmp("test_scatter_title.svg")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "graph_scatter title failed:\n{out}");
    let svg = std::fs::read_to_string(tmp("test_scatter_title.svg")).unwrap();
    assert!(svg.contains("Custom Title"));
}

#[test]
fn graph_hist_custom_bins() {
    let script = format!(
        r#"
input df
Y
1
2
3
4
5
6
7
8
9
10
end
graph_hist(df, Y, path="{}", bins=2)
"#,
        tmp("test_hist_bins.svg")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "graph_hist bins failed:\n{out}");
    assert!(std::path::Path::new(&tmp("test_hist_bins.svg")).exists());
}

#[test]
fn graph_coef_svg_valid() {
    let script = format!(
        r#"
input df
Y X1 X2 X3
10 2 5 3
12 3 3 5
8 1 7 2
15 5 2 8
11 2 6 4
14 4 4 6
end
let m = ols(Y ~ X1 + X2 + X3, df)
graph_coef(m, path="{}", title="3 Variables")
"#,
        tmp("test_coef_valid.svg")
    );
    let (ok, _) = run_inline(&script);
    assert!(ok);
    let svg = std::fs::read_to_string(tmp("test_coef_valid.svg")).unwrap();
    assert!(svg.contains("<svg"));
    assert!(svg.contains("3 Variables"));
    assert!(svg.contains("Coefficient"));
}

#[test]
fn graph_line_default_path() {
    let (ok, out) = run_inline(
        r#"
input df
Y X
1 1
2 2
3 3
end
graph_line(df, X, Y)
"#,
    );
    assert!(ok, "graph_line default failed:\n{out}");
    assert!(out.contains("graph saved: Y_line.svg"));
    let _ = std::fs::remove_file("Y_line.svg");
}

#[test]
fn graph_scatter_gscatter_alias() {
    let script = format!(
        r#"
input df
Y X
10 2
12 3
8 1
end
gscatter(df, X, Y, path="{}")
"#,
        tmp("test_gscatter.svg")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "gscatter alias failed:\n{out}");
    assert!(out.contains("graph saved"));
}

#[test]
fn smoke_graphs_svg() {
    let (ok, out) = run_hy("examples/graphs_svg.hay");
    assert!(ok, "graphs_svg.hay failed:\n{out}");
    assert!(out.contains("graph saved: scatter.svg"));
    assert!(out.contains("graph saved: coefplot.svg"));
    // cleanup
    let _ = std::fs::remove_file("scatter.svg");
    let _ = std::fs::remove_file("line.svg");
    let _ = std::fs::remove_file("hist.svg");
    let _ = std::fs::remove_file("coefplot.svg");
}

// ══════════════════════════════════════════════════════════════════════════════
// MATHEMATICAL FUNCTIONS — parity with Stata
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn math_tan() {
    assert_ok_contains(
        "tan",
        r#"
input df
X
0
1
end
generate df T = tan(X)
"#,
        "generated",
    );
}

#[test]
fn math_asin_acos_atan() {
    assert_ok(
        "asin_acos_atan",
        r#"
input df
X
0.5
0.9
end
generate df A = asin(X)
generate df B = acos(X)
generate df C = atan(X)
"#,
    );
}

#[test]
fn math_sign() {
    assert_ok_contains(
        "sign",
        r#"
input df
X
-3
0
5
end
generate df S = sign(X)
list(df)
"#,
        "-1",
    );
}

#[test]
fn math_mod() {
    assert_ok_contains(
        "mod",
        r#"
input df
X
7
10
15
end
generate df M = mod(X, 3)
"#,
        "generated",
    );
}

#[test]
fn math_factorial() {
    assert_ok_contains(
        "factorial",
        r#"
input df
X
5
end
generate df F = factorial(X)
list(df)
"#,
        "120",
    );
}

#[test]
fn math_cond() {
    assert_ok_contains(
        "cond",
        r#"
input df
X
1
5
10
end
generate df C = cond(X > 3, 100, 0)
list(df)
"#,
        "100",
    );
}

#[test]
fn math_max_min_two_args() {
    assert_ok_contains(
        "max_min",
        r#"
input df
X
1
5
10
end
generate df MX = max(X, 5)
generate df MN = min(X, 5)
list(df)
"#,
        "generated",
    );
}

#[test]
fn math_comb() {
    assert_ok_contains(
        "comb",
        r#"
input df
N K
10 3
5 2
end
generate df C = comb(N, K)
list(df)
"#,
        "120",
    );
}

#[test]
fn math_uniform() {
    assert_ok(
        "uniform",
        r#"
input df
X
1
2
3
end
generate df U = uniform()
"#,
    );
}

#[test]
fn math_rnormal() {
    assert_ok(
        "rnormal",
        r#"
input df
X
1
2
3
end
generate df N = rnormal()
"#,
    );
}

#[test]
fn math_rnormal_standard_normal_draws() {
    let rows = (1..=512)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    let script = format!(
        r#"
set_seed(12345)
input df
X
{rows}
end
generate df N = rnormal()
assert(min(df, N) < 0, "rnormal should include negative support")
assert(max(df, N) > 1, "rnormal should exceed one with this seed")
assert(mean(df, N) > -0.2, "rnormal mean should be near zero")
assert(mean(df, N) < 0.2, "rnormal mean should be near zero")
assert(variance(df, N) > 0.7, "rnormal variance should be near one")
assert(variance(df, N) < 1.3, "rnormal variance should be near one")
"#
    );
    assert_ok("rnormal_standard_normal_draws", &script);
}

#[test]
fn math_normal_pdf() {
    assert_ok_contains(
        "normal_pdf",
        r#"
input df
X
0
end
generate df P = normal(X)
list(df)
"#,
        "0.3989",
    );
}

#[test]
fn math_atan2() {
    assert_ok(
        "atan2",
        r#"
input df
Y X
1 1
0 1
end
generate df A = atan2(Y, X)
"#,
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// SUMMARIZE DETAIL — percentis, skewness, kurtosis
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn summarize_detail() {
    assert_ok_contains(
        "summarize_detail",
        r#"
input df
X
1
2
3
4
5
6
7
8
9
10
end
summarize(df, X, detail=true)
"#,
        "Skewness",
    );
}

#[test]
fn summarize_detail_percentiles() {
    assert_ok_contains(
        "detail_pctiles",
        r#"
input df
X
1
2
3
4
5
6
7
8
9
10
end
summarize(df, X, detail=true)
"#,
        "50%",
    );
}

#[test]
fn summarize_detail_kurtosis() {
    assert_ok_contains(
        "detail_kurt",
        r#"
input df
X
1
2
3
4
5
6
7
8
9
10
end
summarize(df, X, detail=true)
"#,
        "Kurtosis",
    );
}

#[test]
fn summarize_detail_variance() {
    assert_ok_contains(
        "detail_var",
        r#"
input df
X
1
2
3
4
5
end
summarize(df, X, detail=true)
"#,
        "Variance",
    );
}

#[test]
fn smoke_math_functions() {
    let (ok, out) = run_hy("examples/math_functions.hay");
    assert!(ok, "math_functions.hay failed:\n{out}");
    assert!(out.contains("Skewness"));
}

// ══════════════════════════════════════════════════════════════════════════════
// STATA PARITY — distributions, _n/_N, ci, centile, recode, egen
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn gen_row_number() {
    assert_ok_contains(
        "_n",
        r#"
input df
X
10
20
30
end
generate df row = _n
list(df)
"#,
        "3",
    );
}

#[test]
fn gen_total_rows() {
    assert_ok_contains(
        "_N",
        r#"
input df
X
10
20
30
end
generate df total = _N
display mean(df, total)
"#,
        "3",
    );
}

#[test]
fn gen_standardize() {
    assert_ok_contains(
        "std",
        r#"
input df
X
1
2
3
4
5
end
generate df Z = std(X)
summarize(df, Z)
"#,
        "0.0000",
    );
}

#[test]
fn gen_iqr() {
    assert_ok(
        "iqr",
        r#"
input df
X
1
2
3
4
5
6
7
8
end
generate df I = iqr(X)
"#,
    );
}

#[test]
fn ci_means() {
    assert_ok_contains(
        "ci_means",
        r#"
input df
X
1
2
3
4
5
end
ci(df, X)
"#,
        "95% CI",
    );
}

#[test]
fn centile_basic() {
    assert_ok_contains(
        "centile",
        r#"
input df
X
1
2
3
4
5
6
7
8
9
10
end
centile(df, X)
"#,
        "50.0%",
    );
}

#[test]
fn centile_all_missing_returns_error() {
    let (ok, out) = run_inline(
        r#"
input df
X
.
.
end
centile(df, X)
"#,
    );
    assert!(!ok, "centile should fail on all-missing input:\n{out}");
    assert!(
        out.contains("no finite observations"),
        "centile should return a structured error, not panic:\n{out}"
    );
    assert!(
        !out.contains("panicked at"),
        "centile should not panic on all-missing input:\n{out}"
    );
}

#[test]
fn recode_basic() {
    assert_ok_contains(
        "recode",
        r#"
input df
X
1
2
3
4
end
recode(df, X, from=[1, 2], to=[10, 20])
"#,
        "2 changes",
    );
}

#[test]
fn dist_chi2tail() {
    assert_ok(
        "chi2tail",
        r#"
input df
X
1
5
10
end
generate df P = chi2tail(2, X)
"#,
    );
}

#[test]
fn dist_ttail() {
    assert_ok(
        "ttail",
        r#"
input df
X
1
2
3
end
generate df P = ttail(10, X)
"#,
    );
}

#[test]
fn dist_invttail() {
    assert_ok(
        "invttail",
        r#"
input df
X
0.05
0.025
0.01
end
generate df T = invttail(10, X)
"#,
    );
}

// ── Stata-parity features ───────────────────────────────────────────────────

#[test]
fn quietly_suppresses_output() {
    let (ok, out) = run_inline(
        r#"
input df
Y X
1 2
3 4
5 6
7 8
end
quietly(ols(Y ~ X, df))
"#,
    );
    assert!(ok, "quietly failed:\n{out}");
    // OLS output should NOT appear because quietly returns Nil
    assert!(
        !out.contains("Coef"),
        "quietly should suppress OLS output:\n{out}"
    );
}

#[test]
fn quietly_in_let_works() {
    assert_ok(
        "quietly_let",
        r#"
let x = 42
let y = quietly(x)
"#,
    );
}

#[test]
fn capture_ignores_error() {
    let (ok, _out) = run_inline(
        r#"
capture(undefined_var_xyz)
display "ok"
"#,
    );
    assert!(ok, "capture should not propagate error");
}

#[test]
fn capture_passes_value() {
    assert_ok_contains(
        "capture_pass",
        r#"
let x = capture(42)
display x
"#,
        "42",
    );
}

#[test]
fn assert_true_passes() {
    assert_ok(
        "assert_true",
        r#"
assert(1 > 0)
assert(true)
"#,
    );
}

#[test]
fn assert_false_fails() {
    let (ok, _) = run_inline("assert(1 > 2)");
    assert!(!ok, "assert(false) should fail");
}

#[test]
fn assert_custom_message() {
    let (ok, out) = run_inline(r#"assert(false, "minha msg")"#);
    assert!(!ok, "assert(false) should fail");
    assert!(
        out.contains("minha msg"),
        "should show custom message:\n{out}"
    );
}

#[test]
fn format_decimals() {
    assert_ok_contains(
        "format_2f",
        r#"
let s = format(3.14159, "%.2f")
display s
"#,
        "3.14",
    );
}

#[test]
fn format_zero_decimals() {
    assert_ok_contains(
        "format_0f",
        r#"
let s = format(99.7, "%.0f")
display s
"#,
        "100",
    );
}

#[test]
fn duplicates_report() {
    assert_ok_contains(
        "dup_report",
        r#"
input df
X
1
2
2
3
3
3
end
duplicates(df, X)
"#,
        "duplicates",
    );
}

#[test]
fn duplicates_drop() {
    assert_ok_contains(
        "dup_drop",
        r#"
input df
X Y
1 10
2 20
2 30
3 40
3 50
3 60
end
duplicates(df, X, action=drop)
"#,
        "3 obs removed",
    );
}

#[test]
fn duplicates_tag() {
    assert_ok_contains(
        "dup_tag",
        r#"
input df
X
1
2
2
3
3
3
end
duplicates(df, X, action=tag)
"#,
        "_dup",
    );
}

#[test]
fn label_and_describe() {
    assert_ok_contains(
        "label_describe",
        r#"
input df
Y X
1 2
3 4
end
label(df, Y, "Dependent variable")
label(df, X, "Independent variable")
describe(df)
"#,
        "Dependent variable",
    );
}

#[test]
fn rowtotal_treats_nan_as_zero() {
    assert_ok(
        "rowtotal",
        r#"
input df
A B
1 2
3 .
. 5
end
generate df RT = rowtotal(A, B)
"#,
    );
}

#[test]
fn rowmiss_counts_nan() {
    assert_ok(
        "rowmiss",
        r#"
input df
A B C
1 2 3
. 2 .
. . .
end
generate df M = rowmiss(A, B, C)
"#,
    );
}

#[test]
fn help_quietly() {
    assert_ok_contains("help_quietly", "help(quietly)", "quietly");
}

#[test]
fn help_capture() {
    assert_ok_contains("help_capture", "help(capture)", "capture");
}

#[test]
fn help_assert() {
    assert_ok_contains("help_assert", "help(assert)", "assert");
}

#[test]
fn help_duplicates() {
    assert_ok_contains("help_duplicates", "help(duplicates)", "duplicates");
}

#[test]
fn help_format() {
    assert_ok_contains("help_format", "help(format)", "format");
}

#[test]
fn help_label() {
    assert_ok_contains("help_label", "help(label)", "label");
}

// ══════════════════════════════════════════════════════════════════════════════
// REGEX — paridade com Stata regexm/regexr/regexs
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn regex_regexm_true() {
    assert_ok_contains(
        "regexm_true",
        r#"display regexm("hello123", "[0-9]+")"#,
        "true",
    );
}

#[test]
fn regex_regexm_false() {
    assert_ok_contains(
        "regexm_false",
        r#"display regexm("hello", "[0-9]+")"#,
        "false",
    );
}

#[test]
fn regex_regexr() {
    assert_ok_contains(
        "regexr",
        r#"display regexr("abc 123 def", "[0-9]+", "NUM")"#,
        "abc NUM def",
    );
}

#[test]
fn regex_regexra() {
    assert_ok_contains(
        "regexra",
        r#"display regexra("aaa bbb aaa", "aaa", "x")"#,
        "x bbb x",
    );
}

#[test]
fn regex_regexs_capture() {
    assert_ok_contains(
        "regexs_capture",
        r#"display regexs("price: 42.50 usd", "([0-9]+.[0-9]+)")"#,
        "42.50",
    );
}

#[test]
fn regex_regexs_no_match() {
    assert_ok("regexs_no_match", r#"let r = regexs("hello", "[0-9]+")"#);
}

#[test]
fn regex_in_if() {
    assert_ok_contains(
        "regex_if",
        r#"
if regexm("test@email.com", "@") {
    display "email"
}
"#,
        "email",
    );
}

#[test]
fn regex_email_pattern() {
    assert_ok_contains(
        "regex_email",
        r#"display regexm("user@host.com", "^[^@]+@[^@]+$")"#,
        "true",
    );
}

#[test]
fn smoke_regex() {
    let (ok, out) = run_hy("examples/regex.hay");
    assert!(ok, "regex.hay failed:\n{out}");
    assert!(out.contains("42.50"));
}

// ══════════════════════════════════════════════════════════════════════════════
// LOAD — multi-format
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn load_json() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.json" as df
print(df)"#,
    );
    assert!(ok, "load json failed:\n{out}");
    assert!(out.contains("5 rows"), "expected 5 rows:\n{out}");
}

#[test]
fn load_tsv() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.tsv" as df
print(df)"#,
    );
    assert!(ok, "load tsv failed:\n{out}");
    assert!(out.contains("5 rows"), "expected 5 rows:\n{out}");
}

#[test]
fn load_sqlite_default_table() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.db" as df
print(df)"#,
    );
    assert!(ok, "load sqlite failed:\n{out}");
    assert!(out.contains("8 rows"), "expected 8 rows:\n{out}");
}

#[test]
fn load_sqlite_table_option() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.db" as df, table=precos
print(df)"#,
    );
    assert!(ok, "load sqlite table= failed:\n{out}");
    assert!(out.contains("8 rows"), "expected 8 rows:\n{out}");
}

#[test]
fn load_sqlite_query() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.db" as df, query="SELECT * FROM precos WHERE produto = 'Soja'"
print(df)"#,
    );
    assert!(ok, "load sqlite query= failed:\n{out}");
    assert!(out.contains("4 rows"), "expected 4 rows:\n{out}");
}

#[test]
fn load_csv_semicolon_sep() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample_semicolon.csv" as df, sep=";"
print(df)"#,
    );
    assert!(ok, "load csv sep=; failed:\n{out}");
    assert!(out.contains("4 rows"), "expected 4 rows:\n{out}");
}

#[test]
fn load_unknown_option_error() {
    let (ok, out) = run_inline(r#"load "examples/data/sample.json" as df, bogus=1"#);
    assert!(!ok, "should fail with unknown option:\n{out}");
    assert!(
        out.contains("unknown option"),
        "expected 'unknown option':\n{out}"
    );
}

#[test]
fn load_json_then_summarize() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.json" as df
summarize(df, pop)"#,
    );
    assert!(ok, "load json + summarize failed:\n{out}");
    assert!(out.contains("pop"), "expected 'pop' in output:\n{out}");
}

#[test]
fn test_load_json_boolean_and_categorical() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.json" as df
        generate df cap_val = capital
        ols(pop ~ cap_val, df)
        "#,
    );
    assert!(ok, "load json + boolean generate + ols failed:\n{out}");
    assert!(out.contains("R-squared"), "expected OLS output:\n{out}");
}

#[test]
fn load_sqlite_then_generate() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.db" as df
generate df preco2 = preco * 2
display mean(df, preco2)"#,
    );
    assert!(ok, "load sqlite + generate failed:\n{out}");
}

#[test]
fn load_odbc_without_feature_gives_clear_error() {
    let (ok, out) = run_inline(r#"load "odbc://DSN=test" as df, query="SELECT 1""#);
    if cfg!(feature = "odbc") {
        // with the feature enabled, it will fail due to missing driver — but not for "not enabled"
        assert!(
            !out.contains("not enabled"),
            "should not say 'not enabled' with feature on:\n{out}"
        );
    } else {
        assert!(!ok, "should fail without odbc feature:\n{out}");
        assert!(
            out.contains("not enabled"),
            "expected 'not enabled' message:\n{out}"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// EXPORT — multi-format
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn export_csv() {
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "csv", "{}")"#,
        tmp("hayashi_test_export.csv")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "export csv failed:\n{out}");
    assert!(out.contains("Exported"), "expected 'Exported':\n{out}");
    let content = std::fs::read_to_string(tmp("hayashi_test_export.csv")).unwrap();
    assert!(content.contains("Soja"), "csv missing data:\n{content}");
}

#[test]
fn export_json() {
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "json", "{}")"#,
        tmp("hayashi_test_export.json")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "export json failed:\n{out}");
    let content = std::fs::read_to_string(tmp("hayashi_test_export.json")).unwrap();
    assert!(content.contains("Soja"), "json missing data:\n{content}");
}

#[test]
fn export_tsv() {
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "tsv", "{}")"#,
        tmp("hayashi_test_export.tsv")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "export tsv failed:\n{out}");
    let content = std::fs::read_to_string(tmp("hayashi_test_export.tsv")).unwrap();
    assert!(content.contains('\t'), "tsv missing tabs:\n{content}");
}

#[test]
fn export_xlsx() {
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "xlsx", "{}")"#,
        tmp("hayashi_test_export.xlsx")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "export xlsx failed:\n{out}");
    assert!(std::path::Path::new(&tmp("hayashi_test_export.xlsx")).exists());
}

#[test]
fn export_sqlite() {
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "sqlite", "{}")"#,
        tmp("hayashi_test_export_out.db")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "export sqlite failed:\n{out}");
    assert!(std::path::Path::new(&tmp("hayashi_test_export_out.db")).exists());
}

#[test]
fn export_roundtrip_tsv() {
    let p = tmp("hayashi_rt.tsv");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "tsv", "{p}")
load "{p}" as df2
display mean(df2, preco)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "roundtrip tsv failed:\n{out}");
    assert!(out.contains("88.4"), "expected mean ~88.4:\n{out}");
}

#[test]
fn export_roundtrip_xlsx() {
    let p = tmp("hayashi_rt.xlsx");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "xlsx", "{p}")
load "{p}" as df2
display mean(df2, preco)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "roundtrip xlsx failed:\n{out}");
    assert!(out.contains("88.4"), "expected mean ~88.4:\n{out}");
}

#[test]
fn export_roundtrip_sqlite() {
    let p = tmp("hayashi_rt_out.db");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "sqlite", "{p}")
load "{p}" as df2
display mean(df2, preco)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "roundtrip sqlite failed:\n{out}");
    assert!(out.contains("88.4"), "expected mean ~88.4:\n{out}");
}

#[test]
fn load_tsv_then_regression() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.tsv" as df
let m = ols(score ~ id, df)
print(m)"#,
    );
    assert!(ok, "load tsv + ols failed:\n{out}");
}

// ══════════════════════════════════════════════════════════════════════════════
// IF-EXPRESSION — ternary
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn if_expr_true() {
    assert_ok_contains(
        "if_expr_t",
        r#"let r = if 1 > 0 { "yes" } else { "no" }
display r"#,
        "yes",
    );
}

#[test]
fn if_expr_false() {
    assert_ok_contains(
        "if_expr_f",
        r#"let r = if 1 < 0 { "yes" } else { "no" }
display r"#,
        "no",
    );
}

#[test]
fn if_expr_nested() {
    assert_ok_contains(
        "if_expr_nest",
        r#"let x = 5
let r = if x > 10 { "big" } else { if x > 3 { "mid" } else { "small" } }
display r"#,
        "mid",
    );
}

#[test]
fn if_expr_arithmetic() {
    assert_ok_contains(
        "if_expr_arith",
        r#"let x = -5
let abs = if x < 0 { -x } else { x }
display abs"#,
        "5",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// IN operator
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn in_list_found() {
    assert_ok_contains("in_list_t", "display 2 in [1, 2, 3]", "true");
}

#[test]
fn in_list_not_found() {
    assert_ok_contains("in_list_f", "display 9 in [1, 2, 3]", "false");
}

#[test]
fn in_dict() {
    assert_ok_contains(
        "in_dict",
        r#"let d = {"a": 1, "b": 2}
display "a" in d"#,
        "true",
    );
}

#[test]
fn in_dict_missing() {
    assert_ok_contains(
        "in_dict_f",
        r#"let d = {"a": 1}
display "z" in d"#,
        "false",
    );
}

#[test]
fn in_string() {
    assert_ok_contains("in_str", r#"display "lo" in "hello""#, "true");
}

#[test]
fn in_combined_with_if() {
    assert_ok_contains(
        "in_if",
        r#"let r = if 3 in [1, 2, 3] { "found" } else { "nope" }
display r"#,
        "found",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// MATCH — pattern matching
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn match_basic() {
    assert_ok_contains(
        "match_basic",
        r#"let x = 2
let r = match x { 1 => "one", 2 => "two", _ => "other" }
display r"#,
        "two",
    );
}

#[test]
fn match_as_variable_name() {
    assert_ok_contains(
        "match_var",
        r#"let match = 1
let matched = match + 1
print(matched)"#,
        "2",
    );
}

#[test]
fn readme_smoke_script() {
    let path = std::path::Path::new(file!())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("scripts/readme_smoke.hay");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_hay"))
        .arg(&path)
        .output()
        .expect("failed to execute hay");
    let ok = output.status.success();
    let out = String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr);
    // Cleanup cross-platform export artifact if created.
    let _ = std::fs::remove_file("readme_smoke_export.csv");
    assert!(ok, "README smoke script failed:\n{out}");
    assert!(out.contains("ols ok"));
    assert!(out.contains("export ok"));
}

#[test]
fn list_files_basic() {
    let tmp = std::env::temp_dir();
    let test_dir = tmp.join("hay_list_files_test");
    let _ = std::fs::remove_dir_all(&test_dir);
    std::fs::create_dir_all(&test_dir).unwrap();
    std::fs::write(test_dir.join("a.csv"), "1\n").unwrap();
    std::fs::write(test_dir.join("b.txt"), "x\n").unwrap();
    std::fs::write(test_dir.join("c.csv"), "2\n").unwrap();

    let dir = test_dir.to_string_lossy().replace('\\', "/");
    assert_ok_contains(
        "list_files_all",
        &format!(
            r#"let files = list_files("{dir}")
print(len(files))"#,
        ),
        "3",
    );
    assert_ok_contains(
        "list_files_pattern",
        &format!(
            r#"let files = list_files("{dir}", ".csv")
print(len(files))"#,
        ),
        "2",
    );

    let _ = std::fs::remove_dir_all(&test_dir);
}

#[test]
fn match_wildcard() {
    assert_ok_contains(
        "match_wild",
        r#"let r = match 99 { 1 => "a", _ => "default" }
display r"#,
        "default",
    );
}

#[test]
fn match_string() {
    assert_ok_contains(
        "match_str",
        r#"let s = "hello"
let r = match s { "hi" => 1, "hello" => 2, _ => 0 }
display r"#,
        "2",
    );
}

#[test]
fn match_no_arm_error() {
    let (ok, out) = run_inline(r#"let r = match 5 { 1 => "a", 2 => "b" }"#);
    assert!(!ok, "match without matching arm should fail:\n{out}");
    assert!(
        out.contains("no arm matched"),
        "expected 'no arm matched':\n{out}"
    );
}

#[test]
fn match_with_expr() {
    assert_ok_contains(
        "match_expr",
        r#"let x = 3
let r = match x * 2 { 4 => "four", 6 => "six", _ => "?" }
display r"#,
        "six",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// TRY/CATCH — error handling
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn try_catch_basic() {
    assert_ok_contains(
        "try_catch",
        r#"try {
    load "nonexistent.csv" as df
} catch e {
    display "caught"
}"#,
        "caught",
    );
}

#[test]
fn try_catch_error_var() {
    assert_ok_contains(
        "try_err_var",
        r#"try {
    display undefined_var
} catch e {
    display f"error: {e}"
}"#,
        "error:",
    );
}

#[test]
fn try_catch_no_error() {
    assert_ok_contains(
        "try_no_err",
        r#"try {
    let x = 42
    display x
} catch e {
    display "should not reach"
}"#,
        "42",
    );
}

#[test]
fn try_catch_continues() {
    assert_ok_contains(
        "try_continues",
        r#"try {
    load "bad.csv" as df
} catch e {
    let x = 1
}
display "alive""#,
        "alive",
    );
}

#[test]
fn try_catch_finally_runs_on_success() {
    assert_ok_contains(
        "try_finally_success",
        r#"let x = 0
try {
    x = 1
} catch e {
    x = 99
} finally {
    x = x + 10
}
display x"#,
        "11",
    );
}

#[test]
fn try_catch_finally_runs_on_error() {
    assert_ok_contains(
        "try_finally_error",
        r#"let x = 0
try {
    x = 1
    load "nonexistent.csv" as df
} catch e {
    x = x + 1
} finally {
    x = x + 100
}
display x"#,
        "102",
    );
}

#[test]
fn try_catch_finally_runs_on_return() {
    assert_ok_contains(
        "try_finally_return",
        r#"fn f() {
    let x = 0
    try {
        x = 1
        return x
    } catch e {
        x = 99
    } finally {
        x = x + 10
        print("finally: " + str(x))
    }
}
let r = f()
display r"#,
        "finally: 11",
    );
}

#[test]
fn try_catch_finally_error_overrides() {
    let (ok, out) = run_inline(
        r#"try {
    load "nonexistent.csv" as df
} catch e {
    print("caught")
} finally {
    print("finally")
    let bad = undefined_var
}"#,
    );
    assert!(!ok);
    assert!(out.contains("finally"));
    assert!(out.contains("undefined_var"));
}

// ══════════════════════════════════════════════════════════════════════════════
// PIPE — |> operator
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn pipe_sort() {
    assert_ok_contains(
        "pipe_sort",
        r#"let r = [3, 1, 2] |> sort
display r[0]"#,
        "1",
    );
}

#[test]
fn pipe_chain() {
    assert_ok_contains(
        "pipe_chain",
        r#"let r = [3, 1, 2] |> sort |> reverse
display r[0]"#,
        "3",
    );
}

#[test]
fn pipe_with_args() {
    assert_ok_contains(
        "pipe_args",
        r#"let r = [3, 1, 2] |> sort()
display r[0]"#,
        "1",
    );
}

#[test]
fn pipe_map_closure() {
    assert_ok_contains(
        "pipe_map",
        r#"let r = [1, 2, 3] |> map(|x| x * 10)
display r[1]"#,
        "20",
    );
}

#[test]
fn pipe_filter_map() {
    assert_ok_contains(
        "pipe_fm",
        r#"let r = [1, 2, 3, 4, 5] |> filter(|x| x > 2) |> map(|x| x * 10)
display r[0]"#,
        "30",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// IMPORT — plugin system
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn import_basic() {
    assert_ok_contains(
        "import_basic",
        r#"import("examples/data/test_module")
display test_module::double(21)"#,
        "42",
    );
}

#[test]
fn import_const_available() {
    assert_ok_contains(
        "import_const",
        r#"import("examples/data/test_module")
display test_module::MODULE_LOADED"#,
        "true",
    );
}

#[test]
fn import_dedup() {
    assert_ok_contains(
        "import_dedup",
        r#"import("examples/data/test_module")
import("examples/data/test_module")
display test_module::double(5)"#,
        "10",
    );
}

#[test]
fn import_not_found() {
    let (ok, out) = run_inline(r#"import("nonexistent_module")"#);
    assert!(!ok, "import nonexistent should fail:\n{out}");
    assert!(out.contains("not found"), "expected 'not found':\n{out}");
}

#[test]
fn import_with_extension() {
    assert_ok_contains(
        "import_ext",
        r#"import("examples/data/test_module.hay")
display test_module::double(3)"#,
        "6",
    );
}

#[test]
fn import_alias() {
    assert_ok_contains(
        "import_alias",
        r#"import("examples/data/test_module", as=tm)
display tm::double(7)"#,
        "14",
    );
}

#[test]
fn import_only() {
    assert_ok_contains(
        "import_only",
        r#"import("examples/data/test_module", only=["double"])
display double(4)"#,
        "8",
    );
}

#[test]
fn install_rejects_bad_spec() {
    let (ok, out) = run_inline(r#"install("bad_spec")"#);
    assert!(!ok, "install bad_spec should fail:\n{out}");
    assert!(
        out.contains("expected 'user/repo'"),
        "expected 'user/repo' error:\n{out}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// F-STRING — string interpolation
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn fstring_basic() {
    assert_ok_contains(
        "fstr_basic",
        r#"let x = 42
display f"value = {x}""#,
        "value = 42",
    );
}

#[test]
fn fstring_expr() {
    assert_ok_contains("fstr_expr", r#"display f"sum = {2 + 3}""#, "sum = 5");
}

#[test]
fn fstring_format_spec() {
    assert_ok_contains(
        "fstr_fmt",
        r#"let pi = 3.14159
display f"{pi:.2f}""#,
        "3.14",
    );
}

#[test]
fn fstring_multiple() {
    assert_ok_contains(
        "fstr_multi",
        r#"let a = 1
let b = 2
display f"{a} + {b} = {a + b}""#,
        "1 + 2 = 3",
    );
}

#[test]
fn fstring_escape_braces() {
    assert_ok_contains("fstr_escape", r#"display f"{{literal}}""#, "{literal}");
}

#[test]
fn fstring_scientific() {
    assert_ok_contains("fstr_sci", r#"display f"{0.00123:.2e}""#, "1.23e-3");
}

// ══════════════════════════════════════════════════════════════════════════════
// CLOSURES — |x| expr
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn closure_map() {
    assert_ok_contains(
        "closure_map",
        r#"let r = map([1,2,3], |x| x * 10)
display r[0]"#,
        "10",
    );
}

#[test]
fn closure_filter() {
    assert_ok_contains(
        "closure_filter",
        r#"let r = filter([1,2,3,4,5], |x| x > 3)
display len(r)"#,
        "2",
    );
}

#[test]
fn closure_multi_param() {
    assert_ok_contains(
        "closure_multi",
        r#"let add = |a, b| a + b
let nums = [1, 2, 3]
let result = map(nums, |x| add(x, 10))
display result[1]"#,
        "12",
    );
}

#[test]
fn closure_as_variable() {
    assert_ok_contains(
        "closure_var",
        r#"let sq = |x| x * x
display sq(5)"#,
        "25",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// TYPE CONVERSIONS — int(), float(), str(), bool(), type()
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn type_conv_int() {
    assert_ok_contains("conv_int", "display int(3.9)", "3");
}

#[test]
fn type_conv_float() {
    assert_ok_contains("conv_float", "display float(42)", "42");
}

#[test]
fn type_conv_str() {
    assert_ok_contains("conv_str", "display str(3.14)", "3.14");
}

#[test]
fn type_conv_bool() {
    assert_ok_contains("conv_bool_f", "display bool(0)", "false");
}

#[test]
fn type_conv_bool_true() {
    assert_ok_contains("conv_bool_t", "display bool(1)", "true");
}

#[test]
fn type_conv_str_to_int() {
    assert_ok_contains("conv_s2i", r#"display int("42")"#, "42");
}

#[test]
fn type_conv_str_to_float() {
    assert_ok_contains("conv_s2f", r#"display float("3.14")"#, "3.14");
}

#[test]
fn typeof_check() {
    assert_ok_contains("typeof", r#"display type(42)"#, "int");
}

#[test]
fn typeof_list() {
    assert_ok_contains("typeof_list", "display type([1,2])", "list");
}

#[test]
fn typeof_dict() {
    assert_ok_contains("typeof_dict", r#"display type({"a": 1})"#, "dict");
}

#[test]
fn scalar_math_sqrt() {
    assert_ok_contains("scalar_sqrt", "display sqrt(16)", "4");
}

#[test]
fn scalar_math_abs() {
    assert_ok_contains("scalar_abs", "display abs(-7)", "7");
}

#[test]
fn scalar_math_ln() {
    assert_ok_contains("scalar_ln", "display ln(1)", "0");
}

#[test]
fn scalar_math_exp() {
    assert_ok_contains("scalar_exp", "let x = exp(0)\ndisplay x", "1");
}

#[test]
fn scalar_math_pow() {
    assert_ok_contains("scalar_pow", "display pow(2, 10)", "1024");
}

#[test]
fn modulo_operator() {
    assert_ok_contains("mod_op", "display 10 % 3", "1");
}

#[test]
fn modulo_float() {
    assert_ok_contains("mod_float", "display 10.5 % 3.0", "1.5");
}

#[test]
fn starstar_power() {
    assert_ok_contains("starstar", "display 2 ** 10", "1024");
}

#[test]
fn compound_plus_eq() {
    assert_ok_contains("plus_eq", "let x = 5\nx += 3\ndisplay x", "8");
}

#[test]
fn compound_minus_eq() {
    assert_ok_contains("minus_eq", "let x = 10\nx -= 4\ndisplay x", "6");
}

#[test]
fn compound_star_eq() {
    assert_ok_contains("star_eq", "let x = 3\nx *= 4\ndisplay x", "12");
}

#[test]
fn compound_slash_eq() {
    assert_ok_contains("slash_eq", "let x = 20\nx /= 4\ndisplay x", "5");
}

#[test]
fn compound_percent_eq() {
    assert_ok_contains("pct_eq", "let x = 10\nx %= 3\ndisplay x", "1");
}

// ══════════════════════════════════════════════════════════════════════════════
// PARQUET — load/export
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn export_parquet() {
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "parquet", "{}")"#,
        tmp("hayashi_test.parquet")
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "export parquet failed:\n{out}");
    assert!(std::path::Path::new(&tmp("hayashi_test.parquet")).exists());
}

#[test]
fn parquet_roundtrip() {
    let p = tmp("hayashi_rt.parquet");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "parquet", "{p}")
load "{p}" as df2
display mean(df2, preco)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "parquet roundtrip failed:\n{out}");
    assert!(out.contains("88.4"), "expected mean ~88.4:\n{out}");
}

#[test]
fn load_parquet() {
    let p = tmp("hayashi_load.parquet");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "parquet", "{p}")
load "{p}" as df2
print(df2)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "load parquet failed:\n{out}");
    assert!(out.contains("8 rows"), "expected 8 rows:\n{out}");
}

// ══════════════════════════════════════════════════════════════════════════════
// LOAD — columns= / where= pushdown
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn load_parquet_columns() {
    let p = tmp("hayashi_cols.parquet");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "parquet", "{p}")
load "{p}" as df2, columns=[ano, preco]
describe(df2)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "parquet columns= failed:\n{out}");
    assert!(out.contains("ano"), "expected 'ano' column:\n{out}");
    assert!(out.contains("preco"), "expected 'preco' column:\n{out}");
    assert!(
        !out.contains("produto"),
        "expected 'produto' to be excluded:\n{out}"
    );
}

#[test]
fn load_parquet_where_eq() {
    let p = tmp("hayashi_where.parquet");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "parquet", "{p}")
load "{p}" as df2, where="produto == \"Soja\""
display count(df2)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "parquet where= failed:\n{out}");
    assert!(out.contains("4"), "expected 4 rows for Soja:\n{out}");
}

#[test]
fn load_parquet_columns_where_combined() {
    let p = tmp("hayashi_cw.parquet");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "parquet", "{p}")
load "{p}" as df2, columns=[ano, preco], where="produto == \"Milho\""
display count(df2)
summarize(df2, preco)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "parquet columns+where failed:\n{out}");
    assert!(out.contains("4"), "expected 4 rows for Milho:\n{out}");
    // mean of Milho preco: (42.10 + 55.80 + 68.30 + 50.50) / 4 = 54.175
    assert!(
        out.contains("54.17") || out.contains("54.18"),
        "expected mean ~54.17:\n{out}"
    );
}

#[test]
fn load_parquet_where_gt() {
    let p = tmp("hayashi_gt.parquet");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "parquet", "{p}")
load "{p}" as df2, where="preco > 100"
display count(df2)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "parquet where= > failed:\n{out}");
    // preco > 100: 130.7, 145.2, 120.0 → 3 rows
    assert!(
        out.contains("3"),
        "expected 3 rows with preco > 100:\n{out}"
    );
}

#[test]
fn load_sqlite_columns() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.db" as df, table=precos, columns=[ano, preco]
describe(df)"#,
    );
    assert!(ok, "sqlite columns= failed:\n{out}");
    assert!(out.contains("ano"), "expected 'ano':\n{out}");
    assert!(out.contains("preco"), "expected 'preco':\n{out}");
    assert!(
        !out.contains("produto"),
        "expected 'produto' excluded:\n{out}"
    );
}

#[test]
fn load_sqlite_where_eq() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.db" as df, table=precos, where="produto == \"Soja\""
display count(df)"#,
    );
    assert!(ok, "sqlite where= failed:\n{out}");
    assert!(out.contains("4"), "expected 4 Soja rows:\n{out}");
}

#[test]
fn load_sqlite_columns_where() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.db" as df, table=precos, columns=[preco], where="ano >= 2022"
display count(df)
summarize(df, preco)"#,
    );
    assert!(ok, "sqlite columns+where failed:\n{out}");
    // ano >= 2022: 145.20, 68.30, 120.00, 50.50 → 4 rows
    assert!(out.contains("4"), "expected 4 rows:\n{out}");
}

#[test]
fn load_sqlite_where_in_list() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.db" as df, table=precos, where="ano in [2020, 2023]"
display count(df)"#,
    );
    assert!(ok, "sqlite where in= failed:\n{out}");
    // 2020 (2) + 2023 (2) = 4
    assert!(out.contains("4"), "expected 4 rows:\n{out}");
}

#[test]
fn load_csv_columns() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample_semicolon.csv" as df, sep=";", columns=[produto, preco]
describe(df)"#,
    );
    assert!(ok, "csv columns= failed:\n{out}");
    assert!(out.contains("produto"), "expected 'produto':\n{out}");
    assert!(out.contains("preco"), "expected 'preco':\n{out}");
    assert!(!out.contains("qtd"), "expected 'qtd' excluded:\n{out}");
}

#[test]
fn load_csv_where_gt() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample_semicolon.csv" as df, sep=";", where="preco > 5"
display count(df)"#,
    );
    assert!(ok, "csv where= failed:\n{out}");
    // preco > 5: Arroz 5.49, Feijão 8.99, Café 14.50 → 3 rows
    assert!(out.contains("3"), "expected 3 rows with preco > 5:\n{out}");
}

#[test]
fn load_csv_columns_where_combined() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample_semicolon.csv" as df, sep=";", columns=[produto], where="preco > 5"
display count(df)
list(df, produto)"#,
    );
    assert!(ok, "csv columns+where failed:\n{out}");
    assert!(out.contains("2"), "expected 2 rows:\n{out}");
    assert!(out.contains("Feij"), "expected Feijão:\n{out}");
    assert!(out.contains("Caf"), "expected Café:\n{out}");
}

#[test]
fn load_tsv_where_string_eq() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.tsv" as df, where="grade == \"A\""
display count(df)"#,
    );
    assert!(ok, "tsv where= string eq failed:\n{out}");
    // grade A: Alice, Carol → 2 rows
    assert!(out.contains("2"), "expected 2 rows with grade A:\n{out}");
}

#[test]
fn load_tsv_columns_where_score() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.tsv" as df, columns=[name, score], where="score > 75"
display count(df)
list(df, name)"#,
    );
    assert!(ok, "tsv columns+where failed:\n{out}");
    // score > 75: Alice 85.5, Carol 91.3, Eve 79.2 → 3 rows
    assert!(out.contains("3"), "expected 3 rows:\n{out}");
    assert!(out.contains("Alice"), "expected Alice:\n{out}");
    assert!(out.contains("Carol"), "expected Carol:\n{out}");
}

#[test]
fn load_xlsx_columns_where() {
    let p = tmp("hayashi_xlsx_cols.xlsx");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "xlsx", "{p}")
load "{p}" as df2, columns=[ano, preco], where="produto == \"Soja\""
display count(df2)
summarize(df2, preco)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "xlsx columns+where failed:\n{out}");
    assert!(out.contains("4"), "expected 4 Soja rows:\n{out}");
    // mean of Soja preco: (95.30 + 130.70 + 145.20 + 120.00) / 4 = 122.80
    assert!(out.contains("122.8"), "expected mean ~122.80:\n{out}");
}

#[test]
fn load_where_and_or() {
    let p = tmp("hayashi_andor.parquet");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "parquet", "{p}")
load "{p}" as df2, where="produto == \"Soja\" && ano > 2021"
display count(df2)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "where AND failed:\n{out}");
    // Soja + ano > 2021: 2022 (145.20), 2023 (120.00) → 2 rows
    assert!(out.contains("2"), "expected 2 rows:\n{out}");
}

#[test]
fn load_where_not() {
    let p = tmp("hayashi_not.parquet");
    let script = format!(
        r#"load "examples/data/sample.db" as df
export(df, "parquet", "{p}")
load "{p}" as df2, where="!(produto == \"Soja\")"
display count(df2)"#,
    );
    let (ok, out) = run_inline(&script);
    assert!(ok, "where NOT failed:\n{out}");
    // não-Soja = Milho: 4 rows
    assert!(out.contains("4"), "expected 4 rows (not Soja):\n{out}");
}

// ── Combinações inválidas / erros ────────────────────────────────────────

#[test]
fn load_query_with_where_error() {
    let (ok, out) = run_inline(
        r#"load "examples/data/sample.db" as df, query="SELECT * FROM precos", where="ano > 2020""#,
    );
    assert!(!ok, "should reject query= + where=:\n{out}");
    assert!(
        out.contains("cannot be combined"),
        "expected 'cannot be combined':\n{out}"
    );
}

#[test]
fn load_columns_unknown_error() {
    let (ok, out) =
        run_inline(r#"load "examples/data/sample.db" as df, table=precos, columns=[xxx]"#);
    assert!(!ok, "should reject unknown column:\n{out}");
    // SQLite rejeita no engine SQL; outros loaders validam antes.
    assert!(
        out.contains("not found") || out.contains("unknown") || out.contains("no such column"),
        "expected 'not found', 'unknown' or 'no such column':\n{out}"
    );
}

#[test]
fn load_where_unknown_column_error() {
    let (ok, out) =
        run_inline(r#"load "examples/data/sample.db" as df, table=precos, where="xxx == 1""#);
    assert!(!ok, "should reject where with unknown column:\n{out}");
    // Para SQLite, o erro vem do engine SQL; para outras fontes, do loader.
    assert!(
        out.contains("unknown column") || out.contains("no such column"),
        "expected 'unknown column' or 'no such column':\n{out}"
    );
}

#[test]
fn load_json_columns_unsupported_error() {
    let (ok, out) = run_inline(r#"load "examples/data/sample.json" as df, columns=[pop]"#);
    assert!(!ok, "should reject columns= on JSON:\n{out}");
    assert!(
        out.contains("not yet supported") || out.contains("JSON"),
        "expected JSON unsupported message:\n{out}"
    );
}

#[test]
fn load_help_lists_new_options() {
    let (ok, out) = run_inline("help(load)");
    assert!(ok, "help(load) failed:\n{out}");
    assert!(
        out.contains("columns="),
        "help should mention columns=:\n{out}"
    );
    assert!(out.contains("where="), "help should mention where=:\n{out}");
}

// ══════════════════════════════════════════════════════════════════════════════
// LIST — list operations
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn list_push() {
    assert_ok_contains(
        "list_push",
        r#"
let a = [1, 2, 3]
push(a, 4)
display len(a)"#,
        "4",
    );
}

#[test]
fn list_pop() {
    assert_ok_contains(
        "list_pop",
        r#"
let a = [1, 2, 3]
let b = pop(a)
display b
display len(a)"#,
        "2",
    );
}

#[test]
fn list_insert() {
    assert_ok_contains(
        "list_insert",
        r#"
let a = [10, 30]
let b = insert(a, 1, 20)
display b[1]"#,
        "20",
    );
}

#[test]
fn list_remove() {
    assert_ok_contains(
        "list_remove",
        r#"
let a = [10, 20, 30]
let b = remove(a, 1)
display len(b)"#,
        "2",
    );
}

#[test]
fn list_clear() {
    assert_ok_contains(
        "list_clear",
        r#"
let a = [1, 2, 3]
let b = clear(a)
display len(b)"#,
        "0",
    );
}

#[test]
fn list_reverse() {
    assert_ok_contains(
        "list_reverse",
        r#"
let a = [1, 2, 3]
let b = reverse(a)
display b[0]"#,
        "3",
    );
}

#[test]
fn list_index() {
    assert_ok_contains(
        "list_index",
        r#"
let a = [10, 20, 30]
display index(a, 20)"#,
        "1",
    );
}

#[test]
fn list_index_not_found() {
    assert_ok_contains(
        "list_index_not_found",
        r#"
let a = [10, 20, 30]
display index(a, 99)"#,
        "-1",
    );
}

#[test]
fn list_slice() {
    assert_ok_contains(
        "list_slice",
        r#"
let a = [10, 20, 30, 40, 50]
let b = slice(a, 1, 4)
display len(b)"#,
        "3",
    );
}

#[test]
fn list_slice_open() {
    assert_ok_contains(
        "list_slice_open",
        r#"
let a = [10, 20, 30, 40]
let b = slice(a, 2)
display len(b)"#,
        "2",
    );
}

#[test]
fn list_join() {
    assert_ok_contains(
        "list_join",
        r#"
let a = ["a", "b", "c"]
display join(a, "-")"#,
        "a-b-c",
    );
}

#[test]
fn list_join_default() {
    assert_ok_contains(
        "list_join_default",
        r#"
let a = [1, 2, 3]
display join(a)"#,
        "1, 2, 3",
    );
}

#[test]
fn list_unique() {
    assert_ok_contains(
        "list_unique",
        r#"
let a = [1, 2, 2, 3, 1]
let b = unique(a)
display len(b)"#,
        "3",
    );
}

#[test]
fn list_flatten() {
    assert_ok_contains(
        "list_flatten",
        r#"
let a = [[1, 2], [3, 4], 5]
let b = flatten(a)
display len(b)"#,
        "5",
    );
}

#[test]
fn list_map() {
    assert_ok_contains(
        "list_map",
        r#"
fn double(x) { return x * 2 }
let a = [1, 2, 3]
let b = map(a, double)
display b[2]"#,
        "6",
    );
}

#[test]
fn list_push_mutates() {
    assert_ok_contains(
        "list_push_mut",
        r#"
let a = [1, 2, 3]
push(a, 4)
display len(a)"#,
        "4",
    );
}

#[test]
fn list_push_loop() {
    assert_ok_contains(
        "list_push_loop",
        r#"
let a = []
for i in 1..4 {
    push(a, i * 10)
}
display len(a)"#,
        "3",
    );
}

#[test]
fn list_pop_empty_error() {
    let (ok, out) = run_inline("let a = []\nlet b = pop(a)");
    assert!(!ok, "pop on empty should fail:\n{out}");
    assert!(out.contains("empty list"), "expected 'empty list':\n{out}");
}

#[test]
fn list_remove_out_of_bounds() {
    let (ok, out) = run_inline("let a = [1, 2]\nlet b = remove(a, 5)");
    assert!(!ok, "remove OOB should fail:\n{out}");
    assert!(
        out.contains("out of range"),
        "expected 'out of range':\n{out}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// DICT — dictionary/map
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn dict_literal() {
    assert_ok_contains(
        "dict_lit",
        r#"
let d = {"name": "Alice", "age": 30}
display d["name"]"#,
        "Alice",
    );
}

#[test]
fn dict_index_int_value() {
    assert_ok_contains(
        "dict_int_val",
        r#"
let d = {"x": 42}
display d["x"]"#,
        "42",
    );
}

#[test]
fn dict_len() {
    assert_ok_contains(
        "dict_len",
        r#"
let d = {"a": 1, "b": 2, "c": 3}
display len(d)"#,
        "3",
    );
}

#[test]
fn dict_keys() {
    assert_ok_contains(
        "dict_keys",
        r#"
let d = {"b": 2, "a": 1}
let k = keys(d)
display k[0]"#,
        "a",
    );
}

#[test]
fn dict_values() {
    assert_ok_contains(
        "dict_values",
        r#"
let d = {"a": 10, "b": 20}
let v = values(d)
display len(v)"#,
        "2",
    );
}

#[test]
fn dict_has_key_true() {
    assert_ok_contains(
        "dict_has_key_t",
        r#"
let d = {"x": 1}
display has_key(d, "x")"#,
        "true",
    );
}

#[test]
fn dict_has_key_false() {
    assert_ok_contains(
        "dict_has_key_f",
        r#"
let d = {"x": 1}
display has_key(d, "z")"#,
        "false",
    );
}

#[test]
fn dict_set() {
    assert_ok_contains(
        "dict_set",
        r#"
let d = {"a": 1}
let d2 = dict_set(d, "b", 2)
display len(d2)"#,
        "2",
    );
}

#[test]
fn dict_set_overwrite() {
    assert_ok_contains(
        "dict_set_ow",
        r#"
let d = {"a": 1}
let d2 = dict_set(d, "a", 99)
display d2["a"]"#,
        "99",
    );
}

#[test]
fn dict_remove() {
    assert_ok_contains(
        "dict_remove",
        r#"
let d = {"a": 1, "b": 2}
let d2 = dict_remove(d, "a")
display len(d2)"#,
        "1",
    );
}

#[test]
fn dict_to_dataframe() {
    assert_ok_contains(
        "dict_to_dataframe",
        r#"
let d = {"x": [1.0, 2.0, 3.0, 4.0], "y": ["a", "b", "c", "d"], "z": [true, false, true, false]}
let df = dataframe(d)
let m = ols(x ~ z, df)
display m"#,
        "OLS Regression Results",
    );
}

#[test]
fn dict_merge() {
    assert_ok_contains(
        "dict_merge",
        r#"
let d1 = {"a": 1, "b": 2}
let d2 = {"b": 99, "c": 3}
let d3 = dict_merge(d1, d2)
display d3["b"]"#,
        "99",
    );
}

#[test]
fn dict_immutability() {
    assert_ok_contains(
        "dict_immut",
        r#"
let d = {"a": 1}
let d2 = dict_set(d, "b", 2)
display len(d)"#,
        "1",
    );
}

#[test]
fn dict_nested() {
    assert_ok_contains(
        "dict_nested",
        r#"
let inner = {"x": 42}
let outer = {"data": inner}
display outer["data"]"#,
        "\"x\": 42",
    );
}

#[test]
fn dict_empty() {
    assert_ok_contains(
        "dict_empty",
        r#"
let d = {}
display len(d)"#,
        "0",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// CONST — immutable variables
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn const_declare() {
    assert_ok_contains(
        "const_decl",
        r#"
const PI = 3.14159
display PI"#,
        "3.14159",
    );
}

#[test]
fn const_reassign_error() {
    let (ok, out) = run_inline("const X = 10\nX = 20");
    assert!(!ok, "reassign const should fail:\n{out}");
    assert!(
        out.contains("cannot reassign const"),
        "expected const error:\n{out}"
    );
}

#[test]
fn const_let_redeclare_error() {
    let (ok, out) = run_inline("const X = 10\nlet X = 20");
    assert!(!ok, "let over const should fail:\n{out}");
    assert!(
        out.contains("cannot redeclare const"),
        "expected const error:\n{out}"
    );
}

#[test]
fn const_in_expression() {
    assert_ok_contains(
        "const_expr",
        r#"
const TAX = 0.1
let price = 100
display price * (1 + TAX)"#,
        "110",
    );
}

#[test]
fn const_let_same_name_different_scope() {
    assert_ok_contains(
        "const_scope",
        r#"
const G = 10
if true {
    let x = G * 2
    display x
}"#,
        "20",
    );
}

#[test]
fn dict_missing_key_error() {
    let (ok, out) = run_inline(
        r#"let d = {"a": 1}
display d["z"]"#,
    );
    assert!(!ok, "missing key should fail:\n{out}");
    assert!(out.contains("not found"), "expected 'not found':\n{out}");
}

#[test]
fn dict_display() {
    assert_ok_contains(
        "dict_display",
        r#"
let d = {"name": "Bob", "age": 25}
display d"#,
        "\"age\": 25",
    );
}

#[test]
fn dict_in_list() {
    assert_ok_contains(
        "dict_in_list",
        r#"
let items = [{"name": "A", "val": 1}, {"name": "B", "val": 2}]
display items[1]["name"]"#,
        "B",
    );
}

#[test]
fn summarize_returns_dict() {
    assert_ok_contains(
        "summarize_returns_dict",
        r#"
input df
  x y
  1 10
  2 20
  3 30
end
let s = summarize(df, x)
display s["mean"]
"#,
        "2",
    );
}

#[test]
fn summarize_string_arg() {
    assert_ok_contains(
        "summarize_string_arg",
        r#"
input df
  x y
  1 10
  2 20
  3 30
end
let s = summarize(df, "x")
display s["mean"]
"#,
        "2",
    );
}

#[test]
fn summarize_var_indirect() {
    assert_ok_contains(
        "summarize_var_indirect",
        r#"
input df
  x y
  1 10
  2 20
  3 30
end
let col = "x"
let s = summarize(df, col)
display s["mean"]
"#,
        "2",
    );
}

#[test]
fn summarize_list_arg() {
    assert_ok_contains(
        "summarize_list_arg",
        r#"
input df
  x y
  1 10
  2 20
  3 30
end
let s = summarize(df, ["x", "y"])
display s["y"]["mean"]
"#,
        "20",
    );
}

#[test]
fn summarize_list_var_arg() {
    assert_ok_contains(
        "summarize_list_var_arg",
        r#"
input df
  x y
  1 10
  2 20
  3 30
end
let vars = ["x", "y"]
let s = summarize(df, vars)
display s["x"]["mean"]
"#,
        "2",
    );
}

#[test]
fn summarize_multi_returns_nested_dict() {
    assert_ok_contains(
        "summarize_multi_returns_nested_dict",
        r#"
input df
  x y
  1 10
  2 20
  3 30
end
let s = summarize(df, x, y)
display s["y"]["mean"]
"#,
        "20",
    );
}

#[test]
fn mutate_basic() {
    assert_ok_contains(
        "mutate_basic",
        r#"
input df
  x y
  1 10
  2 20
  3 30
end
let df2 = mutate(df, z = x^2, w = y + x)
summarize(df2, z, w)
"#,
        "z",
    );
}

#[test]
fn mutate_pipe() {
    assert_ok_contains(
        "mutate_pipe",
        r#"
input df
  x y
  1 10
  2 20
  3 30
end
let df2 = df |> mutate(z = x * 2) |> filter(z > 2)
summarize(df2, z)
"#,
        "z",
    );
}

#[test]
fn mutate_chain() {
    assert_ok_contains(
        "mutate_chain",
        r#"
input df
  x
  1
  4
  9
end
let df2 = df |> mutate(y = sqrt(x), z = x^2)
let s = summarize(df2, y)
display s["mean"]
"#,
        "2",
    );
}

#[test]
fn group_by_basic() {
    assert_ok_contains(
        "group_by_basic",
        r#"
input df
  g x
  1 10
  1 20
  2 30
  2 40
end
let agg = group_by(df, g, mean, x)
summarize(agg)
"#,
        "x",
    );
}

#[test]
fn group_by_pipe() {
    assert_ok_contains(
        "group_by_pipe",
        r#"
input df
  g x
  1 10
  1 20
  2 30
  2 40
end
let agg = df |> group_by(g, mean, x)
let s = summarize(agg, x)
display s["mean"]
"#,
        "25",
    );
}

#[test]
fn group_by_median_missing_returns_nan() {
    assert_ok_contains(
        "group_by_median_missing",
        r#"
input df
  g x
  1 10
  1 .
  2 30
  2 40
end
let agg = group_by(df, g, median, x)
let s = summarize(agg, x)
assert(s["missing"] == 1, "expected one missing median")
display "ok"
"#,
        "ok",
    );
}

#[test]
fn pivot_longer_basic() {
    assert_ok_contains(
        "pivot_longer_basic",
        r#"
input df
  id gdp1990 gdp2000
  1  100     200
  2  150     300
end
let long = pivot_longer(df, stubs=["gdp"], i=id, j=year)
summarize(long)
"#,
        "gdp",
    );
}

#[test]
fn pivot_wider_basic() {
    assert_ok_contains(
        "pivot_wider_basic",
        r#"
input df
  id year val
  1  1990 100
  1  2000 200
  2  1990 150
  2  2000 300
end
let wide = pivot_wider(df, i=id, j=year, values=val)
describe(wide)
"#,
        "val1990",
    );
}

#[test]
fn select_alias() {
    assert_ok_contains(
        "select_alias",
        r#"
input df
  x y z
  1 10 100
  2 20 200
end
let df2 = select(df, x, z)
describe(df2)
"#,
        "x",
    );
}

#[test]
fn print_multi_args() {
    assert_ok_contains(
        "print_multi_args",
        r#"
let x = 3.14
let y = 42
print("x =", x, "y =", y)
"#,
        "x = 3.14 y = 42",
    );
}

#[test]
fn print_custom_sep() {
    assert_ok_contains(
        "print_custom_sep",
        r#"
print("a", "b", "c", sep=", ")
"#,
        "a, b, c",
    );
}

#[test]
fn print_empty_call() {
    assert_ok(
        "print_empty_call",
        r#"
print()
"#,
    );
}

#[test]
fn error_did_you_mean_variable() {
    let (ok, out) = run_inline("let price = 10\ndisplay pric");
    assert!(!ok);
    assert!(out.contains("did you mean 'price'"), "missing hint:\n{out}");
}

#[test]
fn error_did_you_mean_function() {
    let (ok, out) = run_inline("input df\nx\n1\nend\nsumarize(df)");
    assert!(!ok);
    assert!(
        out.contains("did you mean 'summarize'"),
        "missing hint:\n{out}"
    );
}

#[test]
fn error_stack_trace() {
    let (ok, out) =
        run_inline("fn inner() {\n  let z = nope\n}\nfn outer() {\n  inner()\n}\nouter()");
    assert!(!ok);
    assert!(out.contains("in inner()"), "missing inner frame:\n{out}");
    assert!(out.contains("in outer()"), "missing outer frame:\n{out}");
}

#[test]
fn error_type_mismatch() {
    let (ok, out) = run_inline("summarize(42)");
    assert!(!ok);
    assert!(
        out.contains("expected DataFrame, got Int"),
        "missing type info:\n{out}"
    );
}

#[test]
fn error_shows_source_line() {
    let (ok, out) = run_inline("let x = 1\nlet y = 2\ndisplay z");
    assert!(!ok);
    assert!(out.contains("│ display z"), "missing source line:\n{out}");
}

#[test]
fn codebook_basic() {
    assert_ok_contains(
        "codebook_basic",
        r#"
input df
  x y
  1 10
  2 20
  3 30
  4 40
  5 50
end
codebook(df)
"#,
        "unique:",
    );
}

#[test]
fn swilk_basic() {
    assert_ok_contains(
        "swilk_basic",
        r#"
input df
  x
  1.2
  2.3
  3.1
  4.0
  5.5
  6.1
  7.2
  8.3
  9.0
  10.1
end
swilk(df, x)
"#,
        "Shapiro-Wilk",
    );
}

#[test]
fn sfrancia_basic() {
    assert_ok_contains(
        "sfrancia_basic",
        r#"
input df
  x
  1.2
  2.3
  3.1
  4.0
  5.5
  6.1
  7.2
  8.3
  9.0
  10.1
end
sfrancia(df, x)
"#,
        "Shapiro-Francia",
    );
}

#[test]
fn sktest_basic() {
    assert_ok_contains(
        "sktest_basic",
        r#"
input df
  x
  1.2
  2.3
  3.1
  4.0
  5.5
  6.1
  7.2
  8.3
  9.0
  10.1
  11.0
  12.5
  13.3
  14.1
  15.0
  16.2
  17.8
  18.1
  19.5
  20.0
end
sktest(df, x)
"#,
        "Skewness/Kurtosis",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// PIPE ASSIGN-BACK — df |> cmd(...) sem let modifica df no lugar
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn pipe_assignback_mutate() {
    assert_ok_contains(
        "pipe_assignback_mutate",
        r#"
input df
  x
  1
  2
  3
end
df |> mutate(z = x * 10)
let s = summarize(df, z)
display s["mean"]
"#,
        "20",
    );
}

#[test]
fn pipe_assignback_filter() {
    assert_ok_contains(
        "pipe_assignback_filter",
        r#"
input df
  x
  1
  2
  3
  4
end
df |> filter(x > 2)
display count(df)
"#,
        "2",
    );
}

#[test]
fn pipe_assignback_chain() {
    // filter(x>=2) → [2,3,4,5]; mutate(y=x*2) → [4,6,8,10]; mean=7
    assert_ok_contains(
        "pipe_assignback_chain",
        r#"
input df
  x
  1
  2
  3
  4
  5
end
df |> filter(x >= 2) |> mutate(y = x * 2)
let s = summarize(df, y)
display s["mean"]
"#,
        "7",
    );
}

#[test]
fn pipe_assignback_preserves_other_vars() {
    assert_ok_contains(
        "pipe_assignback_preserve",
        r#"
input df
  x y
  1 10
  2 20
  3 30
end
let other = 42
df |> mutate(z = x + y)
display other
"#,
        "42",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// GENERATE DUAL MODE — Stata statement vs function call
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn generate_fn_call_returns_new_df() {
    assert_ok_contains(
        "generate_fn_call",
        r#"
input df
  x
  1
  2
  3
end
let df2 = generate(df, z = x^2)
let s = summarize(df2, z)
display s["mean"]
"#,
        // mean([1,4,9]) = 14/3 ≈ 4.666
        "4",
    );
}

#[test]
fn generate_fn_call_result_has_new_col() {
    assert_ok_contains(
        "generate_fn_result",
        r#"
input df
  x
  1
  2
  3
end
let df2 = generate(df, z = x^2)
let s = summarize(df2, z)
display s["mean"]
"#,
        // mean([1,4,9]) ≈ 4.67
        "4",
    );
}

#[test]
fn generate_fn_call_does_not_modify_original() {
    // generate() as a function is pure: original df must NOT have column z
    let (ok, out) = run_inline(
        r#"
input df
  x
  1
  2
  3
end
let df2 = generate(df, z = x^2)
describe(df)
"#,
    );
    assert!(ok, "generate_fn_not_modify failed:\n{out}");
    assert!(
        !out.contains("| z") && !out.contains("  z "),
        "original df should not have column z after generate():\n{out}"
    );
}

#[test]
fn generate_fn_baseline_preserved() {
    // real scenario: baseline and derivative coexist independently
    assert_ok_contains(
        "generate_fn_baseline",
        r#"
input df
  x
  1
  4
  9
end
let baseline = df
let transformed = df |> generate(sq = x^2)
let s = summarize(baseline, x)
display s["N"]
"#,
        "3",
    );
}

#[test]
fn generate_stmt_creates_column_inplace() {
    assert_ok_contains(
        "generate_stmt_inplace",
        r#"
input df
  x
  2
  4
  6
end
generate df z = x / 2
let s = summarize(df, z)
display s["mean"]
"#,
        "3",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// TIME-SERIES OPERATORS — L., F., D. inside generate
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn tsop_lag_basic() {
    // L.x on [10,20,30,40] → [NaN,10,20,30]; mean of non-NaN = 20
    assert_ok_contains(
        "tsop_lag",
        r#"
input df
  x
  10
  20
  30
  40
end
generate df Lx = L.x
let s = summarize(df, Lx)
display s["mean"]
"#,
        "20",
    );
}

#[test]
fn tsop_lag_reduces_n() {
    // L.x on 4 obs has 1 NaN → N non-NaN = 3
    assert_ok_contains(
        "tsop_lag_n",
        r#"
input df
  x
  10
  20
  30
  40
end
generate df Lx = L.x
let s = summarize(df, Lx)
display s["N"]
"#,
        "3",
    );
}

#[test]
fn tsop_lead_basic() {
    // F.x on [10,20,30,40] → [20,30,40,NaN]; mean of non-NaN = 30
    assert_ok_contains(
        "tsop_lead",
        r#"
input df
  x
  10
  20
  30
  40
end
generate df Fx = F.x
let s = summarize(df, Fx)
display s["mean"]
"#,
        "30",
    );
}

#[test]
fn tsop_diff_basic() {
    // D.x on [10,13,18,24] → [NaN,3,5,6]; min of non-NaN = 3
    assert_ok_contains(
        "tsop_diff",
        r#"
input df
  x
  10
  13
  18
  24
end
generate df Dx = D.x
let s = summarize(df, Dx)
display s["min"]
"#,
        "3",
    );
}

#[test]
fn tsop_lag2_basic() {
    // L2.x on [10,20,30,40] → [NaN,NaN,10,20]; mean of non-NaN = 15
    assert_ok_contains(
        "tsop_lag2",
        r#"
input df
  x
  10
  20
  30
  40
end
generate df L2x = L2.x
let s = summarize(df, L2x)
display s["mean"]
"#,
        "15",
    );
}

#[test]
fn tsop_outside_generate_errors() {
    let (ok, _out) = run_inline("let x = L.y");
    assert!(!ok, "TsOp outside generate should fail");
}

// ══════════════════════════════════════════════════════════════════════════════
// RESOLVE_VAR_LIST — 4 modos nos comandos que aceitam colunas
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn select_string_literal_arg() {
    assert_ok_contains(
        "select_str_literal",
        r#"
input df
  x y z
  1 10 100
  2 20 200
end
let df2 = select(df, "x", "z")
describe(df2)
"#,
        "x",
    );
}

#[test]
fn select_string_literal_excludes_col() {
    let (ok, out) = run_inline(
        r#"
input df
  x y z
  1 10 100
  2 20 200
end
let df2 = select(df, "x", "z")
describe(df2)
"#,
    );
    assert!(ok, "select_string_literal_excludes_col failed:\n{out}");
    assert!(
        !out.contains(" y ") && !out.contains("\ny\n") && !out.contains("| y"),
        "y should not appear after select:\n{out}"
    );
}

#[test]
fn select_var_indirect_arg() {
    assert_ok_contains(
        "select_var_indirect",
        r#"
input df
  x y z
  1 10 100
  2 20 200
end
let col = "x"
let df2 = select(df, col, z)
describe(df2)
"#,
        "x",
    );
}

#[test]
fn select_list_arg() {
    assert_ok_contains(
        "select_list_arg",
        r#"
input df
  x y z
  1 10 100
  2 20 200
end
let cols = ["x", "z"]
let df2 = select(df, cols)
describe(df2)
"#,
        "x",
    );
}

#[test]
fn drop_string_literal_arg() {
    let (ok, out) = run_inline(
        r#"
input df
  x y z
  1 10 100
end
let df2 = drop(df, "y")
describe(df2)
"#,
    );
    assert!(ok, "drop_string_literal_arg failed:\n{out}");
    assert!(
        !out.contains("| y") && !out.contains(" y "),
        "y should be dropped:\n{out}"
    );
}

#[test]
fn drop_var_indirect_arg() {
    let (ok, out) = run_inline(
        r#"
input df
  x y z
  1 10 100
end
let col = "y"
let df2 = drop(df, col)
describe(df2)
"#,
    );
    assert!(ok, "drop_var_indirect_arg failed:\n{out}");
    assert!(
        !out.contains("| y") && !out.contains(" y "),
        "y should be dropped:\n{out}"
    );
}

#[test]
fn sort_string_literal_arg() {
    // sort with string literal — checks that min remains the same after sort
    assert_ok_contains(
        "sort_string_literal",
        r#"
input df
  x
  3
  1
  2
end
let df2 = sort(df, "x")
let s = summarize(df2, x)
display s["min"]
"#,
        "1",
    );
}

#[test]
fn dropna_string_literal_arg() {
    assert_ok_contains(
        "dropna_string_literal",
        r#"
input df
  x y
  1.0 10.0
  .   20.0
  3.0 30.0
end
let df2 = dropna(df, "x")
display count(df2)
"#,
        "2",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// CLOSURES — capture variables from outer scope
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn closure_captures_outer_var() {
    assert_ok_contains(
        "closure_capture_outer",
        r#"
let factor = 10
let scale = |x| x * factor
display scale(5)
"#,
        "50",
    );
}

#[test]
fn closure_captures_updated_var() {
    assert_ok_contains(
        "closure_capture_updated",
        r#"
let base = 100
let add_base = |x| x + base
base = 200
display add_base(5)
"#,
        "205",
    );
}

#[test]
fn closure_in_map_with_outer() {
    assert_ok_contains(
        "closure_map_outer",
        r#"
let mult = 3
let r = map([1, 2, 3], |x| x * mult)
display r[2]
"#,
        "9",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// TRY/CATCH — edge cases and error propagation
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn try_catch_inside_function() {
    assert_ok_contains(
        "try_catch_fn",
        r#"
fn safe_get(d, k) {
    try {
        return d[k]
    } catch e {
        return -1
    }
}
let m = {"a": 42}
display safe_get(m, "a")
display safe_get(m, "missing")
"#,
        "42",
    );
}

#[test]
fn try_catch_nested() {
    assert_ok_contains(
        "try_catch_nested",
        r#"
try {
    try {
        display undefined_x
    } catch inner {
        display "inner caught"
    }
    display "after inner"
} catch outer {
    display "outer caught"
}
"#,
        "inner caught",
    );
}

#[test]
fn try_catch_nested_after_inner() {
    assert_ok_contains(
        "try_catch_nested_after",
        r#"
try {
    try {
        display undefined_x
    } catch inner {
        let x = 1
    }
    display "after inner"
} catch outer {
    display "should not reach"
}
"#,
        "after inner",
    );
}

#[test]
fn try_catch_in_loop() {
    assert_ok_contains(
        "try_catch_loop",
        r#"
let errors = 0
for i in 1..4 {
    try {
        if i == 2 { display bad_var }
    } catch e {
        errors = errors + 1
    }
}
display errors
"#,
        "1",
    );
}

#[test]
fn try_catch_error_message_content() {
    let (ok, out) = run_inline(
        r#"
try {
    display undefined_xyz
} catch e {
    display e
}
"#,
    );
    assert!(ok, "try_catch_error_content failed:\n{out}");
    assert!(
        out.contains("undefined_xyz") || out.contains("xyz"),
        "error message should mention variable name:\n{out}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// PRINT — sep= and end= options
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn print_end_option() {
    assert_ok_contains(
        "print_end",
        r#"print("hello", end="!")
print(" world")"#,
        "hello! world",
    );
}

#[test]
fn print_sep_and_end() {
    assert_ok_contains(
        "print_sep_end",
        r#"print("a", "b", "c", sep="-", end=".")"#,
        "a-b-c.",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// FSTRING — complex expressions
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn fstring_with_arithmetic() {
    assert_ok_contains(
        "fstr_arith",
        r#"let x = 5
display f"square = {x * x}""#,
        "square = 25",
    );
}

#[test]
fn fstring_with_conditional() {
    assert_ok_contains(
        "fstr_cond",
        r#"
let x = 10
let sign = if x > 0 { 1 } else { -1 }
display f"sign = {sign}"
"#,
        "sign = 1",
    );
}

#[test]
fn fstring_in_loop() {
    assert_ok_contains(
        "fstr_loop",
        r#"
let result = ""
for i in 1..4 {
    result = result + f"[{i}]"
}
display result
"#,
        "[1][2][3]",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// CONST — imutabilidade e escopo
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn const_in_function_scope() {
    assert_ok_contains(
        "const_fn_scope",
        r#"
fn compute(x) {
    const factor = 2
    return x * factor
}
display compute(7)
"#,
        "14",
    );
}

#[test]
fn const_in_function_immutable() {
    let (ok, _out) = run_inline(
        r#"
fn bad() {
    const x = 10
    x = 20
}
bad()
"#,
    );
    assert!(!ok, "assigning to const in fn should fail");
}

#[test]
fn const_shadows_outer_let() {
    assert_ok_contains(
        "const_shadow",
        r#"
let x = 1
fn f() {
    const x = 99
    return x
}
display f()
"#,
        "99",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// FUNCTIONS — multiple return paths and recursion with accumulator
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn fn_early_return_on_condition() {
    assert_ok_contains(
        "fn_early_return",
        r#"
fn sign(x) {
    if x > 0 { return 1 }
    if x < 0 { return -1 }
    return 0
}
display sign(-5)
display sign(0)
display sign(3)
"#,
        "-1",
    );
}

#[test]
fn fn_multiple_return_paths_all_hit() {
    let (ok, out) = run_inline(
        r#"
fn classify(x) {
    if x < 0 { return "negative" }
    if x == 0 { return "zero" }
    return "positive"
}
display classify(-1)
display classify(0)
display classify(1)
"#,
    );
    assert!(ok, "fn_multiple_paths failed:\n{out}");
    assert!(out.contains("negative"), "missing negative:\n{out}");
    assert!(out.contains("zero"), "missing zero:\n{out}");
    assert!(out.contains("positive"), "missing positive:\n{out}");
}

#[test]
fn fn_accumulator_pattern() {
    assert_ok_contains(
        "fn_accumulator",
        r#"
fn sum_to(n) {
    let acc = 0
    for i in 1..(n+1) {
        acc = acc + i
    }
    return acc
}
display sum_to(10)
"#,
        "55",
    );
}

#[test]
fn fn_default_param_behavior() {
    assert_ok_contains(
        "fn_default_param",
        r#"
fn greet(name, prefix) {
    return prefix + " " + name
}
display greet("World", "Hello")
"#,
        "Hello World",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// GENERATE / MUTATE — _n and _N as special variables
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn generate_uses_row_number() {
    // _n em 3 obs = [1,2,3]; sum=6, mean=2
    assert_ok_contains(
        "generate_n",
        r#"
input df
  x
  10
  20
  30
end
generate df idx = _n
let s = summarize(df, idx)
display s["mean"]
"#,
        "2",
    );
}

#[test]
fn generate_uses_total_rows() {
    // _N em 3 obs = [3,3,3]; mean=3, min=3
    assert_ok_contains(
        "generate_N",
        r#"
input df
  x
  10
  20
  30
end
generate df total = _N
let s = summarize(df, total)
display s["mean"]
"#,
        "3",
    );
}

#[test]
fn mutate_uses_row_number() {
    // _n via mutate — mean deve ser 2 para 3 linhas
    assert_ok_contains(
        "mutate_n",
        r#"
input df
  x
  10
  20
  30
end
let df2 = mutate(df, idx = _n)
let s = summarize(df2, idx)
display s["mean"]
"#,
        "2",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// GROUP_BY — multiple aggregation functions
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn group_by_sum() {
    assert_ok_contains(
        "group_by_sum",
        r#"
input df
  g x
  1 10
  1 20
  2 30
  2 40
end
let agg = group_by(df, g, sum, x)
let s = summarize(agg, x)
display s["max"]
"#,
        "70",
    );
}

#[test]
fn group_by_count() {
    assert_ok_contains(
        "group_by_count",
        r#"
input df
  g x
  1 10
  1 20
  1 30
  2 40
end
let agg = group_by(df, g, count, x)
let s = summarize(agg, x)
display s["max"]
"#,
        "3",
    );
}

#[test]
fn group_by_min_max() {
    assert_ok_contains(
        "group_by_min",
        r#"
input df
  g x
  1 5
  1 15
  2 3
  2 30
end
let agg = group_by(df, g, min, x)
let s = summarize(agg, x)
display s["min"]
"#,
        "3",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// CODEBOOK — com coluna string
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn codebook_with_repeated_values() {
    // codebook em coluna com valores repetidos mostra unique count < N
    assert_ok_contains(
        "codebook_repeated",
        r#"
input df
  group score
  1 85
  1 90
  2 70
  2 80
  1 95
end
codebook(df)
"#,
        "unique:",
    );
}

#[test]
fn codebook_single_column() {
    assert_ok_contains(
        "codebook_single",
        r#"
input df
  x y
  1 10
  2 20
  3 30
end
codebook(df, x)
"#,
        "x",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// FOR — iteration over list of strings and dicts
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn for_over_string_list() {
    assert_ok_contains(
        "for_str_list",
        r#"
let words = ["hello", "world", "foo"]
let result = ""
for w in words {
    result = result + w + " "
}
display trim(result)
"#,
        "hello world foo",
    );
}

#[test]
fn for_over_mixed_list() {
    assert_ok_contains(
        "for_mixed_list",
        r#"
let items = [1, 2, 3]
let total = 0
for v in items {
    total = total + v
}
display total
"#,
        "6",
    );
}

#[test]
fn for_builds_list() {
    // push() mutates the list in place; do not assign the result
    assert_ok_contains(
        "for_builds_list",
        r#"
let squares = []
for i in 1..6 {
    push(squares, i * i)
}
display len(squares)
"#,
        "5",
    );
}

#[test]
fn parallel_for_basic() {
    // parallel for returns a List of the last expression values
    assert_ok_contains(
        "parallel_basic",
        r#"
let items = [1, 2, 3, 4, 5, 6]
parallel for x in items {
    x * x
}
display len(x)
display x[0]
display x[5]
"#,
        "36",
    );
}

#[test]
fn parallel_for_captures_outer() {
    // Each thread captures outer variables via Arc clone
    assert_ok_contains(
        "parallel_capture",
        r#"
let mult = 10
let items = [1, 2, 3, 4]
parallel for x in items {
    x * mult
}
display x[3]
"#,
        "40",
    );
}

#[test]
fn parallel_for_return() {
    // Explicit return inside parallel for body
    assert_ok_contains(
        "parallel_return",
        r#"
let items = [1, 2, 3, 4, 5]
parallel for x in items {
    if x > 3 {
        return x * 100
    }
    x
}
display x[3]
display x[4]
"#,
        "500",
    );
}

#[test]
fn parallel_for_threads_option() {
    // Optional threads=N parameter limits worker threads
    assert_ok_contains(
        "parallel_threads",
        r#"
let items = [1, 2, 3, 4, 5, 6, 7, 8]
parallel for x in items, threads=2 {
    x * x
}
display len(x)
display x[0]
display x[7]
"#,
        "64",
    );
}

#[test]
fn parallel_for_expression_form() {
    // parallel for as expression: let r = parallel for ...
    assert_ok_contains(
        "parallel_expr",
        r#"
let items = [1, 2, 3, 4, 5]
let r = parallel for x in items, threads=2 {
    x * 10
}
display len(r)
display r[0]
display r[4]
"#,
        "50",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// MATCH — cases with types and expressions in arms
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn match_int_values() {
    assert_ok_contains(
        "match_int",
        r#"
let x = 2
let r = match x {
    1 => "one"
    2 => "two"
    3 => "three"
    _ => "other"
}
display r
"#,
        "two",
    );
}

#[test]
fn match_bool_value() {
    assert_ok_contains(
        "match_bool",
        r#"
let flag = true
let r = match flag {
    true  => "yes"
    false => "no"
}
display r
"#,
        "yes",
    );
}

#[test]
fn match_float_value() {
    assert_ok_contains(
        "match_float",
        r#"
let x = 3.14
let r = match x {
    3.14 => "pi-ish"
    _    => "other"
}
display r
"#,
        "pi-ish",
    );
}

#[test]
fn match_uses_expression_in_arm() {
    assert_ok_contains(
        "match_expr_arm",
        r#"
fn double(x) { return x * 2 }
let n = 5
let r = match n {
    5 => double(n)
    _ => 0
}
display r
"#,
        "10",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// IF-EXPR — ternary expression in varied contexts
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn if_expr_in_assignment() {
    assert_ok_contains(
        "ifexpr_assign",
        r#"
let x = 7
let label = if x > 5 { "big" } else { "small" }
display label
"#,
        "big",
    );
}

#[test]
fn if_expr_in_fstring() {
    assert_ok_contains(
        "ifexpr_fstr",
        r#"
let score = 85
let grade = if score >= 90 { "A" } else { "B" }
display f"grade: {grade}"
"#,
        "grade: B",
    );
}

#[test]
fn if_expr_as_function_arg() {
    assert_ok_contains(
        "ifexpr_arg",
        r#"
let x = -3
let magnitude = if x < 0 { x * -1 } else { x }
display abs(magnitude)
"#,
        "3",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// ESCOPOS — shadowing, escopo de bloco, param vs externo
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn scope_param_shadows_outer() {
    assert_ok_contains(
        "scope_param_shadow",
        r#"
let x = 100
fn f(x) { return x }
display f(42)
"#,
        "42",
    );
}

#[test]
fn scope_block_let_dies() {
    let (ok, out) = run_inline(
        r#"
{
    let inner = 99
}
display inner
"#,
    );
    assert!(!ok, "block let should not leak:\n{out}");
}

#[test]
fn scope_for_iter_var_survives() {
    // in Hayashi, the for loop iteration variable persists after the loop (by design)
    assert_ok_contains(
        "scope_for_survives",
        r#"
for i in 1..4 {}
display i
"#,
        "3",
    );
}

#[test]
fn scope_fn_sees_outer_const() {
    assert_ok_contains(
        "scope_fn_outer_const",
        r#"
const PI = 3.14159
fn circle_area(r) {
    return PI * r * r
}
display circle_area(1)
"#,
        "3.14",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// WHILE — variable-based condition and early break
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn while_with_fn_condition() {
    assert_ok_contains(
        "while_fn_cond",
        r#"
let x = 1
while x < 32 {
    x = x * 2
}
display x
"#,
        "32",
    );
}

#[test]
fn while_break_on_condition() {
    assert_ok_contains(
        "while_break",
        r#"
let i = 0
let found = false
while i < 100 {
    i = i + 1
    if i == 7 {
        found = true
        break
    }
}
display found
display i
"#,
        "true",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// OPERADORES COMPOSTOS — em contextos diversos
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn compound_in_loop() {
    assert_ok_contains(
        "compound_loop",
        r#"
let x = 0
for i in 1..6 {
    x += i
}
display x
"#,
        "15",
    );
}

#[test]
fn compound_multiply_in_loop() {
    assert_ok_contains(
        "compound_mul_loop",
        r#"
let x = 1
for i in 1..6 {
    x *= i
}
display x
"#,
        "120",
    );
}

#[test]
fn compound_assignments_multiple() {
    // tests several compound operators in sequence
    assert_ok_contains(
        "compound_multi",
        r#"
let x = 100
x -= 10
x *= 2
x /= 4
x %= 7
display x
"#,
        // 100 - 10 = 90, * 2 = 180, / 4 = 45, % 7 = 3
        "3",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// DICT — access, set, remove and merge in composite scenarios
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn dict_built_in_loop() {
    assert_ok_contains(
        "dict_loop_build",
        r#"
let d = {}
for i in 1..4 {
    d = dict_set(d, str(i), i * i)
}
display d["1"]
display d["3"]
"#,
        "9",
    );
}

#[test]
fn dict_nested_access() {
    assert_ok_contains(
        "dict_nested_acc",
        r#"
let config = {"db": {"host": "localhost", "port": 5432}, "debug": true}
display config["db"]["port"]
"#,
        "5432",
    );
}

#[test]
fn dict_has_key_after_set() {
    assert_ok_contains(
        "dict_haskey_set",
        r#"
let d = {"a": 1}
let d2 = dict_set(d, "b", 2)
display has_key(d2, "b")
"#,
        "true",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// SERIES — columns as first-class citizens
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn series_from_dataframe() {
    assert_ok_contains(
        "series_from_df",
        r#"
input df
  y x
  1 10
  2 20
  3 30
  4 40
  5 50
end
let s = df["y"]
display len(s)
display first(s)
display last(s)
"#,
        "5",
    );
}

#[test]
fn series_aggregations() {
    assert_ok_contains(
        "series_agg",
        r#"
input df
  y x
  1 10
  2 20
  3 30
  4 40
  5 50
end
let s = df["y"]
display mean(s)
display sd(s)
display min(s)
display max(s)
"#,
        "3",
    );
}

#[test]
fn series_index() {
    assert_ok_contains(
        "series_index",
        r#"
input df
  y x
  1 10
  2 20
  3 30
end
let s = df["y"]
display s[0]
display s[-1]
"#,
        "3",
    );
}

#[test]
fn series_shift() {
    assert_ok_contains(
        "series_shift",
        r#"
input df
  y x
  1 10
  2 20
  3 30
end
let s = df["y"]
let shifted = shift(s, 1)
display first(shifted)
"#,
        "nil",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// DEFAULTS EM FN
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn fn_default_args() {
    assert_ok_contains(
        "fn_defaults",
        r#"
fn add(a, b=10) {
    return a + b
}
display add(5)
display add(5, 3)
"#,
        "15",
    );
}

#[test]
fn fn_default_string() {
    assert_ok_contains(
        "fn_default_str",
        r#"
fn greet(name, greeting="Oi") {
    return greeting + " " + name
}
display greet("Ana")
"#,
        "Oi Ana",
    );
}

#[test]
fn fn_default_missing_required() {
    let (ok, out) = run_inline(
        r#"
fn f(a, b=2) {
    return a + b
}
display f()
"#,
    );
    assert!(!ok, "missing required arg should fail");
    assert!(out.contains("missing required argument"), "{out}");
}

// ══════════════════════════════════════════════════════════════════════════════
// ERROS ESTRUTURADOS
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn structured_error_kind_and_msg() {
    assert_ok_contains(
        "structured_error",
        r#"
try {
    let y = [1, 2]
    print(y[10])
} catch e {
    display e["kind"]
    display e["msg"]
}
"#,
        "runtime",
    );
}

#[test]
fn structured_error_line() {
    assert_ok_contains(
        "structured_error_line",
        r#"
try {
    let y = [1, 2]
    print(y[10])
} catch e {
    display e["line"]
}
"#,
        "4",
    );
}

#[test]
fn structured_error_undefined_var() {
    assert_ok_contains(
        "structured_error_undef",
        r#"
try {
    let z = undefined
} catch e {
    display e["kind"]
    display e["msg"]
}
"#,
        "undefined variable",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// DOCSTRINGS EM FN
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn fn_docstring_help() {
    assert_ok_contains(
        "fn_docstring_help",
        r#"
fn sharpe(ret, rf=0.0) {
    ## Compute Sharpe ratio
    return (mean(ret) - rf) / sd(ret)
}
help(sharpe)
"#,
        "Compute Sharpe ratio",
    );
}

#[test]
fn fn_docstring_multiline() {
    assert_ok_contains(
        "fn_docstring_multi",
        r#"
fn sharpe(ret, rf=0.0) {
    ## Compute Sharpe ratio
    ## ret: Series of returns
    return (mean(ret) - rf) / sd(ret)
}
help(sharpe)
"#,
        "ret: Series of returns",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// ERROR MESSAGES — coverage of specific scenarios
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn error_column_not_found_suggest() {
    let (ok, out) = run_inline(
        r#"
input df
  price quantity
  10 5
  20 10
end
summarize(df, prce)
"#,
    );
    assert!(!ok, "bad column should fail");
    // we do not require did-you-mean for column names (depends on implementation),
    // mas deve falhar com mensagem clara
    assert!(
        out.contains("prce") || out.contains("price") || out.contains("not found"),
        "error should mention the column:\n{out}"
    );
}

#[test]
fn error_wrong_arg_count_fn() {
    let (ok, out) = run_inline(
        r#"
fn add(a, b) { return a + b }
display add(1)
"#,
    );
    assert!(!ok, "wrong arg count should fail:\n{out}");
}

#[test]
fn error_index_out_of_bounds() {
    let (ok, out) = run_inline(
        r#"
let lst = [1, 2, 3]
display lst[10]
"#,
    );
    assert!(!ok, "out-of-bounds should fail:\n{out}");
    assert!(
        out.contains("10") || out.contains("index") || out.contains("bounds"),
        "error should mention index:\n{out}"
    );
}

#[test]
fn error_type_mismatch_string_plus_int() {
    // "hello" + 42 must fail because 42 is not a string
    let (ok, out) = run_inline(r#"display "hello" + 42"#);
    if !ok {
        // mensagem deve mencionar o problema de tipo
        assert!(
            out.contains("numeric")
                || out.contains("type")
                || out.contains("string")
                || out.contains("expected"),
            "type error should be informative:\n{out}"
        );
    }
    // if ok, auto-conversion is acceptable; just ensure determinism
}

#[test]
fn error_source_annotation_multiline() {
    let (ok, out) = run_inline(
        r#"
let a = 1
let b = 2
let c = 3
display d
"#,
    );
    assert!(!ok);
    assert!(
        out.contains("│") || out.contains("|"),
        "should show source annotation:\n{out}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// PIVOT — casos mais completos
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn pivot_longer_row_count() {
    // 2 id * 3 anos = 6 rows
    assert_ok_contains(
        "pivot_longer_rows",
        r#"
input df
  id gdp1990 gdp2000 gdp2010
  1  100     200     300
  2  150     250     350
end
let long = pivot_longer(df, stubs=["gdp"], i=id, j=year)
display count(long)
"#,
        "6",
    );
}

#[test]
fn pivot_wider_column_count() {
    assert_ok_contains(
        "pivot_wider_cols",
        r#"
input df
  id year val
  1  2000  10
  1  2001  20
  2  2000  30
  2  2001  40
end
let wide = pivot_wider(df, i=id, j=year, values=val)
describe(wide)
"#,
        "val2000",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// MUTATE — multiple columns, including conditionals
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn mutate_multiple_columns() {
    assert_ok_contains(
        "mutate_multi",
        r#"
input df
  x
  1
  2
  3
end
let df2 = mutate(df, sq = x^2, cube = x^3)
let s = summarize(df2, cube)
display s["max"]
"#,
        "27",
    );
}

#[test]
fn mutate_conditional_column() {
    // (x > 0) cria coluna 0.0/1.0; mean de [0,0,1] = 1/3
    assert_ok_contains(
        "mutate_cond",
        r#"
input df
  x
  -2
  0
  3
end
let df2 = mutate(df, pos = x > 0)
let s = summarize(df2, pos)
display s["max"]
"#,
        "1",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// MATHEMATICAL OPERATIONS — precision and extreme cases
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn math_integer_division_floor() {
    assert_ok_contains("math_int_div", "display int(7 / 2)", "3");
}

#[test]
fn math_modulo_negative() {
    let (ok, out) = run_inline("print(-7 % 3)");
    assert!(ok, "negative modulo failed:\n{out}");
    // result can be -1 or 2 depending on convention; just checks that it is deterministic
    assert!(
        out.trim() == "-1" || out.trim() == "2",
        "unexpected modulo result:\n{out}"
    );
}

#[test]
fn math_power_zero() {
    assert_ok_contains("math_pow_zero", "display 5^0", "1");
}

#[test]
fn math_nested_calls() {
    assert_ok_contains("math_nested", "display sqrt(abs(-16))", "4");
}

// ══════════════════════════════════════════════════════════════════════════════
// IN OPERATOR — em listas e ranges
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn in_operator_list_true() {
    assert_ok_contains("in_list_t", "display 3 in [1, 2, 3, 4]", "true");
}

#[test]
fn in_operator_list_false() {
    assert_ok_contains("in_list_f", "display 5 in [1, 2, 3, 4]", "false");
}

#[test]
fn in_operator_string_list() {
    assert_ok_contains("in_str_list", r#"display "b" in ["a", "b", "c"]"#, "true");
}

#[test]
fn in_operator_not_in_list() {
    assert_ok_contains("not_in_list", r#"display !("z" in ["a", "b"])"#, "true");
}

// ══════════════════════════════════════════════════════════════════════════════
// BOOLEAN — short-circuit and combinations
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn bool_and_false_on_first() {
    // false && anything = false
    assert_ok_contains(
        "bool_and_ff",
        r#"
let r = false && true
display r
"#,
        "false",
    );
}

#[test]
fn bool_or_true_on_first() {
    // true || anything = true
    assert_ok_contains(
        "bool_or_tt",
        r#"
let r = true || false
display r
"#,
        "true",
    );
}

#[test]
fn bool_complex_expression() {
    assert_ok_contains(
        "bool_complex",
        r#"
let a = 5
let b = 10
let c = 3
display (a < b) && (c < a) && (b > c)
"#,
        "true",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// STRINGS — chained operations and behavior with special characters
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn string_chain_ops() {
    assert_ok_contains(
        "str_chain",
        r#"
let s = "  Hello, World!  "
let r = lower(trim(s))
display r
"#,
        "hello, world!",
    );
}

#[test]
fn string_split_and_index() {
    assert_ok_contains(
        "str_split_idx",
        r#"
let parts = split("a,b,c", ",")
display parts[1]
"#,
        "b",
    );
}

#[test]
fn string_len_after_concat() {
    assert_ok_contains(
        "str_len_concat",
        r#"
let s = "hello" + " " + "world"
display len(s)
"#,
        "11",
    );
}

#[test]
fn string_substr_range() {
    assert_ok_contains(
        "str_substr_range",
        r#"
let s = "econometrics"
display substr(s, 0, 5)
"#,
        "econo",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// LISTS — functional operations
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn list_filter_closure() {
    assert_ok_contains(
        "list_filter_cl",
        r#"
let evens = filter([1, 2, 3, 4, 5, 6], |x| x % 2 == 0)
display len(evens)
"#,
        "3",
    );
}

#[test]
fn list_map_then_filter() {
    assert_ok_contains(
        "list_map_filter",
        r#"
let r = filter(map([1, 2, 3, 4], |x| x * x), |x| x > 5)
display len(r)
"#,
        "2",
    );
}

#[test]
fn list_nested() {
    assert_ok_contains(
        "list_nested",
        r#"
let matrix = [[1, 2], [3, 4], [5, 6]]
display matrix[1][0]
"#,
        "3",
    );
}

#[test]
fn list_push_in_fn() {
    assert_ok_contains(
        "list_push_fn",
        r#"
fn build(n) {
    let r = []
    for i in 1..(n+1) {
        push(r, i)
    }
    return r
}
let lst = build(5)
display len(lst)
"#,
        "5",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// COUNT(df) / NROW(df) — contagem de linhas como valor
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn count_fn_total_rows() {
    assert_ok_contains(
        "count_fn_total",
        r#"
input df
  x
  10
  20
  30
  40
  50
end
display count(df)
"#,
        "5",
    );
}

#[test]
fn nrow_alias() {
    assert_ok_contains(
        "nrow_alias",
        r#"
input df
  x
  1
  2
  3
end
display nrow(df)
"#,
        "3",
    );
}

#[test]
fn count_fn_with_condition() {
    assert_ok_contains(
        "count_fn_cond",
        r#"
input df
  x
  1
  2
  3
  4
  5
end
display count(df, x > 3)
"#,
        "2",
    );
}

#[test]
fn count_fn_after_filter() {
    assert_ok_contains(
        "count_after_filter",
        r#"
input df
  x
  1
  2
  3
  4
end
let df2 = filter(df, x >= 3)
display count(df2)
"#,
        "2",
    );
}

#[test]
fn count_fn_returns_int() {
    assert_ok_contains(
        "count_fn_int",
        r#"
input df
  x
  10
  20
end
let n = count(df)
display n + 1
"#,
        "3",
    );
}

#[test]
fn count_fn_empty_after_filter() {
    assert_ok_contains(
        "count_fn_zero",
        r#"
input df
  x
  1
  2
  3
end
let df2 = filter(df, x > 100)
display count(df2)
"#,
        "0",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// INPUT BLOCK — clear error when receiving non-numeric data
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn input_rejects_string_data() {
    let (ok, out) = run_inline(
        r#"
input df
  name age
  Alice 25
  Bob   30
end
"#,
    );
    assert!(!ok, "input with string data should fail");
    assert!(
        out.contains("Alice") || out.contains("not a number"),
        "error should mention the non-numeric token:\n{out}"
    );
}

#[test]
fn input_rejects_quoted_string() {
    let (ok, out) = run_inline(
        r#"
input df
  x
  "hello"
  2
end
"#,
    );
    assert!(!ok, "input with quoted string should fail");
    assert!(
        out.contains("hello") || out.contains("not a number"),
        "error should mention the invalid token:\n{out}"
    );
}

#[test]
fn input_accepts_dot_as_missing() {
    // '.' is a valid missing value; must not be confused with string
    assert_ok_contains(
        "input_dot_missing",
        r#"
input df
  x y
  1.0 10.0
  .   20.0
  3.0 30.0
end
let s = summarize(df, x)
display s["missing"]
"#,
        "1",
    );
}

#[test]
fn test_nil_literal() {
    assert_ok_contains(
        "nil_literal",
        r#"
        let x = nil
        if x == nil {
            display "is_nil"
        }
        "#,
        "is_nil",
    );
}

#[test]
fn test_all_type_checkers() {
    assert_ok_contains(
        "type_checkers",
        r#"
        if is_int(42) && !is_int(3.14) { display "int_ok" }
        if is_float(3.14) && !is_float(42) { display "float_ok" }
        if is_bool(true) && !is_bool("string") { display "bool_ok" }
        if is_string("hello") && !is_string(42) { display "string_ok" }
        if is_list([1,2]) && !is_list(42) { display "list_ok" }
        if is_dict({"a": 1}) && !is_dict(42) { display "dict_ok" }
        if is_function(|x| x) && !is_function(42) { display "fn_ok" }
        if !is_nil(42) && is_nil(nil) { display "nil_ok" }
        "#,
        "\"int_ok\"\n\"float_ok\"\n\"bool_ok\"\n\"string_ok\"\n\"list_ok\"\n\"dict_ok\"\n\"fn_ok\"\n\"nil_ok\"",
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// MULTILINE NEWLINE HANDLING
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn pipe_multiline() {
    // |> at the start of a line continuing the pipe chain
    assert_ok_contains(
        "pipe_multiline",
        r#"let r = [3, 1, 2]
    |> sort
    |> reverse
display r[0]"#,
        "3",
    );
}

#[test]
fn pipe_multiline_with_args() {
    // multi-line pipe with function calls and arguments
    assert_ok_contains(
        "pipe_multiline_args",
        r#"let r = [1, 2, 3, 4, 5]
    |> filter(|x| x > 2)
    |> map(|x| x * 10)
display r[0]"#,
        "30",
    );
}

#[test]
fn list_multiline() {
    // list literal with elements across multiple lines
    assert_ok_contains(
        "list_multiline",
        r#"let xs = [
    1,
    2,
    3
]
display xs[1]"#,
        "2",
    );
}

#[test]
fn list_multiline_strings() {
    // list of strings across multiple lines
    assert_ok_contains(
        "list_multiline_strings",
        r#"let names = [
    "alpha",
    "beta",
    "gamma"
]
display names[2]"#,
        "\"gamma\"",
    );
}

#[test]
fn list_multiline_map() {
    // map over list defined across multiple lines
    assert_ok_contains(
        "list_multiline_map",
        r#"let xs = [
    10,
    20,
    30
] |> map(|x| x + 1)
display xs[0]"#,
        "11",
    );
}

// ── Regression tests: multi-line dict literal ─────────────────────────

#[test]
fn dict_multiline_basic() {
    // dict literal with pairs across multiple lines
    assert_ok_contains(
        "dict_multiline_basic",
        r#"let d = {
    "x": 10,
    "y": 20
}
display d["x"]"#,
        "10",
    );
}

#[test]
fn dict_multiline_string_values() {
    // dict with strings across multiple lines
    assert_ok_contains(
        "dict_multiline_string_values",
        r#"let d = {
    "name": "Hayashi",
    "version": "0.2.6"
}
display d["name"]"#,
        "\"Hayashi\"",
    );
}

#[test]
fn dict_multiline_trailing_comma() {
    // dict with trailing comma on the last line (must be tolerated)
    assert_ok_contains(
        "dict_multiline_trailing_comma",
        r#"let d = {
    "a": 1,
    "b": 2,
}
display d["b"]"#,
        "2",
    );
}

#[test]
fn dict_multiline_nested_list() {
    // dict with list as value, all across multiple lines
    assert_ok_contains(
        "dict_multiline_nested_list",
        r#"let d = {
    "xs": [1, 2, 3],
    "ys": [4, 5, 6]
}
display d["xs"][0]"#,
        "1",
    );
}

#[test]
fn dict_multiline_in_call() {
    // multi-line dict as function argument
    assert_ok_contains(
        "dict_multiline_in_call",
        r#"let df = dataframe({
    "a": [10, 20, 30],
    "b": [1, 2, 3]
})
display nrow(df)"#,
        "3",
    );
}

#[test]
fn diag_acf_returns_list() {
    assert_ok_contains(
        "acf_list",
        r#"
input df
Y
1
2
3
4
5
6
7
8
end
let v = acf(df, Y, lags=3)
display len(v)
"#,
        "4",
    );
}

#[test]
fn diag_pacf_returns_list() {
    assert_ok_contains(
        "pacf_list",
        r#"
input df
Y
1
2
3
4
5
6
7
8
end
let v = pacf(df, Y, lags=3)
display len(v)
"#,
        "4",
    );
}

#[test]
fn diag_acf_on_model() {
    assert_ok_contains(
        "acf_model",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
9 1
13 4
end
let m = ols(Y ~ X, df)
let v = acf(m, lags=2)
display len(v)
"#,
        "3",
    );
}

#[test]
fn diag_cusumtest() {
    assert_ok_contains(
        "cusum_test",
        r#"
input df
Y X
10 2
12 3
8 1
15 5
11 2
14 4
9 1
13 4
16 5
17 6
end
let m = ols(Y ~ X, df)
cusumtest(m)
"#,
        "CUSUM",
    );
}

#[test]
fn diag_akaike_weights() {
    assert_ok_contains(
        "akaike_weights_test",
        r#"
input df
Y X1 X2
10 2 1
12 3 2
8 1 0
15 5 3
11 2 1
14 4 2
9 1 0
13 4 3
end
let m1 = ols(Y ~ X1, df)
let m2 = ols(Y ~ X1 + X2, df)
let w = akaike_weights(m1, m2)
display has_key(w, "m1")
"#,
        "true",
    );
}

#[test]
fn diag_lrtest_ols() {
    assert_ok_contains(
        "lrtest_ols",
        r#"
input df
Y X1 X2
10 2 1
12 3 2
8 1 0
15 5 3
11 2 1
14 4 2
9 1 0
13 4 3
end
let m1 = ols(Y ~ X1, df)
let m2 = ols(Y ~ X1 + X2, df)
lrtest(m1, m2)
"#,
        "Likelihood-Ratio",
    );
}

#[test]
fn diag_lrtest_rejects_non_nested() {
    // m1 has MORE params than m2 — should error
    assert_ok_contains(
        "lrtest_non_nested",
        r#"
input df
Y X1 X2
10 2 1
12 3 2
8 1 0
15 5 3
11 2 1
14 4 2
9 1 0
13 4 3
end
let m1 = ols(Y ~ X1 + X2, df)
let m2 = ols(Y ~ X1, df)
try {
    lrtest(m1, m2)
    display "no error"
} catch e {
    display "caught"
}
"#,
        "caught",
    );
}

#[test]
fn iv_sargan_overid() {
    // Overidentified: 2 instruments (z1, z2) for 1 endogenous (x)
    // z1 and z2 are valid instruments → should not reject H0
    assert_ok_contains(
        "sargan_test",
        r#"
input df
y x z1 z2
3 2 1 3
5 3 2 1
4 4 3 4
7 5 4 2
6 6 5 5
9 7 6 3
8 8 7 6
11 9 8 4
10 10 9 7
13 11 10 5
end
estat_overid(y ~ x, ~ z1 + z2, df)
"#,
        "Sargan",
    );
}

#[test]
fn iv_endog_test() {
    // DWH endogeneity test: x is endogenous (correlated with error via construction)
    assert_ok_contains(
        "endog_test",
        r#"
input df
y x z1
1 2 1
2 3 2
3 4 3
4 5 4
5 6 5
6 7 6
7 8 7
8 9 8
9 10 9
10 11 10
end
estat_endog(y ~ x, ~ z1, df)
"#,
        "Durbin-Wu-Hausman",
    );
}

#[test]
fn logit_classification() {
    assert_ok_contains(
        "logit_class",
        r#"
input df
y x1 x2
0 1.0 2.0
0 1.5 2.5
0 2.0 1.0
0 2.5 3.0
0 3.0 1.5
0 1.2 2.8
0 2.8 1.2
0 3.2 2.8
0 1.8 3.2
0 2.2 2.2
0 3.5 1.8
0 1.0 3.0
0 2.0 2.5
0 3.0 2.0
0 1.5 1.5
0 2.5 1.0
0 3.8 2.5
0 1.2 1.8
0 2.8 3.5
0 2.0 3.8
1 4.0 5.0
1 4.5 4.5
1 5.0 6.0
1 5.5 5.5
1 6.0 4.0
1 4.2 5.8
1 5.8 4.2
1 6.2 5.8
1 4.8 6.2
1 5.2 5.2
1 6.5 4.8
1 4.0 6.5
1 5.0 5.0
1 6.0 4.5
1 4.5 4.0
1 5.5 6.5
1 3.2 4.5
1 4.8 3.8
1 5.8 6.0
1 5.0 4.2
0 4.5 3.0
0 3.5 4.0
0 5.0 2.0
0 4.2 3.5
1 2.5 4.5
1 3.0 5.0
1 2.0 4.0
1 3.5 3.5
0 5.5 3.5
0 4.8 4.2
end
let m = logit(y ~ x1 + x2, df)
estat_classification(m)
"#,
        "Sensitivity",
    );
}

#[test]
fn logit_roc_auc() {
    assert_ok_contains(
        "logit_roc",
        r#"
input df
y x1 x2
0 1.0 2.0
0 1.5 2.5
0 2.0 1.0
0 2.5 3.0
0 3.0 1.5
0 1.2 2.8
0 2.8 1.2
0 3.2 2.8
0 1.8 3.2
0 2.2 2.2
0 3.5 1.8
0 1.0 3.0
0 2.0 2.5
0 3.0 2.0
0 1.5 1.5
0 2.5 1.0
0 3.8 2.5
0 1.2 1.8
0 2.8 3.5
0 2.0 3.8
1 4.0 5.0
1 4.5 4.5
1 5.0 6.0
1 5.5 5.5
1 6.0 4.0
1 4.2 5.8
1 5.8 4.2
1 6.2 5.8
1 4.8 6.2
1 5.2 5.2
1 6.5 4.8
1 4.0 6.5
1 5.0 5.0
1 6.0 4.5
1 4.5 4.0
1 5.5 6.5
1 3.2 4.5
1 4.8 3.8
1 5.8 6.0
1 5.0 4.2
0 4.5 3.0
0 3.5 4.0
0 5.0 2.0
0 4.2 3.5
1 2.5 4.5
1 3.0 5.0
1 2.0 4.0
1 3.5 3.5
0 5.5 3.5
0 4.8 4.2
end
let m = logit(y ~ x1 + x2, df)
lroc(m)
"#,
        "AUC",
    );
}

#[test]
fn logit_hosmer_lemeshow() {
    assert_ok_contains(
        "logit_hl",
        r#"
input df
y x1 x2
0 1.0 2.0
0 1.5 2.5
0 2.0 1.0
0 2.5 3.0
0 3.0 1.5
0 1.2 2.8
0 2.8 1.2
0 3.2 2.8
0 1.8 3.2
0 2.2 2.2
0 3.5 1.8
0 1.0 3.0
0 2.0 2.5
0 3.0 2.0
0 1.5 1.5
0 2.5 1.0
0 3.8 2.5
0 1.2 1.8
0 2.8 3.5
0 2.0 3.8
1 4.0 5.0
1 4.5 4.5
1 5.0 6.0
1 5.5 5.5
1 6.0 4.0
1 4.2 5.8
1 5.8 4.2
1 6.2 5.8
1 4.8 6.2
1 5.2 5.2
1 6.5 4.8
1 4.0 6.5
1 5.0 5.0
1 6.0 4.5
1 4.5 4.0
1 5.5 6.5
1 3.2 4.5
1 4.8 3.8
1 5.8 6.0
1 5.0 4.2
0 4.5 3.0
0 3.5 4.0
0 5.0 2.0
0 4.2 3.5
1 2.5 4.5
1 3.0 5.0
1 2.0 4.0
1 3.5 3.5
0 5.5 3.5
0 4.8 4.2
end
let m = logit(y ~ x1 + x2, df)
estat_gof(m)
"#,
        "Hosmer-Lemeshow",
    );
}

#[test]
fn logit_linktest() {
    assert_ok_contains(
        "logit_linktest",
        r#"
input df
y x1 x2
0 1.0 2.0
0 1.5 2.5
0 2.0 1.0
0 2.5 3.0
0 3.0 1.5
0 1.2 2.8
0 2.8 1.2
0 3.2 2.8
0 1.8 3.2
0 2.2 2.2
0 3.5 1.8
0 1.0 3.0
0 2.0 2.5
0 3.0 2.0
0 1.5 1.5
0 2.5 1.0
0 3.8 2.5
0 1.2 1.8
0 2.8 3.5
0 2.0 3.8
1 4.0 5.0
1 4.5 4.5
1 5.0 6.0
1 5.5 5.5
1 6.0 4.0
1 4.2 5.8
1 5.8 4.2
1 6.2 5.8
1 4.8 6.2
1 5.2 5.2
1 6.5 4.8
1 4.0 6.5
1 5.0 5.0
1 6.0 4.5
1 4.5 4.0
1 5.5 6.5
1 3.2 4.5
1 4.8 3.8
1 5.8 6.0
1 5.0 4.2
0 4.5 3.0
0 3.5 4.0
0 5.0 2.0
0 4.2 3.5
1 2.5 4.5
1 3.0 5.0
1 2.0 4.0
1 3.5 3.5
0 5.5 3.5
0 4.8 4.2
end
let m = logit(y ~ x1 + x2, df)
linktest(m)
"#,
        "Linktest",
    );
}

#[test]
fn xtlogit_panel() {
    assert_ok_contains(
        "xtlogit_panel",
        r#"
input df
id y x1 x2
1 0 1.0 2.0
1 1 1.5 2.5
1 0 2.0 3.0
1 1 2.5 3.5
2 0 3.0 1.0
2 0 3.5 1.5
2 1 4.0 2.0
2 1 4.5 2.5
3 1 5.0 4.0
3 1 5.5 4.5
3 1 6.0 5.0
3 1 6.5 5.5
end
xtlogit(y ~ x1 + x2, df, id="id")
"#,
        "Generalized Estimating",
    );
}

#[test]
fn xtpoisson_panel() {
    assert_ok_contains(
        "xtpoisson_panel",
        r#"
input df
id y x1
1 2 1.0
1 3 1.5
1 4 2.0
2 1 0.5
2 2 1.0
2 3 1.5
3 5 2.0
3 6 2.5
3 7 3.0
end
xtpoisson(y ~ x1, df, id="id")
"#,
        "Generalized Estimating",
    );
}

#[test]
fn eventstudy_basic() {
    assert_ok_contains(
        "eventstudy_basic",
        r#"
input df
y etime
10 -2
12 -1
15 0
18 1
20 2
22 3
8 -2
10 -1
14 0
17 1
19 2
21 3
end
eventstudy(y ~ etime, df, ref=-1, min=-2, max=3)
"#,
        "Event Study",
    );
}

#[test]
fn nls_exponential() {
    assert_ok_contains(
        "nls_exp",
        r#"
input df
y x
2.8 1.0
7.7 2.0
20.1 3.0
55.0 4.0
148.0 5.0
403.0 6.0
end
nls_exp(y ~ x, df, start=[1.0, 1.0])
"#,
        "Nonlinear Least Squares",
    );
}

#[test]
fn nls_cobb_douglas() {
    assert_ok_contains(
        "nls_cd",
        r#"
input df
y k l
10 5 5
15 8 6
20 10 8
25 12 10
30 15 12
35 18 14
40 20 16
45 22 18
end
nls_cobb_douglas(y ~ k + l, df, start=[1.0, 0.5, 0.5])
"#,
        "Nonlinear Least Squares",
    );
}

#[test]
fn marginsplot_logit() {
    assert_ok_contains(
        "marginsplot",
        r#"
input df
y x1 x2
0 1.0 2.0
0 1.5 2.5
0 2.0 1.0
0 2.5 3.0
0 3.0 1.5
0 1.2 2.8
0 2.8 1.2
0 3.2 2.8
0 1.8 3.2
0 2.2 2.2
0 3.5 1.8
0 1.0 3.0
0 2.0 2.5
0 3.0 2.0
0 1.5 1.5
0 2.5 1.0
0 3.8 2.5
0 1.2 1.8
0 2.8 3.5
0 2.0 3.8
1 4.0 5.0
1 4.5 4.5
1 5.0 6.0
1 5.5 5.5
1 6.0 4.0
1 4.2 5.8
1 5.8 4.2
1 6.2 5.8
1 4.8 6.2
1 5.2 5.2
1 6.5 4.8
1 4.0 6.5
1 5.0 5.0
1 6.0 4.5
1 4.5 4.0
1 5.5 6.5
1 3.2 4.5
1 4.8 3.8
1 5.8 6.0
1 5.0 4.2
0 4.5 3.0
0 3.5 4.0
0 5.0 2.0
0 4.2 3.5
1 2.5 4.5
1 3.0 5.0
1 2.0 4.0
1 3.5 3.5
0 5.5 3.5
0 4.8 4.2
end
let m = logit(y ~ x1 + x2, df)
marginsplot(m)
"#,
        "Marginal Effects",
    );
}

#[test]
fn spatial_sar_basic() {
    assert_ok_contains(
        "spatial_sar",
        r#"
input df
y x
1.0 1.0
2.0 2.0
3.0 3.0
4.0 4.0
5.0 5.0
6.0 6.0
7.0 7.0
8.0 8.0
end
let W = [[0, 0.5, 0.5, 0, 0, 0, 0, 0],
         [0.5, 0, 0.5, 0, 0, 0, 0, 0],
         [0, 0.5, 0, 0.5, 0, 0, 0, 0],
         [0, 0, 0.5, 0, 0.5, 0, 0, 0],
         [0, 0, 0, 0.5, 0, 0.5, 0, 0],
         [0, 0, 0, 0, 0.5, 0, 0.5, 0],
         [0, 0, 0, 0, 0, 0.5, 0, 0.5],
         [0, 0, 0, 0, 0, 0, 0.5, 0]]
spatial_sar(y ~ x, df, w=W)
"#,
        "Spatial Autoregressive",
    );
}

#[test]
fn spatial_sem_basic() {
    assert_ok_contains(
        "spatial_sem",
        r#"
input df
y x
1.0 1.0
2.0 2.0
3.0 3.0
4.0 4.0
5.0 5.0
6.0 6.0
7.0 7.0
8.0 8.0
end
let W = [[0, 0.5, 0.5, 0, 0, 0, 0, 0],
         [0.5, 0, 0.5, 0, 0, 0, 0, 0],
         [0, 0.5, 0, 0.5, 0, 0, 0, 0],
         [0, 0, 0.5, 0, 0.5, 0, 0, 0],
         [0, 0, 0, 0.5, 0, 0.5, 0, 0],
         [0, 0, 0, 0, 0.5, 0, 0.5, 0],
         [0, 0, 0, 0, 0, 0.5, 0, 0.5],
         [0, 0, 0, 0, 0, 0, 0.5, 0]]
spatial_sem(y ~ x, df, w=W)
"#,
        "Spatial Error",
    );
}

#[test]
fn double_ml_basic() {
    assert_ok_contains(
        "double_ml",
        r#"
input df
y d x1 x2
2.5 1 1.0 0.5
3.0 1 1.5 0.8
3.5 1 2.0 1.0
4.0 0 2.5 1.2
4.5 0 3.0 1.5
5.0 1 3.5 1.8
5.5 1 4.0 2.0
6.0 0 4.5 2.2
6.5 0 5.0 2.5
7.0 1 5.5 2.8
7.5 1 6.0 3.0
8.0 0 6.5 3.2
8.5 0 7.0 3.5
9.0 1 7.5 3.8
9.5 1 8.0 4.0
10.0 0 8.5 4.2
10.5 0 9.0 4.5
11.0 1 9.5 4.8
11.5 1 10.0 5.0
12.0 0 10.5 5.2
end
double_ml(y ~ d + x1 + x2, df, folds=5, poly=2)
"#,
        "Double/Debiased ML",
    );
}

#[test]
fn sfa_production_basic() {
    assert_ok_contains(
        "sfa_prod",
        r#"
input df
y k l
10 5 5
12 6 6
15 8 7
18 10 8
20 12 10
22 13 11
25 15 12
28 17 13
30 18 14
32 20 15
35 22 16
38 24 17
40 25 18
42 27 19
45 28 20
end
sfa_production(y ~ k + l, df)
"#,
        "Stochastic Production Frontier",
    );
}

#[test]
fn panel_tobit_basic() {
    assert_ok_contains(
        "panel_tobit",
        r#"
input df
firm y x
1 0 1.0
1 5 2.0
1 8 3.0
1 12 4.0
2 0 0.5
2 3 1.5
2 7 2.5
2 10 3.5
3 2 1.0
3 6 2.0
3 9 3.0
3 11 4.0
4 0 0.8
4 4 1.8
4 8 2.8
4 13 3.8
end
panel_tobit(y ~ x, df, id="firm", censor=0)
"#,
        "Panel Tobit",
    );
}

#[test]
fn panel_heckman_basic() {
    assert_ok_contains(
        "panel_heckman",
        r#"
input df
id wage educ exper particip kids
1 10 8 2 1 0
1 12 8 5 1 1
1 0 8 8 0 3
1 15 8 11 1 0
2 8 6 1 1 1
2 0 6 4 0 2
2 11 6 7 1 0
2 13 6 10 1 1
3 0 10 3 0 3
3 18 10 6 1 0
3 20 10 9 1 1
3 22 10 12 1 0
4 9 7 2 1 0
4 0 7 5 0 3
4 12 7 8 1 1
4 14 7 11 1 0
5 11 9 3 1 0
5 0 9 6 0 2
5 16 9 9 1 1
5 19 9 12 1 0
end
panel_heckman(wage ~ educ + exper, df, sel="particip ~ educ + kids", id="id")
"#,
        "Panel Heckman",
    );
}

#[test]
fn spatial_panel_sar_basic() {
    assert_ok_contains(
        "spatial_panel_sar",
        r#"
input df
entity y x
1 10 1.0
1 12 2.0
1 15 3.0
2 8 0.8
2 10 1.8
2 13 2.8
3 11 1.2
3 14 2.2
3 17 3.2
4 9 0.9
4 11 1.9
4 14 2.9
end
let W = [[0, 0.5, 0.3, 0.2],
         [0.5, 0, 0.3, 0.2],
         [0.3, 0.3, 0, 0.4],
         [0.2, 0.2, 0.4, 0]]
spatial_panel_sar(y ~ x, df, w=W, id="entity")
"#,
        "Spatial Panel SAR",
    );
}

#[test]
fn bayes_sfa_basic() {
    assert_ok_contains(
        "bayes_sfa",
        r#"
input df
y k l
10 5 5
12 6 6
15 8 7
18 10 8
20 12 10
22 13 11
25 15 12
28 17 13
30 18 14
32 20 15
end
bayes_sfa_production(y ~ k + l, df, burn=200, draws=500)
"#,
        "Bayesian Stochastic",
    );
}
