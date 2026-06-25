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
smoke!(
    smoke_count_models,
    "examples/count_models.hay",
    "Poisson"
);
smoke!(smoke_panel, "examples/panel.hay", "FE");

// ══════════════════════════════════════════════════════════════════════════════
// SCOPING — block scoping com destruição determinística
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
// PARSER — lexer, parser, expressões
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn parser_comments() {
    assert_ok_contains(
        "comments",
        r#"
// isto é um comentário
# isto também
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
// EXPRESSÕES — aritmética, comparação, funções
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
// ESTIMAÇÃO — OLS, fórmulas, opções
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
// PÓS-ESTIMAÇÃO — test, esttab, coefplot, nlcom, predict
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

// ══════════════════════════════════════════════════════════════════════════════
// FINANÇAS — portsort, doublesort, fmb
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
// EDGE CASES — comportamento em condições limítrofes
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
// GRÁFICOS SVG — plotters backend
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
// FUNÇÕES MATEMÁTICAS — paridade com Stata
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
// PARIDADE STATA — distribuições, _n/_N, ci, centile, recode, egen
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
        "duplicatas",
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
        "3 obs removidas",
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
label(df, Y, "Variável dependente")
label(df, X, "Variável independente")
describe(df)
"#,
        "Variável dependente",
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
        // com feature ativa, vai falhar por falta de driver — mas não por "not enabled"
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
// LIST — operações de lista
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
// DICT — dicionário/mapa
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
// CONST — variáveis imutáveis
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
