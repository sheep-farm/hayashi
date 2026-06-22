use std::process::Command;

fn run_hy(script: &str) -> (bool, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_hayashi"))
        .arg(script)
        .output()
        .expect("failed to execute hayashi");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output.status.success(), format!("{stdout}{stderr}"))
}

fn run_inline(src: &str) -> (bool, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_hayashi"))
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
        .expect("failed to execute hayashi");
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
    assert!(out.contains(needle), "{name}: output missing '{needle}':\n{out}");
}

// ══════════════════════════════════════════════════════════════════════════════
// SMOKE TESTS — exemplos .hy offline (17 que passam sem rede)
// ══════════════════════════════════════════════════════════════════════════════

macro_rules! smoke {
    ($name:ident, $path:expr, $needle:expr) => {
        #[test]
        fn $name() {
            let (ok, out) = run_hy($path);
            assert!(ok, "{} failed:\n{}", $path, out);
            assert!(out.contains($needle), "{}: missing '{}'\n{}", $path, $needle, out);
        }
    };
}

smoke!(smoke_input_display,     "exemplos/input_display.hy",     "OLS Regression");
smoke!(smoke_eststo_loop,       "exemplos/eststo_loop.hy",       "estclear");
smoke!(smoke_foreach_regression,"exemplos/foreach_regression.hy","●");
smoke!(smoke_portfolio_sort,    "exemplos/portfolio_sort.hy",    "H-L");
smoke!(smoke_fama_macbeth,      "exemplos/fama_macbeth.hy",      "Fama-MacBeth");
smoke!(smoke_double_sort,       "exemplos/double_sort.hy",       "Double Sort");
smoke!(smoke_pwcorr,            "exemplos/pwcorr.hy",            "***");
smoke!(smoke_missing_values,    "exemplos/missing_values.hy",    "Missing");
smoke!(smoke_preserve_restore,  "exemplos/preserve_restore.hy",  "restore");
smoke!(smoke_test_cov_display,  "exemplos/test_cov_display.hy",  "OLS (cluster)");
smoke!(smoke_tres_workflows,    "exemplos/tres_workflows.hy",    "Hausman");
smoke!(smoke_modelos_contagem,  "exemplos/modelos_contagem.hy",  "Poisson");
smoke!(smoke_painel,            "exemplos/painel.hy",            "FE");

// ══════════════════════════════════════════════════════════════════════════════
// SCOPING — block scoping com destruição determinística
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn scope_let_block() {
    assert_ok_contains("scope_let_block", r#"
let x = 10
if x > 5 {
    let inner = 42
    display inner
}
display x
"#, "42");
}

#[test]
fn scope_assign_outer() {
    assert_ok_contains("scope_assign_outer", r#"
let counter = 0
for i in 1..4 {
    counter = counter + 1
}
display counter
"#, "3");
}

#[test]
fn scope_fn_params_die() {
    assert_ok_contains("scope_fn_params", r#"
fn add(a, b) {
    return a + b
}
display add(3, 4)
"#, "7");
}

#[test]
fn scope_for_var_dies() {
    let (_, out) = run_inline(r#"
for v in [10, 20, 30] {
    display v
}
"#);
    assert!(out.contains("10") && out.contains("20") && out.contains("30"));
}

#[test]
fn scope_nested_blocks() {
    assert_ok_contains("nested_blocks", r#"
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
"#, "3");
}

#[test]
fn scope_while_cleanup() {
    assert_ok_contains("while_cleanup", r#"
let n = 0
while n < 3 {
    let temp = n * 10
    n = n + 1
}
display n
"#, "3");
}

#[test]
fn scope_fn_recursive() {
    assert_ok_contains("fn_recursive", r#"
fn factorial(n) {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}
display factorial(5)
"#, "120");
}

// ══════════════════════════════════════════════════════════════════════════════
// PARSER — lexer, parser, expressões
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn parser_comments() {
    assert_ok_contains("comments", r#"
// isto é um comentário
# isto também
display 42
"#, "42");
}

#[test]
fn parser_string_concat() {
    assert_ok_contains("string_concat", r#"
let a = "hello"
let b = " world"
display a + b
"#, "hello world");
}

#[test]
fn parser_negative_numbers() {
    assert_ok_contains("negative", r#"
display -5 + 3
"#, "-2");
}

#[test]
fn parser_boolean_ops() {
    assert_ok_contains("bool_and", r#"
let x = 5
if x > 3 && x < 10 {
    display 1
}
"#, "1");
}

#[test]
fn parser_list_indexing() {
    assert_ok_contains("list_index", r#"
let v = [10, 20, 30]
display v[1]
"#, "20");
}

#[test]
fn parser_not_operator() {
    assert_ok_contains("not_op", r#"
let x = false
if !x {
    display 99
}
"#, "99");
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
    assert_ok_contains("comparison", r#"
if 5 > 3 { display 1 }
if 3 >= 3 { display 2 }
if 2 < 3 { display 3 }
if 2 != 3 { display 4 }
"#, "4");
}

#[test]
fn expr_math_functions() {
    assert_ok_contains("math_fns", r#"
input df
X
1
4
end
generate df E = exp(X)
generate df S = sqrt(X)
generate df A = abs(X)
list(df, n=2)
"#, "generated");
}

// ══════════════════════════════════════════════════════════════════════════════
// DADOS — input, generate, replace, summarize, list
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn data_input_block() {
    assert_ok_contains("input", r#"
input df
Y X
1 2
3 4
5 6
end
display mean(df, Y)
"#, "3");
}

#[test]
fn data_generate() {
    assert_ok_contains("generate", r#"
input df
X
2
4
6
end
generate df Y = X * 2 + 1
display mean(df, Y)
"#, "9");
}

#[test]
fn data_replace_if() {
    assert_ok_contains("replace_if", r#"
input df
Y group
10 1
20 1
30 2
40 2
end
replace df Y = 0 if group == 2
display mean(df, Y)
"#, "real changes");
}

#[test]
fn data_mean_conditional() {
    assert_ok_contains("mean_if", r#"
input df
Y X
10 1
20 1
30 0
40 0
end
scalar m = mean(df, Y, if = X == 1)
display m
"#, "15");
}

#[test]
fn data_summarize() {
    assert_ok_contains("summarize", r#"
input df
Y
1
2
3
4
5
end
summarize(df)
"#, "Mean");
}

#[test]
fn data_tabulate() {
    assert_ok_contains("tabulate", r#"
input df
group
1
1
2
2
2
end
tabulate(df, group)
"#, "Freq");
}

#[test]
fn data_list_n() {
    assert_ok_contains("list_n", r#"
input df
Y X
1 2
3 4
5 6
7 8
end
list(df, n=2)
"#, "more observations");
}

#[test]
fn data_missing_nan() {
    assert_ok_contains("missing", r#"
input df
Y
1
.
3
end
summarize(df)
"#, "Missing");
}

// ══════════════════════════════════════════════════════════════════════════════
// ESTIMAÇÃO — OLS, fórmulas, opções
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn ols_basic() {
    assert_ok_contains("ols_basic", r#"
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
"#, "R-squared");
}

#[test]
fn ols_reg_alias() {
    assert_ok_contains("reg_alias", r#"
input df
Y X
10 2
12 3
8 1
15 5
end
reg(Y ~ X, df)
"#, "OLS Regression");
}

#[test]
fn ols_robust() {
    assert_ok_contains("ols_robust", r#"
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
"#, "Robust (HC1)");
}

#[test]
fn ols_hc3() {
    assert_ok_contains("ols_hc3", r#"
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
"#, "HC3");
}

#[test]
fn ols_dynamic_formula() {
    assert_ok_contains("dynamic_formula", r#"
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
"#, "OLS Regression");
}

#[test]
fn ols_if_condition() {
    assert_ok_contains("ols_if", r#"
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
"#, "No. Observations:                  4");
}

#[test]
fn ols_cluster() {
    assert_ok_contains("ols_cluster", r#"
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
"#, "Clustered");
}

// ══════════════════════════════════════════════════════════════════════════════
// PÓS-ESTIMAÇÃO — test, esttab, coefplot, nlcom, predict
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn post_test_wald() {
    assert_ok_contains("test_wald", r#"
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
test(m, X1)
"#, "H₀");
}

#[test]
fn post_test_restriction() {
    assert_ok_contains("test_restriction", r#"
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
"#, "H₀: X1 = X2");
}

#[test]
fn post_test_white() {
    assert_ok_contains("test_white", r#"
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
test(m, white)
"#, "White Test");
}

#[test]
fn post_esttab() {
    assert_ok_contains("esttab", r#"
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
"#, "R²");
}

#[test]
fn post_coefplot() {
    assert_ok_contains("coefplot", r#"
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
"#, "●");
}

#[test]
fn post_nlcom() {
    assert_ok_contains("nlcom", r#"
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
"#, "delta method");
}

#[test]
fn post_predict() {
    assert_ok_contains("predict", r#"
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
predict df resid = m, residuals
list(df, n=2)
"#, "predicted");
}

#[test]
fn post_estat() {
    assert_ok_contains("estat", r#"
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
"#, "AIC");
}

// ══════════════════════════════════════════════════════════════════════════════
// FINANÇAS — portsort, doublesort, fmb
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn finance_portsort() {
    assert_ok_contains("portsort", r#"
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
"#, "H-L");
}

#[test]
fn finance_fmb() {
    assert_ok_contains("fmb", r#"
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
"#, "Fama-MacBeth");
}

#[test]
fn finance_fmb_nw() {
    assert_ok_contains("fmb_nw", r#"
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
"#, "NW(1)");
}

// ══════════════════════════════════════════════════════════════════════════════
// DADOS — winsor, encode, tabgen, xtset
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn data_winsor() {
    assert_ok_contains("winsor", r#"
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
"#, "winsor");
}

#[test]
fn data_tabgen() {
    assert_ok_contains("tabgen", r#"
input df
group
1
1
2
2
3
end
tabgen(df, group)
"#, "dummies");
}

#[test]
fn data_xtset() {
    assert_ok_contains("xtset", r#"
input df
Y X firm year
10 2 1 2020
12 3 1 2021
15 5 2 2020
17 6 2 2021
end
xtset(df, firm, year)
"#, "xtset df");
}

// ══════════════════════════════════════════════════════════════════════════════
// LINGUAGEM — for, while, fn, if/else, source, help, timer
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn lang_for_range() {
    let (ok, out) = run_inline(r#"
let s = 0
for i in 1..6 {
    s = s + i
}
display s
"#);
    assert!(ok);
    assert!(out.contains("15"));
}

#[test]
fn lang_for_list() {
    let (ok, out) = run_inline(r#"
for x in [10, 20, 30] {
    display x
}
"#);
    assert!(ok);
    assert!(out.contains("10") && out.contains("20") && out.contains("30"));
}

#[test]
fn lang_while() {
    assert_ok_contains("while", r#"
let i = 0
let s = 0
while i < 5 {
    s = s + i
    i = i + 1
}
display s
"#, "10");
}

#[test]
fn lang_if_else() {
    assert_ok_contains("if_else", r#"
let x = 5
if x > 10 {
    display 1
} else {
    display 2
}
"#, "2");
}

#[test]
fn lang_fn_return() {
    assert_ok_contains("fn_return", r#"
fn square(x) {
    return x * x
}
display square(7)
"#, "49");
}

#[test]
fn lang_break_continue() {
    assert_ok_contains("break_continue", r#"
let s = 0
for i in 1..100 {
    if i > 5 { break }
    if i == 3 { continue }
    s = s + i
}
display s
"#, "12");
}

#[test]
fn lang_nested_fn() {
    assert_ok_contains("nested_fn", r#"
fn double(x) { return x * 2 }
fn quad(x) { return double(double(x)) }
display quad(3)
"#, "12");
}

#[test]
fn lang_eststo_esttab() {
    assert_ok_contains("eststo_esttab", r#"
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
"#, "estclear");
}

#[test]
fn lang_help() {
    assert_ok_contains("help", "help(ols)", "cluster=var");
}

#[test]
fn lang_help_index() {
    assert_ok_contains("help_index", "help()", "ESTIMADORES");
}

#[test]
fn lang_timer() {
    assert_ok_contains("timer", r#"
input df
Y X
10 2
12 3
8 1
15 5
end
timer(ols(Y ~ X, df))
"#, "elapsed");
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
    let (ok, out) = run_inline(r#"
display 42
display 3.14
display "hello"
display true
"#);
    assert!(ok);
    assert!(out.contains("42") && out.contains("3.14") && out.contains("hello") && out.contains("true"));
}

#[test]
fn lang_scalar() {
    assert_ok_contains("scalar", r#"
scalar x = 2 + 3
display x
"#, "5");
}

#[test]
fn lang_preserve_restore() {
    assert_ok("preserve_restore", r#"
input df
Y X
1 2
3 4
end
preserve(df)
generate df Z = X * 2
restore(df)
"#);
}
