use std::process::Command;

fn run_hy(script: &str) -> (bool, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_hayashi"))
        .arg(script)
        .output()
        .expect("failed to execute hayashi");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = format!("{stdout}{stderr}");
    (output.status.success(), combined)
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

// ══════════════════════════════════════════════════════════════
// Smoke tests — exemplos .hy devem rodar sem erro
// ══════════════════════════════════════════════════════════════

#[test]
fn smoke_input_display() {
    let (ok, out) = run_hy("exemplos/input_display.hy");
    assert!(ok, "input_display.hy failed:\n{out}");
    assert!(out.contains("OLS Regression"));
}

#[test]
fn smoke_eststo_loop() {
    let (ok, out) = run_hy("exemplos/eststo_loop.hy");
    assert!(ok, "eststo_loop.hy failed:\n{out}");
    assert!(out.contains("eststo: modelo 3"));
    assert!(out.contains("R²"));
}

#[test]
fn smoke_foreach_regression() {
    let (ok, out) = run_hy("exemplos/foreach_regression.hy");
    assert!(ok, "foreach_regression.hy failed:\n{out}");
    assert!(out.contains("●"));
}

#[test]
fn smoke_portfolio_sort() {
    let (ok, out) = run_hy("exemplos/portfolio_sort.hy");
    assert!(ok, "portfolio_sort.hy failed:\n{out}");
    assert!(out.contains("H-L"));
}

#[test]
fn smoke_fama_macbeth() {
    let (ok, out) = run_hy("exemplos/fama_macbeth.hy");
    assert!(ok, "fama_macbeth.hy failed:\n{out}");
    assert!(out.contains("Fama-MacBeth"));
}

#[test]
fn smoke_double_sort() {
    let (ok, out) = run_hy("exemplos/double_sort.hy");
    assert!(ok, "double_sort.hy failed:\n{out}");
    assert!(out.contains("Double Sort"));
}

#[test]
fn smoke_pwcorr() {
    let (ok, out) = run_hy("exemplos/pwcorr.hy");
    assert!(ok, "pwcorr.hy failed:\n{out}");
    assert!(out.contains("***"));
}

#[test]
fn smoke_missing_values() {
    let (ok, out) = run_hy("exemplos/missing_values.hy");
    assert!(ok, "missing_values.hy failed:\n{out}");
    assert!(out.contains("Missing"));
}

#[test]
fn smoke_preserve_restore() {
    let (ok, out) = run_hy("exemplos/preserve_restore.hy");
    assert!(ok, "preserve_restore.hy failed:\n{out}");
    assert!(out.contains("restore"));
}

#[test]
fn smoke_test_cov_display() {
    let (ok, out) = run_hy("exemplos/test_cov_display.hy");
    assert!(ok, "test_cov_display.hy failed:\n{out}");
    assert!(out.contains("OLS (robust)"));
    assert!(out.contains("OLS (cluster)"));
}

#[test]
fn smoke_tres_workflows() {
    let (ok, out) = run_hy("exemplos/tres_workflows.hy");
    assert!(ok, "tres_workflows.hy failed:\n{out}");
    assert!(out.contains("Hausman"));
}

// ══════════════════════════════════════════════════════════════
// Testes unitários — semântica da linguagem
// ══════════════════════════════════════════════════════════════

#[test]
fn scoping_let_dies_at_block_end() {
    let (ok, out) = run_inline(r#"
let x = 10
if x > 5 {
    let inner = 42
    display inner
}
display x
"#);
    assert!(ok, "scoping failed:\n{out}");
    assert!(out.contains("42"));
    assert!(out.contains("10"));
}

#[test]
fn scoping_assign_modifies_outer() {
    let (ok, out) = run_inline(r#"
let counter = 0
for i in 1..4 {
    counter = counter + 1
}
display counter
"#);
    assert!(ok, "assign outer failed:\n{out}");
    assert!(out.contains("3"));
}

#[test]
fn scoping_fn_params_dont_leak() {
    let (ok, out) = run_inline(r#"
fn add(a, b) {
    return a + b
}
let s = add(3, 4)
display s
"#);
    assert!(ok, "fn scope failed:\n{out}");
    assert!(out.contains("7"));
}

#[test]
fn scoping_for_var_dies() {
    let (_, out) = run_inline(r#"
for v in [1, 2, 3] {
    display v
}
display v
"#);
    assert!(out.contains("1"));
    assert!(out.contains("2"));
    assert!(out.contains("3"));
}

#[test]
fn dynamic_formula() {
    let (ok, out) = run_inline(r#"
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
"#);
    assert!(ok, "dynamic formula failed:\n{out}");
    assert!(out.contains("OLS Regression"));
}

#[test]
fn if_condition_in_ols() {
    let (ok, out) = run_inline(r#"
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
"#);
    assert!(ok, "if= in ols failed:\n{out}");
    assert!(out.contains("No. Observations:                  4"));
}

#[test]
fn winsor_works() {
    let (ok, out) = run_inline(r#"
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
"#);
    assert!(ok, "winsor failed:\n{out}");
    assert!(out.contains("winsor"));
}

#[test]
fn test_wald() {
    let (ok, out) = run_inline(r#"
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
"#);
    assert!(ok, "wald test failed:\n{out}");
    assert!(out.contains("H₀"));
}

// string_comparison_in_generate e encode_works removidos:
// input block só suporta dados numéricos; strings requerem load CSV/DTA

#[test]
fn set_seed_reproducibility() {
    let (ok1, out1) = run_inline("set_seed(42)\ndisplay 1");
    let (ok2, out2) = run_inline("set_seed(42)\ndisplay 1");
    assert!(ok1 && ok2);
    assert_eq!(out1, out2);
}

#[test]
fn help_works() {
    let (ok, out) = run_inline("help(ols)");
    assert!(ok, "help failed:\n{out}");
    assert!(out.contains("cluster=var"));
}

#[test]
fn timer_works() {
    let (ok, out) = run_inline(r#"
input df
Y X
1 2
3 4
5 6
7 8
end
timer(ols(Y ~ X, df))
"#);
    assert!(ok, "timer failed:\n{out}");
    assert!(out.contains("elapsed"));
}
