//! Regression tests for PR 1: user-facing runtime errors must be in English.
//!
//! These cover the narrow argument-validation paths translated in PR 1
//! (row-wise helpers + estimator/diagnostic model-requirement errors). Each
//! test triggers an invalid call and asserts the English wording appears and
//! that the previous Portuguese fragments do NOT.

use std::process::Command;

/// Run a Hayashi program from stdin and return combined stdout+stderr.
fn run_inline(src: &str) -> String {
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
    format!("{stdout}{stderr}")
}

/// Portuguese fragments that must not appear in messages covered by PR 1.
const FORBIDDEN_PT: &[&str] = &[
    "requer",
    "coluna",
    "variável",
    "não",
    "deve ser",
    "inválid",
    "vazia",
    "ao menos",
    "pelo menos",
    "esperad",
    "argumento",
    "modelo",
];

fn assert_english(test: &str, out: &str, expected_substr: &str) {
    assert!(
        out.contains(expected_substr),
        "{test}: expected English message containing '{expected_substr}', got:\n{out}"
    );
    let lower = out.to_lowercase();
    for term in FORBIDDEN_PT {
        assert!(
            !lower.contains(term),
            "{test}: output still contains Portuguese term '{term}':\n{out}"
        );
    }
}

#[test]
fn rowmean_requires_a_column() {
    let out = run_inline(
        r#"input df
x
1
end
generate df z = rowmean()
"#,
    );
    assert_english("rowmean", &out, "rowmean() requires at least one column");
}

#[test]
fn cooks_rejects_non_ols_argument() {
    // Passing a non-OLS value hits the "only supports OLS models" type-error path.
    let out = run_inline("x = 1\ncooks(x)\n");
    assert!(
        out.contains("cooks() only supports OLS models"),
        "cooks: expected English message, got:\n{out}"
    );
}

#[test]
fn irf_requires_var_model() {
    let out = run_inline("x = 1\nirf(x)\n");
    assert!(
        out.contains("irf() requires a VAR model"),
        "irf: expected English message, got:\n{out}"
    );
}

#[test]
fn sirf_requires_svar_model() {
    let out = run_inline("x = 1\nsirf(x)\n");
    assert_english("sirf", &out, "sirf() requires an SVAR model");
}
