use std::collections::HashMap;
use std::rc::Rc;
use ndarray::{Array1, Array2};
use greeners::{DataFrame, Formula as GFormula, OLS, CovarianceType, IV, Logit, Probit, FixedEffects, RandomEffects};
use greeners::diagnostics::Diagnostics;
use greeners::specification_tests::SpecificationTests;
use crate::lang::ast::*;
use crate::lang::error::{HayashiError, Result};

// ── Distribuição chi2 (Numerical Recipes — série + fração continuada) ─────────

fn gammln(xx: f64) -> f64 {
    let coefs = [76.18009172947146_f64, -86.50532032941677, 24.01409824083091,
                 -1.231739572450155, 1.208650973866179e-3, -5.395239384953e-6];
    let y = xx;
    let tmp = xx + 5.5 - (xx + 0.5) * (xx + 5.5).ln();
    let mut ser = 1.000000000190015_f64;
    let mut x = y;
    for c in &coefs { x += 1.0; ser += c / x; }
    -tmp + (2.5066282746310005 * ser / y).ln()
}

fn gammp_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    for _ in 0..200 {
        ap += 1.0; del *= x / ap; sum += del;
        if del.abs() < sum.abs() * 1e-10 { break; }
    }
    sum * (-x + a * x.ln() - gammln(a)).exp()
}

fn gammq_cf(a: f64, x: f64) -> f64 {
    let fpmin = 1e-300_f64;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1_i32..=200 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b; if d.abs() < fpmin { d = fpmin; }
        c = b + an / c; if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d;
        let del = d * c; h *= del;
        if (del - 1.0).abs() < 1e-10 { break; }
    }
    (-x + a * x.ln() - gammln(a)).exp() * h
}

// ── Distribuição t (via função beta incompleta) ───────────────────────────────

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let fpmin = 1e-300_f64;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin { d = fpmin; }
    d = 1.0 / d;
    let mut h = d;
    for m in 1_i32..=200 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if d.abs() < fpmin { d = fpmin; }
        c = 1.0 + aa / c; if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d; h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if d.abs() < fpmin { d = fpmin; }
        c = 1.0 + aa / c; if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d;
        let del = d * c; h *= del;
        if (del - 1.0).abs() < 1e-10 { break; }
    }
    h
}

fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let lbeta = gammln(a) + gammln(b) - gammln(a + b);
    let bt = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp();
    if x < (a + 1.0) / (a + b + 2.0) { bt * betacf(a, b, x) / a }
    else                               { 1.0 - bt * betacf(b, a, 1.0 - x) / b }
}

fn t_pvalue_two(t: f64, df: f64) -> f64 {
    betai(df / 2.0, 0.5, df / (df + t * t))
}

fn t_critical_95(df: f64) -> f64 {
    // busca binária para t tal que t_pvalue_two(t, df) = 0.05
    let (mut lo, mut hi) = (0.0_f64, 50.0_f64);
    for _ in 0..120 {
        let mid = (lo + hi) / 2.0;
        if t_pvalue_two(mid, df) > 0.05 { lo = mid; } else { hi = mid; }
    }
    (lo + hi) / 2.0
}

fn chi2_pvalue(stat: f64, df: usize) -> f64 {
    // p = Q(df/2, stat/2) = 1 - P(df/2, stat/2)
    let a = df as f64 / 2.0;
    let x = stat / 2.0;
    if x <= 0.0 { return 1.0; }
    if x < a + 1.0 { 1.0 - gammp_series(a, x) } else { gammq_cf(a, x) }
}

// ── Wrappers que preservam a matriz X para diagnósticos e predict ────────────

#[derive(Clone)]
pub struct OlsModel {
    pub result: Rc<greeners::OlsResult>,
    pub residuals: Array1<f64>,
    pub x: Array2<f64>,
}

impl std::fmt::Display for OlsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

#[derive(Clone)]
pub struct BinaryModel {
    pub result: Rc<greeners::discrete::BinaryModelResult>,
    pub x: Array2<f64>,
}

impl std::fmt::Display for BinaryModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.result)
    }
}

// ── Valores em runtime ────────────────────────────────────────────────────────

#[derive(Clone)]
pub enum Value {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    DataFrame(DataFrame),
    OlsResult(OlsModel),
    IvResult(Rc<greeners::iv::IvResult>),
    BinaryResult(BinaryModel),
    PanelResult(Rc<greeners::panel::PanelResult>),
    ReResult(Rc<greeners::panel::RandomEffectsResult>),
    Nil,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Float(v)        => write!(f, "{v}"),
            Value::Int(v)          => write!(f, "{v}"),
            Value::Bool(v)         => write!(f, "{v}"),
            Value::Str(v)          => write!(f, "{v}"),
            Value::DataFrame(df)   => write!(f, "{df}"),
            Value::OlsResult(m)    => write!(f, "{m}"),
            Value::IvResult(r)     => write!(f, "{r}"),
            Value::BinaryResult(m) => write!(f, "{m}"),
            Value::PanelResult(r)  => write!(f, "{r}"),
            Value::ReResult(r)     => write!(f, "{r}"),
            Value::Nil             => write!(f, "nil"),
        }
    }
}

// ── Ambiente de variáveis ─────────────────────────────────────────────────────

pub struct Env {
    vars: HashMap<String, Value>,
}

impl Env {
    pub fn new() -> Self {
        Self { vars: HashMap::new() }
    }

    pub fn set(&mut self, name: &str, val: Value) {
        self.vars.insert(name.to_string(), val);
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.vars.get(name)
    }
}

// ── Interpetador ──────────────────────────────────────────────────────────────

pub struct Interpreter {
    pub env: Env,
}

impl Interpreter {
    pub fn new() -> Self {
        Self { env: Env::new() }
    }

    // ── Avalia expressão ──────────────────────────────────────────────────────

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value> {
        match expr {
            Expr::Float(v) => Ok(Value::Float(*v)),
            Expr::Int(v)   => Ok(Value::Int(*v)),
            Expr::Bool(v)  => Ok(Value::Bool(*v)),
            Expr::Str(v)   => Ok(Value::Str(v.clone())),

            Expr::Var(name) => {
                self.env.get(name)
                    .cloned()
                    .ok_or_else(|| HayashiError::Runtime(format!("undefined variable '{name}'")))
            }

            Expr::Formula(_f) => {
                Err(HayashiError::Runtime("formula must be used inside an estimator call".into()))
            }

            Expr::BinOp { .. } | Expr::Neg(_) => {
                Err(HayashiError::Runtime(
                    "arithmetic expressions are only valid inside generate".into()
                ))
            }

            Expr::Call { func, args, opts } => {
                self.eval_call(func, args, opts)
            }

            Expr::Field { obj, field, args, opts } => {
                self.eval_field(obj, field, args, opts)
            }
        }
    }

    // ── Converte fórmula AST → string Greeners ────────────────────────────────

    fn formula_to_string(f: &Formula) -> String {
        let rhs_parts: Vec<String> = f.rhs.iter().map(|t| match t {
            RhsTerm::Var(v)              => v.clone(),
            RhsTerm::Categorical(v)      => format!("C({v})"),
            RhsTerm::Transform(fn_, v)   => format!("{fn_}({v})"),
            RhsTerm::Interaction(a, b)   => format!("{a}:{b}"),
        }).collect();

        let mut formula_str = if f.lhs.is_empty() {
            format!("~ {}", rhs_parts.join(" + "))
        } else {
            format!("{} ~ {}", f.lhs, rhs_parts.join(" + "))
        };
        if !f.fe.is_empty() {
            formula_str.push_str(" | ");
            formula_str.push_str(&f.fe.join(" + "));
        }
        formula_str
    }

    fn resolve_cov(opt_val: Option<&Value>) -> Result<CovarianceType> {
        match opt_val {
            None => Ok(CovarianceType::HC3),
            Some(Value::Str(s)) => match s.as_str() {
                "nonrobust" | "ols"  => Ok(CovarianceType::NonRobust),
                "HC1"                => Ok(CovarianceType::HC1),
                "HC2"                => Ok(CovarianceType::HC2),
                "HC3"                => Ok(CovarianceType::HC3),
                "HC4"                => Ok(CovarianceType::HC4),
                other => Err(HayashiError::Type(format!("unknown covariance type '{other}'"))),
            },
            _ => Err(HayashiError::Type("cov= must be a string".into())),
        }
    }

    // ── Funções built-in ──────────────────────────────────────────────────────

    fn eval_call(&mut self, func: &str, args: &[Expr], opts: &[Opt]) -> Result<Value> {
        // avalia opts primeiro
        let opt_map: HashMap<String, Value> = opts.iter()
            .map(|o| Ok((o.name.clone(), self.eval_expr(&o.value)?)))
            .collect::<Result<_>>()?;

        match func {
            // ── OLS ───────────────────────────────────────────────────────────
            "ols" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("ols() requires (formula, dataframe)".into()));
                }
                let formula_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("first argument must be a formula".into())),
                };
                let df_name = match &args[1] {
                    Expr::Var(name) => name.clone(),
                    _ => return Err(HayashiError::Type("second argument must be a DataFrame variable".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' is not a DataFrame"))),
                };
                let formula_str = Self::formula_to_string(&formula_ast);
                let cov = Self::resolve_cov(opt_map.get("cov"))?;

                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let (y, x) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = OLS::from_formula(&g_formula, &df, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let fitted = x.dot(&result.params);
                let residuals = &y - &fitted;

                Ok(Value::OlsResult(OlsModel {
                    result: Rc::new(result),
                    residuals,
                    x,
                }))
            }

            // ── IV / 2SLS ─────────────────────────────────────────────────────
            "iv" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "iv() requires (endog_formula, instrument_formula, dataframe)".into(),
                    ));
                }
                let endog_ast = match &args[0] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("first argument must be a formula".into())),
                };
                let instr_ast = match &args[1] {
                    Expr::Formula(f) => f.clone(),
                    _ => return Err(HayashiError::Type("second argument must be a formula".into())),
                };
                let df_name = match &args[2] {
                    Expr::Var(name) => name.clone(),
                    _ => return Err(HayashiError::Type("third argument must be a DataFrame variable".into())),
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(df)) => df.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df_name}' is not a DataFrame"))),
                };
                let cov = Self::resolve_cov(opt_map.get("cov"))?;

                let endog_str = Self::formula_to_string(&endog_ast);
                let instr_str = Self::formula_to_string(&instr_ast);

                let g_endog = GFormula::parse(&endog_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let g_instr = GFormula::parse(&instr_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let result = IV::from_formula(&g_endog, &g_instr, &df, cov)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::IvResult(Rc::new(result)))
            }

            // ── Logit ─────────────────────────────────────────────────────────
            "logit" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (_, x) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = Logit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::BinaryResult(BinaryModel { result: Rc::new(result), x }))
            }

            // ── Probit ────────────────────────────────────────────────────────
            "probit" => {
                let (formula_ast, df) = self.extract_binary_args(args)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let (_, x) = df.to_design_matrix(&g_formula)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let result = Probit::from_formula(&g_formula, &df)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                Ok(Value::BinaryResult(BinaryModel { result: Rc::new(result), x }))
            }

            // ── Fixed Effects ─────────────────────────────────────────────────
            "fe" => {
                let (formula_ast, df, id_col) = self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                // FE elimina o intercepto via within-transform; forçamos - 1
                // para evitar coluna de zeros pós-demeaning (singular matrix)
                let formula_no_const = if formula_str.contains("- 1") {
                    formula_str
                } else {
                    format!("{} - 1", formula_str)
                };
                let g_formula = GFormula::parse(&formula_no_const)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // tenta int; cai para string se não existir como int
                let result = if let Ok(ids) = df.get_int(&id_col) {
                    let ids_vec: Vec<i64> = ids.to_vec();
                    FixedEffects::from_formula(&g_formula, &df, &ids_vec)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                } else if let Ok(ids) = df.get_string(&id_col) {
                    let ids_vec: Vec<String> = ids.to_vec();
                    FixedEffects::from_formula(&g_formula, &df, &ids_vec)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?
                } else {
                    return Err(HayashiError::Runtime(
                        format!("column '{id_col}' not found or not usable as entity ID"),
                    ));
                };

                Ok(Value::PanelResult(Rc::new(result)))
            }

            // ── Random Effects ────────────────────────────────────────────────
            "re" => {
                let (formula_ast, df, id_col) = self.extract_panel_args(args, &opt_map)?;
                let formula_str = Self::formula_to_string(&formula_ast);
                let g_formula = GFormula::parse(&formula_str)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                let ids = df.get_int(&id_col)
                    .map_err(|_| HayashiError::Runtime(
                        format!("column '{id_col}' must be integer for re()"),
                    ))?;

                let result = RandomEffects::from_formula(&g_formula, &df, ids)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                Ok(Value::ReResult(Rc::new(result)))
            }

            // ── Diagnósticos ──────────────────────────────────────────────────
            "test" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime("test(model, name) requires 2 arguments".into()));
                }
                let model = self.eval_expr(&args[0])?;
                let test_name = match &args[1] {
                    Expr::Var(s) => s.clone(),
                    _ => return Err(HayashiError::Type("second argument must be a test name".into())),
                };

                let ols = match &model {
                    Value::OlsResult(m) => m.clone(),
                    _ => return Err(HayashiError::Type(
                        "test() currently supports OLS models only".into(),
                    )),
                };

                match test_name.as_str() {
                    "white" => {
                        match SpecificationTests::white_test(&ols.residuals, &ols.x) {
                            Ok((stat, p, df)) => {
                                println!("White Test for Heteroskedasticity");
                                println!("  LM statistic : {:.4}", stat);
                                println!("  p-value      : {:.4}", p);
                                println!("  df           : {}", df);
                                let verdict = if p < 0.05 { "Reject H0 — evidence of heteroskedasticity" }
                                              else { "Fail to reject H0 — no evidence of heteroskedasticity" };
                                println!("  Conclusion   : {}", verdict);
                            }
                            Err(e) => eprintln!("White test error: {e}"),
                        }
                    }
                    "bp" => {
                        match Diagnostics::breusch_pagan(&ols.residuals, &ols.x) {
                            Ok((stat, p)) => {
                                println!("Breusch-Pagan Test for Heteroskedasticity");
                                println!("  LM statistic : {:.4}", stat);
                                println!("  p-value      : {:.4}", p);
                                let verdict = if p < 0.05 { "Reject H0 — evidence of heteroskedasticity" }
                                              else { "Fail to reject H0 — no evidence of heteroskedasticity" };
                                println!("  Conclusion   : {}", verdict);
                            }
                            Err(e) => eprintln!("Breusch-Pagan test error: {e}"),
                        }
                    }
                    "dw" => {
                        let stat = Diagnostics::durbin_watson(&ols.residuals);
                        println!("Durbin-Watson Test for Autocorrelation");
                        println!("  DW statistic : {:.4}", stat);
                        let verdict = if stat < 1.5 { "Positive autocorrelation suspected" }
                                      else if stat > 2.5 { "Negative autocorrelation suspected" }
                                      else { "No strong evidence of autocorrelation" };
                        println!("  Conclusion   : {}", verdict);
                    }
                    other => return Err(HayashiError::Runtime(
                        format!("unknown test '{other}' — available: white, bp, dw"),
                    )),
                }

                Ok(Value::Nil)
            }

            // ── describe ─────────────────────────────────────────────────────
            "describe" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("describe() takes 1 argument".into()));
                }
                match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => {
                        println!("{}", df);
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Type("describe() requires a DataFrame".into())),
                }
            }

            // ── correlate ────────────────────────────────────────────────────
            "correlate" | "corr" | "pwcorr" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "correlate() requires a DataFrame as first argument".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };

                // variáveis pedidas ou todas as numéricas
                let names: Vec<String> = if args.len() > 1 {
                    args[1..].iter().map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variable list must be identifiers".into())),
                    }).collect::<Result<_>>()?
                } else {
                    use greeners::Column;
                    let mut ns: Vec<String> = df.column_names().into_iter().filter(|n| {
                        matches!(df.get_column(n), Ok(Column::Float(_)) | Ok(Column::Int(_)))
                    }).collect();
                    ns.sort();
                    ns
                };

                if names.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "correlate() needs at least 2 numeric variables".into(),
                    ));
                }

                let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                let sub = df.select(&refs)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let mat = sub.corr()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // corr() ordena colunas alfabeticamente — sincronizar com a matriz
                let mut sorted_names = names.clone();
                sorted_names.sort();

                let col_w = 10usize;
                let row_label_w = 16usize;
                let trunc = |s: &str, w: usize| {
                    if s.len() > w { s[..w].to_string() } else { s.to_string() }
                };

                // cabeçalho
                print!("{:>width$} |", "", width = row_label_w);
                for name in &sorted_names {
                    print!(" {:>width$}", trunc(name, col_w), width = col_w);
                }
                println!();
                println!("{}-+{}", "-".repeat(row_label_w), "-".repeat((col_w + 1) * sorted_names.len()));

                // triângulo inferior (estilo Stata)
                for (i, row_name) in sorted_names.iter().enumerate() {
                    print!("{:>width$} |", trunc(row_name, row_label_w), width = row_label_w);
                    for j in 0..=i {
                        print!(" {:>10.4}", mat[[i, j]]);
                    }
                    println!();
                }
                println!();
                Ok(Value::Nil)
            }

            // ── summarize ────────────────────────────────────────────────────
            "summarize" | "sum" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "summarize() requires a DataFrame as first argument".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };

                // variáveis pedidas (args[1..]) ou todas as colunas numéricas
                let requested: Vec<String> = if args.len() > 1 {
                    args[1..].iter().map(|a| match a {
                        Expr::Var(name) => Ok(name.clone()),
                        Expr::Str(s)    => Ok(s.clone()),
                        _ => Err(HayashiError::Type(
                            "summarize() variable list must be identifiers".into(),
                        )),
                    }).collect::<Result<_>>()?
                } else {
                    let mut names = df.column_names();
                    names.sort();
                    names
                };

                // cabeçalho
                println!(
                    "\n{:<16} {:>9} {:>12} {:>12} {:>12} {:>12}",
                    "Variable", "Obs", "Mean", "Std. Dev.", "Min", "Max"
                );
                println!("{}", "-".repeat(77));

                for name in &requested {
                    use greeners::Column;
                    let col = df.get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                    // extrai valores f64 conforme o tipo da coluna
                    let vals: Vec<f64> = match col {
                        Column::Float(arr) => arr.iter().copied()
                            .filter(|x| x.is_finite()).collect(),
                        Column::Int(arr) => arr.iter().map(|&x| x as f64).collect(),
                        _ => {
                            // coluna não numérica: exibe linha vazia
                            println!("{:<16} {:>9}", name, "(non-numeric)");
                            continue;
                        }
                    };

                    let n = vals.len();
                    if n == 0 {
                        println!("{:<16} {:>9}", name, 0);
                        continue;
                    }

                    let mean = vals.iter().sum::<f64>() / n as f64;
                    let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / (n as f64 - 1.0).max(1.0);
                    let sd  = variance.sqrt();
                    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    println!(
                        "{:<16} {:>9} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                        name, n, mean, sd, min, max
                    );
                }
                println!();
                Ok(Value::Nil)
            }

            // ── ttest ────────────────────────────────────────────────────────
            "ttest" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("ttest() requires a DataFrame".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };

                let get_finite = |df: &DataFrame, col: &str| -> Result<Vec<f64>> {
                    use greeners::Column;
                    match df.get_column(col) {
                        Ok(Column::Float(a)) => Ok(a.iter().filter(|x| x.is_finite()).copied().collect()),
                        Ok(Column::Int(a))   => Ok(a.iter().map(|&x| x as f64).collect()),
                        _ => Err(HayashiError::Type(format!("'{col}' is not numeric"))),
                    }
                };

                let stats = |v: &[f64]| -> (f64, f64, f64) { // (mean, sd, n)
                    let n = v.len() as f64;
                    let m = v.iter().sum::<f64>() / n;
                    let s = if n > 1.0 { (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() } else { f64::NAN };
                    (m, s, n)
                };

                // ── um argumento variável → uni-amostral ou por grupo ─────────
                if args.len() >= 2 {
                    let var1 = match &args[1] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => return Err(HayashiError::Type("variable name must be an identifier".into())),
                    };

                    // ── PAREADO: ttest(df, v1, v2, paired=true) ──────────────
                    if args.len() >= 3 && matches!(opt_map.get("paired"), Some(Value::Bool(true))) {
                        let var2 = match &args[2] {
                            Expr::Var(n) | Expr::Str(n) => n.clone(),
                            _ => return Err(HayashiError::Type("variable name must be an identifier".into())),
                        };
                        let v1 = get_finite(&df, &var1)?;
                        let v2 = get_finite(&df, &var2)?;
                        if v1.len() != v2.len() {
                            return Err(HayashiError::Runtime("paired ttest requires equal-length columns".into()));
                        }
                        let diff: Vec<f64> = v1.iter().zip(&v2).map(|(a, b)| a - b).collect();
                        let (m, s, n) = stats(&diff);
                        let se = s / n.sqrt();
                        let t  = m / se;
                        let df_t = n - 1.0;
                        let p  = t_pvalue_two(t, df_t);
                        let tc = t_critical_95(df_t);
                        println!("\nPaired t-test: {var1} - {var2}");
                        println!("{}", "─".repeat(62));
                        println!("{:<14} {:>6}  {:>10}  {:>10}  {:>10}", "Variable", "Obs", "Mean", "Std. Err.", "[95% CI]");
                        println!("{}", "─".repeat(62));
                        println!("{:<14} {:>6.0}  {:>10.4}  {:>10.4}  [{:.4}, {:.4}]",
                                 format!("{var1}-{var2}"), n, m, se, m - tc * se, m + tc * se);
                        println!("{}", "─".repeat(62));
                        println!("H0: mean(diff) = 0   t = {t:.4}   df = {df_t:.0}   p = {p:.4}");
                        println!();

                    // ── DOIS GRUPOS: ttest(df, var, by=group) ────────────────
                    } else if let Some(Value::Str(by_col)) = opt_map.get("by") {
                        let by_col = by_col.clone();
                        let vals   = get_finite(&df, &var1)?;
                        let groups = Self::col_to_strings(&df, &by_col)?;

                        let mut group_data: HashMap<String, Vec<f64>> = HashMap::new();
                        for (i, g) in groups.iter().enumerate() {
                            group_data.entry(g.clone()).or_default().push(vals[i]);
                        }
                        let mut gkeys: Vec<String> = group_data.keys().cloned().collect();
                        if gkeys.len() != 2 {
                            return Err(HayashiError::Runtime(
                                format!("two-sample ttest requires exactly 2 groups, got {}", gkeys.len())
                            ));
                        }
                        let all_num = gkeys.iter().all(|s| s.parse::<f64>().is_ok());
                        if all_num { gkeys.sort_by(|a, b| a.parse::<f64>().unwrap().partial_cmp(&b.parse::<f64>().unwrap()).unwrap()); }
                        else       { gkeys.sort(); }

                        let v1 = &group_data[&gkeys[0]];
                        let v2 = &group_data[&gkeys[1]];
                        let (m1, s1, n1) = stats(v1);
                        let (m2, s2, n2) = stats(v2);

                        // Welch's t
                        let se1sq = s1 * s1 / n1;
                        let se2sq = s2 * s2 / n2;
                        let se    = (se1sq + se2sq).sqrt();
                        let t     = (m1 - m2) / se;
                        let df_t  = (se1sq + se2sq).powi(2) /
                                    (se1sq.powi(2) / (n1 - 1.0) + se2sq.powi(2) / (n2 - 1.0));
                        let p     = t_pvalue_two(t, df_t);
                        let tc    = t_critical_95(df_t);

                        println!("\nTwo-sample t-test (Welch): {var1} by {by_col}");
                        println!("{}", "─".repeat(68));
                        println!("{:<10} {:>6}  {:>10}  {:>10}  {:>10}  {:>10}", "Group", "Obs", "Mean", "Std. Err.", "Std. Dev.", "[95% CI]");
                        println!("{}", "─".repeat(68));
                        for (g, m, s, n, se_g) in [
                            (&gkeys[0], m1, s1, n1, (s1*s1/n1).sqrt()),
                            (&gkeys[1], m2, s2, n2, (s2*s2/n2).sqrt()),
                        ] {
                            println!("{:<10} {:>6.0}  {:>10.4}  {:>10.4}  {:>10.4}  [{:.4}, {:.4}]",
                                     g, n, m, se_g, s, m - tc * se_g, m + tc * se_g);
                        }
                        println!("{}", "─".repeat(68));
                        println!("diff = mean({}) - mean({})", gkeys[0], gkeys[1]);
                        println!("H0: diff = 0   Welch's t = {t:.4}   df = {df_t:.2}   p = {p:.4}");
                        println!();

                    // ── UNI-AMOSTRAL: ttest(df, var, mu=0) ───────────────────
                    } else {
                        let mu = match opt_map.get("mu") {
                            Some(Value::Float(f)) => *f,
                            Some(Value::Int(i))   => *i as f64,
                            None => 0.0,
                            _ => return Err(HayashiError::Type("mu= must be numeric".into())),
                        };
                        let v    = get_finite(&df, &var1)?;
                        let (m, s, n) = stats(&v);
                        let se   = s / n.sqrt();
                        let t    = (m - mu) / se;
                        let df_t = n - 1.0;
                        let p    = t_pvalue_two(t, df_t);
                        let tc   = t_critical_95(df_t);

                        println!("\nOne-sample t-test: {var1}   H0: mean = {mu}");
                        println!("{}", "─".repeat(62));
                        println!("{:<14} {:>6}  {:>10}  {:>10}  {:>10}", "Variable", "Obs", "Mean", "Std. Err.", "[95% CI]");
                        println!("{}", "─".repeat(62));
                        println!("{:<14} {:>6.0}  {:>10.4}  {:>10.4}  [{:.4}, {:.4}]",
                                 var1, n, m, se, m - tc * se, m + tc * se);
                        println!("{}", "─".repeat(62));
                        println!("t = {t:.4}   df = {df_t:.0}   p = {p:.4}");
                        println!();
                    }
                } else {
                    return Err(HayashiError::Runtime(
                        "ttest() requires a variable name as second argument".into(),
                    ));
                }

                Ok(Value::Nil)
            }

            // ── collapse ─────────────────────────────────────────────────────
            "collapse" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "collapse() requires (df, func, [vars...], by=col)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let func_name = match &args[1] {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(HayashiError::Type(
                        "second argument must be a function name (mean, sum, min, max, count, sd, median)".into(),
                    )),
                };
                let by_col = match opt_map.get("by") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err(HayashiError::Runtime("collapse() requires by=colname".into())),
                };

                // validar função antes de qualquer cálculo
                match func_name.as_str() {
                    "mean" | "sum" | "min" | "max" | "count" | "sd" | "median" => {}
                    other => return Err(HayashiError::Runtime(format!(
                        "unknown aggregation '{other}' — use: mean, sum, min, max, count, sd, median"
                    ))),
                }

                // variáveis a agregar: args[2..] ou todas as numéricas exceto by
                let agg_vars: Vec<String> = if args.len() > 2 {
                    args[2..].iter().map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variable names must be identifiers".into())),
                    }).collect::<Result<_>>()?
                } else {
                    use greeners::Column;
                    df.column_names().into_iter()
                        .filter(|n| n != &by_col && matches!(df.get_column(n), Ok(Column::Float(_)) | Ok(Column::Int(_))))
                        .collect()
                };

                // dados das colunas numéricas a agregar
                let col_data: Vec<Vec<f64>> = agg_vars.iter().map(|col| {
                    use greeners::Column;
                    match df.get_column(col) {
                        Ok(Column::Float(a)) => Ok(a.to_vec()),
                        Ok(Column::Int(a))   => Ok(a.iter().map(|&x| x as f64).collect()),
                        _ => Err(HayashiError::Type(format!("'{col}' is not numeric"))),
                    }
                }).collect::<Result<_>>()?;

                // agrupa índices de linha por valor de by
                let by_strs = Self::col_to_strings(&df, &by_col)?;
                let n_obs = df.n_rows();
                let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
                for (i, v) in by_strs.iter().enumerate() {
                    groups.entry(v.clone()).or_default().push(i);
                }

                // ordena chaves de grupo
                let mut keys: Vec<String> = groups.keys().cloned().collect();
                let keys_numeric = keys.iter().all(|s| s.parse::<f64>().is_ok());
                if keys_numeric {
                    keys.sort_by(|a, b| a.parse::<f64>().unwrap().partial_cmp(&b.parse::<f64>().unwrap()).unwrap());
                } else {
                    keys.sort();
                }

                // função de agregação
                let agg = |vals: &[f64]| -> f64 {
                    match func_name.as_str() {
                        "count" => vals.len() as f64,
                        _ => {
                            let fin: Vec<f64> = vals.iter().filter(|x| x.is_finite()).copied().collect();
                            let n = fin.len();
                            if n == 0 { return f64::NAN; }
                            match func_name.as_str() {
                                "mean"   => fin.iter().sum::<f64>() / n as f64,
                                "sum"    => fin.iter().sum::<f64>(),
                                "min"    => fin.iter().cloned().fold(f64::INFINITY,     f64::min),
                                "max"    => fin.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                                "sd"     => {
                                    if n < 2 { return f64::NAN; }
                                    let m = fin.iter().sum::<f64>() / n as f64;
                                    (fin.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
                                }
                                "median" => {
                                    let mut s = fin.clone();
                                    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                    if n % 2 == 0 { (s[n/2 - 1] + s[n/2]) / 2.0 } else { s[n/2] }
                                }
                                _ => f64::NAN,
                            }
                        }
                    }
                };

                // constrói o DataFrame resultado
                let mut builder = DataFrame::builder();

                // coluna by (numérica ou string)
                use greeners::Column;
                if matches!(df.get_column(&by_col), Ok(Column::Float(_)) | Ok(Column::Int(_))) {
                    let vals: Vec<f64> = keys.iter()
                        .map(|k| k.parse::<f64>().unwrap_or(f64::NAN))
                        .collect();
                    builder = builder.add_column(&by_col, vals);
                } else {
                    builder = builder.add_string(&by_col, keys.clone());
                }

                // colunas agregadas
                for (ci, col_name) in agg_vars.iter().enumerate() {
                    let vals: Vec<f64> = keys.iter().map(|key| {
                        let subset: Vec<f64> = groups[key].iter().map(|&i| col_data[ci][i]).collect();
                        agg(&subset)
                    }).collect();
                    builder = builder.add_column(col_name, vals);
                }

                let new_df = builder.build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({} groups from {} observations)", keys.len(), n_obs);
                Ok(Value::DataFrame(new_df))
            }

            // ── append ───────────────────────────────────────────────────────
            "append" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime("append() requires (df1, df2)".into()));
                }
                let df1 = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let df2 = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("second argument must be a DataFrame".into())),
                };

                let names1 = df1.column_names();
                let names2 = df2.column_names();
                let set1: std::collections::HashSet<&str> = names1.iter().map(String::as_str).collect();
                let n1 = df1.n_rows();
                let n2 = df2.n_rows();

                // união de colunas: ordem de df1 primeiro, depois novas de df2
                let mut all_names = names1.clone();
                for n in &names2 { if !set1.contains(n.as_str()) { all_names.push(n.clone()); } }

                let get_num = |df: &DataFrame, col: &str, n: usize| -> Vec<f64> {
                    use greeners::Column;
                    match df.get_column(col) {
                        Ok(Column::Float(a)) => a.to_vec(),
                        Ok(Column::Int(a))   => a.iter().map(|&x| x as f64).collect(),
                        _                    => vec![f64::NAN; n],
                    }
                };
                let get_str = |df: &DataFrame, col: &str, n: usize| -> Vec<String> {
                    df.get_string(col).map(|a| a.to_vec()).unwrap_or_else(|_| vec![String::new(); n])
                };

                let mut builder = DataFrame::builder();
                for col in &all_names {
                    use greeners::Column;
                    let in1 = names1.contains(col);
                    let in2 = names2.contains(col);
                    let is_num = if in1 { matches!(df1.get_column(col), Ok(Column::Float(_)) | Ok(Column::Int(_))) }
                                 else   { matches!(df2.get_column(col), Ok(Column::Float(_)) | Ok(Column::Int(_))) };
                    if is_num {
                        let p1 = if in1 { get_num(&df1, col, n1) } else { vec![f64::NAN; n1] };
                        let p2 = if in2 { get_num(&df2, col, n2) } else { vec![f64::NAN; n2] };
                        builder = builder.add_column(col, p1.into_iter().chain(p2).collect::<Vec<_>>());
                    } else {
                        let p1 = if in1 { get_str(&df1, col, n1) } else { vec![String::new(); n1] };
                        let p2 = if in2 { get_str(&df2, col, n2) } else { vec![String::new(); n2] };
                        builder = builder.add_string(col, p1.into_iter().chain(p2).collect::<Vec<_>>());
                    }
                }

                let new_df = builder.build().map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("({} + {} = {} observations)", n1, n2, n1 + n2);
                Ok(Value::DataFrame(new_df))
            }

            // ── merge ─────────────────────────────────────────────────────────
            "merge" => {
                if args.len() != 2 {
                    return Err(HayashiError::Runtime(
                        "merge() requires (df1, df2, key=varname [, type=left|inner|outer])".into(),
                    ));
                }
                let df1 = match self.eval_expr(&args[0])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let df2 = match self.eval_expr(&args[1])? {
                    Value::DataFrame(d) => d,
                    _ => return Err(HayashiError::Type("second argument must be a DataFrame".into())),
                };
                let key_col = match opt_map.get("key") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err(HayashiError::Runtime("merge() requires key=colname".into())),
                };
                let join_type = match opt_map.get("type") {
                    Some(Value::Str(s)) => s.clone(),
                    None => "left".to_string(),
                    _ => return Err(HayashiError::Runtime("type= must be a string".into())),
                };

                // índice de busca no df2: key_str → primeiro índice de linha
                let key2_strs = Self::col_to_strings(&df2, &key_col)?;
                let mut lookup: HashMap<String, usize> = HashMap::new();
                for (j, v) in key2_strs.iter().enumerate().rev() {
                    lookup.insert(v.clone(), j); // rev para ficar com o primeiro
                }

                let key1_strs = Self::col_to_strings(&df1, &key_col)?;
                let n1 = df1.n_rows();
                let n2 = df2.n_rows();

                // pares (idx_df1, idx_df2) para cada linha do resultado
                type RowPair = (Option<usize>, Option<usize>);
                let mut result_rows: Vec<RowPair> = (0..n1)
                    .map(|i| (Some(i), lookup.get(&key1_strs[i]).copied()))
                    .collect();

                match join_type.as_str() {
                    "left" => {}
                    "inner" => result_rows.retain(|(_, r2)| r2.is_some()),
                    "outer" | "full" => {
                        let matched: std::collections::HashSet<usize> =
                            result_rows.iter().filter_map(|(_, r2)| *r2).collect();
                        for j in 0..n2 {
                            if !matched.contains(&j) { result_rows.push((None, Some(j))); }
                        }
                    }
                    other => return Err(HayashiError::Runtime(
                        format!("unknown merge type '{other}' — use: left, inner, outer"),
                    )),
                }

                let names1 = df1.column_names();
                let names2 = df2.column_names();
                let set1: std::collections::HashSet<&str> = names1.iter().map(String::as_str).collect();

                // colunas extra de df2 (exclui key; sufixo _2 em colisão)
                let extra: Vec<(String, String)> = names2.iter()
                    .filter(|n| *n != &key_col)
                    .map(|n| (n.clone(), if set1.contains(n.as_str()) { format!("{n}_2") } else { n.clone() }))
                    .collect();

                let get_num = |df: &DataFrame, col: &str| -> Vec<f64> {
                    use greeners::Column;
                    match df.get_column(col) {
                        Ok(Column::Float(a)) => a.to_vec(),
                        Ok(Column::Int(a))   => a.iter().map(|&x| x as f64).collect(),
                        _                    => vec![],
                    }
                };
                let get_str_col = |df: &DataFrame, col: &str| -> Vec<String> {
                    df.get_string(col).map(|a| a.to_vec()).unwrap_or_default()
                };

                let mut builder = DataFrame::builder();

                // colunas de df1
                for col in &names1 {
                    use greeners::Column;
                    if matches!(df1.get_column(col), Ok(Column::Float(_)) | Ok(Column::Int(_))) {
                        let src = get_num(&df1, col);
                        builder = builder.add_column(col,
                            result_rows.iter().map(|(r1, _)| r1.map_or(f64::NAN, |i| src[i])).collect::<Vec<_>>());
                    } else {
                        let src = get_str_col(&df1, col);
                        builder = builder.add_string(col,
                            result_rows.iter().map(|(r1, _)| r1.map_or(String::new(), |i| src[i].clone())).collect::<Vec<_>>());
                    }
                }

                // colunas extras de df2
                for (src_col, out_col) in &extra {
                    use greeners::Column;
                    if matches!(df2.get_column(src_col), Ok(Column::Float(_)) | Ok(Column::Int(_))) {
                        let src = get_num(&df2, src_col);
                        builder = builder.add_column(out_col,
                            result_rows.iter().map(|(_, r2)| r2.map_or(f64::NAN, |j| src[j])).collect::<Vec<_>>());
                    } else {
                        let src = get_str_col(&df2, src_col);
                        builder = builder.add_string(out_col,
                            result_rows.iter().map(|(_, r2)| r2.map_or(String::new(), |j| src[j].clone())).collect::<Vec<_>>());
                    }
                }

                // indicador _merge: 3=matched, 1=left only, 2=right only
                builder = builder.add_column("_merge",
                    result_rows.iter().map(|(r1, r2)| match (r1, r2) {
                        (Some(_), Some(_)) => 3.0,
                        (Some(_), None)    => 1.0,
                        _                  => 2.0,
                    }).collect::<Vec<_>>());

                let new_df = builder.build().map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let n_matched = result_rows.iter().filter(|(_, r2)| r2.is_some()).count();
                let n_out = result_rows.len();
                println!("({n_matched} matched, {} not matched, {n_out} total)", n_out - n_matched);
                Ok(Value::DataFrame(new_df))
            }

            // ── sort ─────────────────────────────────────────────────────────
            "sort" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "sort() requires (dataframe, var1, ...)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let sort_vars: Vec<String> = args[1..].iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("sort keys must be identifiers".into())),
                    })
                    .collect::<Result<_>>()?;
                let desc = matches!(opt_map.get("desc"), Some(Value::Bool(true)));

                // extrai chaves de ordenação
                enum SortKey { Num(Vec<f64>), Str(Vec<String>) }
                let keys: Vec<SortKey> = sort_vars.iter().map(|v| {
                    use greeners::Column;
                    match df.get_column(v) {
                        Ok(Column::Float(arr)) => Ok(SortKey::Num(arr.to_vec())),
                        Ok(Column::Int(arr))   => Ok(SortKey::Num(arr.iter().map(|&x| x as f64).collect())),
                        _ => df.get_string(v)
                                .map(|arr| SortKey::Str(arr.to_vec()))
                                .map_err(|_| HayashiError::Runtime(format!("column '{v}' not found"))),
                    }
                }).collect::<Result<_>>()?;

                let n = df.n_rows();
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| {
                    use std::cmp::Ordering;
                    for key in &keys {
                        let ord = match key {
                            SortKey::Num(v) => match (v[a].is_nan(), v[b].is_nan()) {
                                (true,  true)  => Ordering::Equal,
                                (true,  false) => Ordering::Greater,
                                (false, true)  => Ordering::Less,
                                (false, false) => v[a].partial_cmp(&v[b]).unwrap(),
                            },
                            SortKey::Str(v) => v[a].cmp(&v[b]),
                        };
                        if ord != Ordering::Equal {
                            return if desc { ord.reverse() } else { ord };
                        }
                    }
                    Ordering::Equal
                });

                let all_names = df.column_names();
                let mut builder = DataFrame::builder();
                for col_name in &all_names {
                    use greeners::Column;
                    match df.get_column(col_name) {
                        Ok(Column::Float(arr)) => {
                            builder = builder.add_column(col_name, idx.iter().map(|&i| arr[i]).collect::<Vec<_>>());
                        }
                        Ok(Column::Int(arr)) => {
                            builder = builder.add_column(col_name, idx.iter().map(|&i| arr[i] as f64).collect::<Vec<_>>());
                        }
                        _ => {
                            if let Ok(arr) = df.get_string(col_name) {
                                let v = arr.to_vec();
                                builder = builder.add_string(col_name, idx.iter().map(|&i| v[i].clone()).collect::<Vec<_>>());
                            }
                        }
                    }
                }

                let new_df = builder.build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                println!("({n} observations sorted)");
                Ok(Value::DataFrame(new_df))
            }

            // ── list ──────────────────────────────────────────────────────────
            "list" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("list() requires a DataFrame".into()));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };

                // args[1]: Int → nrows; Ident → coluna
                let mut n_show = 10usize;
                let mut col_names: Vec<String> = Vec::new();

                for arg in &args[1..] {
                    match arg {
                        Expr::Int(n)            => n_show = (*n).max(0) as usize,
                        Expr::Var(n) | Expr::Str(n) => col_names.push(n.clone()),
                        _ => return Err(HayashiError::Type("list() arguments must be identifiers or row count".into())),
                    }
                }
                if col_names.is_empty() {
                    col_names = df.column_names();
                }

                let n_rows = n_show.min(df.n_rows());

                // extrai dados das colunas
                let cols_data: Vec<(String, Vec<String>)> = col_names.iter().map(|name| {
                    use greeners::Column;
                    let vals: Vec<String> = match df.get_column(name) {
                        Ok(Column::Float(arr)) => arr.iter().take(n_rows).map(|x| {
                            if x.is_nan() { ".".into() }
                            else if x.fract() == 0.0 && x.abs() < 1e14 { format!("{}", *x as i64) }
                            else { format!("{:.4}", x) }
                        }).collect(),
                        Ok(Column::Int(arr)) => arr.iter().take(n_rows).map(|x| x.to_string()).collect(),
                        _ => df.get_string(name).map(|a| a.to_vec().into_iter().take(n_rows).collect())
                                .unwrap_or_else(|_| vec!["?".into(); n_rows]),
                    };
                    (name.clone(), vals)
                }).collect();

                // larguras de coluna
                let row_num_w = n_rows.to_string().len().max(1);
                let widths: Vec<usize> = cols_data.iter().map(|(name, vals)| {
                    vals.iter().map(|v| v.len()).max().unwrap_or(0).max(name.len()) + 1
                }).collect();

                // cabeçalho
                print!("{:>rw$} |", "", rw = row_num_w);
                for (i, (name, _)) in cols_data.iter().enumerate() {
                    print!(" {:>w$}", name, w = widths[i]);
                }
                println!();
                println!("{}-+{}", "-".repeat(row_num_w), "-".repeat(widths.iter().sum::<usize>() + widths.len()));

                // linhas
                for r in 0..n_rows {
                    print!("{:>rw$} |", r + 1, rw = row_num_w);
                    for (i, (_, vals)) in cols_data.iter().enumerate() {
                        print!(" {:>w$}", vals[r], w = widths[i]);
                    }
                    println!();
                }
                if df.n_rows() > n_rows {
                    println!("  ({} more observations not shown)", df.n_rows() - n_rows);
                }
                println!();
                Ok(Value::Nil)
            }

            // ── dropna ───────────────────────────────────────────────────────
            "dropna" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "dropna() requires a DataFrame as first argument".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };

                // colunas a verificar: as listadas ou todas as float
                let check: Vec<String> = if args.len() > 1 {
                    args[1..].iter().map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variable names must be identifiers".into())),
                    }).collect::<Result<_>>()?
                } else {
                    use greeners::Column;
                    df.column_names().into_iter()
                        .filter(|n| matches!(df.get_column(n), Ok(Column::Float(_))))
                        .collect()
                };

                let n = df.n_rows();
                let mut keep = vec![true; n];

                for col_name in &check {
                    use greeners::Column;
                    if let Ok(Column::Float(arr)) = df.get_column(col_name) {
                        for (i, &v) in arr.iter().enumerate() {
                            if v.is_nan() { keep[i] = false; }
                        }
                    }
                }

                let n_drop = keep.iter().filter(|&&k| !k).count();
                let n_kept = n - n_drop;

                // reconstrói o DataFrame filtrando as linhas
                let all_names = df.column_names();
                let mut builder = DataFrame::builder();

                for col_name in &all_names {
                    use greeners::Column;
                    match df.get_column(col_name) {
                        Ok(Column::Float(arr)) => {
                            let vals: Vec<f64> = arr.iter().enumerate()
                                .filter(|(i, _)| keep[*i])
                                .map(|(_, &v)| v)
                                .collect();
                            builder = builder.add_column(col_name, vals);
                        }
                        Ok(Column::Int(arr)) => {
                            let vals: Vec<f64> = arr.iter().enumerate()
                                .filter(|(i, _)| keep[*i])
                                .map(|(_, &v)| v as f64)
                                .collect();
                            builder = builder.add_column(col_name, vals);
                        }
                        _ => {
                            if let Ok(arr) = df.get_string(col_name) {
                                let vals: Vec<String> = arr.to_vec().into_iter().enumerate()
                                    .filter(|(i, _)| keep[*i])
                                    .map(|(_, v)| v)
                                    .collect();
                                builder = builder.add_string(col_name, vals);
                            }
                        }
                    }
                }

                let new_df = builder.build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({n_drop} observations dropped, {n_kept} remaining)");
                Ok(Value::DataFrame(new_df))
            }

            // ── rename ───────────────────────────────────────────────────────
            "rename" => {
                if args.len() != 3 {
                    return Err(HayashiError::Runtime(
                        "rename() requires (dataframe, oldname, newname)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let old = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("oldname must be an identifier".into())),
                };
                let new = match &args[2] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("newname must be an identifier".into())),
                };

                let all_names = df.column_names();
                if !all_names.contains(&old) {
                    return Err(HayashiError::Runtime(
                        format!("column '{old}' not found in DataFrame"),
                    ));
                }

                let mut builder = DataFrame::builder();
                for col_name in &all_names {
                    let out_name = if col_name == &old { &new } else { col_name };
                    use greeners::Column;
                    match df.get_column(col_name) {
                        Ok(Column::Float(arr)) => {
                            builder = builder.add_column(out_name, arr.to_vec());
                        }
                        Ok(Column::Int(arr)) => {
                            let vals: Vec<f64> = arr.iter().map(|&v| v as f64).collect();
                            builder = builder.add_column(out_name, vals);
                        }
                        _ => {
                            if let Ok(arr) = df.get_string(col_name) {
                                builder = builder.add_string(out_name, arr.to_vec());
                            }
                        }
                    }
                }

                let new_df = builder.build()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({old} → {new})");
                Ok(Value::DataFrame(new_df))
            }

            // ── drop ─────────────────────────────────────────────────────────
            "drop" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "drop() requires (dataframe, var1, ...)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let drop_names: std::collections::HashSet<String> = args[1..].iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variable names must be identifiers".into())),
                    })
                    .collect::<Result<_>>()?;

                let all = df.column_names();
                let keep: Vec<&str> = all.iter()
                    .filter(|n| !drop_names.contains(*n))
                    .map(String::as_str)
                    .collect();

                let new_df = df.select(&keep)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({} variables dropped, {} remaining)", drop_names.len(), keep.len());
                Ok(Value::DataFrame(new_df))
            }

            // ── keep ──────────────────────────────────────────────────────────
            "keep" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "keep() requires (dataframe, var1, ...)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let keep_names: Vec<String> = args[1..].iter()
                    .map(|a| match a {
                        Expr::Var(n) | Expr::Str(n) => Ok(n.clone()),
                        _ => Err(HayashiError::Type("variable names must be identifiers".into())),
                    })
                    .collect::<Result<_>>()?;

                let refs: Vec<&str> = keep_names.iter().map(String::as_str).collect();
                let n_before = df.column_names().len();
                let new_df = df.select(&refs)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                println!("({} variables kept, {} dropped)", refs.len(), n_before - refs.len());
                Ok(Value::DataFrame(new_df))
            }

            // ── tabulate ─────────────────────────────────────────────────────
            "tabulate" | "tab" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "tabulate() requires (dataframe, varname) or (dataframe, var1, var2)".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    _ => return Err(HayashiError::Type("first argument must be a DataFrame".into())),
                };
                let var1 = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => return Err(HayashiError::Type("variable name must be an identifier".into())),
                };

                if args.len() == 2 {
                    Self::tabulate_one(&df, &var1)?;
                } else {
                    let var2 = match &args[2] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => return Err(HayashiError::Type("variable name must be an identifier".into())),
                    };
                    let do_chi2 = matches!(opt_map.get("chi2"), Some(Value::Bool(true)));
                    Self::tabulate_two(&df, &var1, &var2, do_chi2)?;
                }

                Ok(Value::Nil)
            }

            other => Err(HayashiError::Runtime(format!("unknown function '{other}'"))),
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn extract_panel_args(
        &mut self,
        args: &[Expr],
        opt_map: &HashMap<String, Value>,
    ) -> Result<(Formula, DataFrame, String)> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime(
                "panel estimator requires (formula, dataframe, id=col)".into(),
            ));
        }
        let formula_ast = match &args[0] {
            Expr::Formula(f) => f.clone(),
            _ => return Err(HayashiError::Type("first argument must be a formula".into())),
        };
        let df_name = match &args[1] {
            Expr::Var(name) => name.clone(),
            _ => return Err(HayashiError::Type("second argument must be a DataFrame variable".into())),
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(df)) => df.clone(),
            _ => return Err(HayashiError::Runtime(format!("'{df_name}' is not a DataFrame"))),
        };
        let id_col = match opt_map.get("id") {
            Some(Value::Str(s)) => s.clone(),
            _ => return Err(HayashiError::Runtime("panel estimator requires id=column_name".into())),
        };
        Ok((formula_ast, df, id_col))
    }

    // ── Extrai coluna como Vec<String> (para tabulate) ────────────────────────

    fn col_to_strings(df: &DataFrame, name: &str) -> Result<Vec<String>> {
        use greeners::Column;
        match df.get_column(name) {
            Ok(Column::Int(arr)) => Ok(arr.iter().map(|x| x.to_string()).collect()),
            Ok(Column::Float(arr)) => Ok(arr.iter().map(|x| {
                if x.is_nan() { ".".to_string() }
                else if x.fract() == 0.0 && x.abs() < 1e14 { format!("{}", *x as i64) }
                else { format!("{:.4}", x) }
            }).collect()),
            _ => df.get_string(name)
                    .map(|arr| arr.to_vec())
                    .map_err(|_| HayashiError::Runtime(
                        format!("column '{name}' not found or has unsupported type for tabulate")
                    )),
        }
    }

    // ── Tabela de frequências (uni-variada) ───────────────────────────────────

    fn tabulate_one(df: &DataFrame, var: &str) -> Result<()> {
        let vals = Self::col_to_strings(df, var)?;
        let n = vals.len();

        let mut counts: HashMap<String, usize> = HashMap::new();
        for v in &vals { *counts.entry(v.clone()).or_insert(0) += 1; }

        let mut unique: Vec<String> = counts.keys().cloned().collect();
        let all_num = unique.iter().all(|s| s.parse::<f64>().is_ok());
        if all_num {
            unique.sort_by(|a, b| {
                a.parse::<f64>().unwrap().partial_cmp(&b.parse::<f64>().unwrap()).unwrap()
            });
        } else {
            unique.sort();
        }

        let label_w = var.len().max(12).max(unique.iter().map(|s| s.len()).max().unwrap_or(0)) + 2;
        let sep = format!("{}-+{}", "-".repeat(label_w), "-".repeat(36));

        println!("\n{:>lw$} | {:>10}  {:>10}  {:>10}", var, "Freq.", "Percent", "Cum.",
                 lw = label_w);
        println!("{sep}");

        let mut cum = 0.0_f64;
        for key in &unique {
            let freq = counts[key];
            let pct  = freq as f64 / n as f64 * 100.0;
            cum += pct;
            println!("{:>lw$} | {:>10}  {:>10.2}  {:>10.2}", key, freq, pct, cum,
                     lw = label_w);
        }
        println!("{sep}");
        println!("{:>lw$} | {:>10}  {:>10.2}", "Total", n, 100.0_f64, lw = label_w);
        println!();
        Ok(())
    }

    // ── Tabela cruzada (bi-variada, opcional chi2) ────────────────────────────

    fn tabulate_two(df: &DataFrame, row_var: &str, col_var: &str, do_chi2: bool) -> Result<()> {
        let rows = Self::col_to_strings(df, row_var)?;
        let cols = Self::col_to_strings(df, col_var)?;

        if rows.len() != cols.len() {
            return Err(HayashiError::Runtime("columns have different lengths".into()));
        }

        // valores únicos, ordenados
        let sort_strs = |mut v: Vec<String>| -> Vec<String> {
            let all_num = v.iter().all(|s| s.parse::<f64>().is_ok());
            if all_num {
                v.sort_by(|a, b| a.parse::<f64>().unwrap()
                    .partial_cmp(&b.parse::<f64>().unwrap()).unwrap());
            } else {
                v.sort();
            }
            v
        };

        let mut row_set: Vec<String> = rows.iter().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
        row_set = sort_strs(row_set);
        let mut col_set: Vec<String> = cols.iter().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
        col_set = sort_strs(col_set);

        // contagens
        let mut cell: HashMap<(String, String), usize> = HashMap::new();
        for (r, c) in rows.iter().zip(cols.iter()) {
            *cell.entry((r.clone(), c.clone())).or_insert(0) += 1;
        }

        let n = rows.len();
        let col_totals: Vec<usize> = col_set.iter()
            .map(|c| row_set.iter().map(|r| *cell.get(&(r.clone(), c.clone())).unwrap_or(&0)).sum())
            .collect();
        let row_totals: Vec<usize> = row_set.iter()
            .map(|r| col_set.iter().map(|c| *cell.get(&(r.clone(), c.clone())).unwrap_or(&0)).sum())
            .collect();

        // larguras de coluna
        let cell_w   = 10usize;
        let row_lw   = row_var.len().max(12).max(row_set.iter().map(|s| s.len()).max().unwrap_or(0)) + 2;
        let col_head_w = col_set.len() * (cell_w + 1) + 1;
        let total_w  = cell_w + 2;

        // linha de cabeçalho do col_var
        println!("\n{:>rw$} | {:^chw$}| {:>tw$}",
                 "", col_var, "Total",
                 rw = row_lw, chw = col_head_w, tw = total_w);

        // linha com os valores das colunas
        print!("{:>rw$} |", row_var, rw = row_lw);
        for cv in &col_set { print!(" {:>cw$}", cv, cw = cell_w); }
        println!(" | {:>cw$}", "Total", cw = cell_w);

        let sep = format!("{}-+{}-+{}", "-".repeat(row_lw), "-".repeat(col_head_w), "-".repeat(total_w));
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
        for ct in &col_totals { print!(" {:>cw$}", ct, cw = cell_w); }
        println!(" | {:>cw$}", n, cw = cell_w);
        println!();

        if do_chi2 {
            let mut stat = 0.0_f64;
            for (i, rv) in row_set.iter().enumerate() {
                for (j, cv) in col_set.iter().enumerate() {
                    let obs = *cell.get(&(rv.clone(), cv.clone())).unwrap_or(&0) as f64;
                    let exp = row_totals[i] as f64 * col_totals[j] as f64 / n as f64;
                    if exp > 0.0 { stat += (obs - exp).powi(2) / exp; }
                }
            }
            let df = (row_set.len() - 1) * (col_set.len() - 1);
            let p   = chi2_pvalue(stat, df);
            println!("  Pearson chi2({df}) = {stat:.4}   Pr = {p:.4}");
            println!();
        }

        Ok(())
    }

    fn extract_binary_args(&mut self, args: &[Expr]) -> Result<(Formula, DataFrame)> {
        if args.len() < 2 {
            return Err(HayashiError::Runtime("estimator requires (formula, dataframe)".into()));
        }
        let formula_ast = match &args[0] {
            Expr::Formula(f) => f.clone(),
            _ => return Err(HayashiError::Type("first argument must be a formula".into())),
        };
        let df_name = match &args[1] {
            Expr::Var(name) => name.clone(),
            _ => return Err(HayashiError::Type("second argument must be a DataFrame variable".into())),
        };
        let df = match self.env.get(&df_name) {
            Some(Value::DataFrame(df)) => df.clone(),
            _ => return Err(HayashiError::Runtime(format!("'{df_name}' is not a DataFrame"))),
        };
        Ok((formula_ast, df))
    }

    // ── Métodos de objetos ────────────────────────────────────────────────────

    fn eval_field(&mut self, obj: &Expr, field: &str, _args: &[Expr], _opts: &[Opt]) -> Result<Value> {
        let val = self.eval_expr(obj)?;
        match (&val, field) {
            (Value::OlsResult(m), "summary")    => { println!("{}", m.result); Ok(Value::Nil) }
            (Value::IvResult(r), "summary")     => { println!("{r}"); Ok(Value::Nil) }
            (Value::BinaryResult(m), "summary") => { println!("{m}"); Ok(Value::Nil) }
            (Value::PanelResult(r), "summary")  => { println!("{r}"); Ok(Value::Nil) }
            (Value::ReResult(r), "summary")     => { println!("{r}"); Ok(Value::Nil) }
            (_, f) => Err(HayashiError::Runtime(format!("unknown method '{f}'"))),
        }
    }

    // ── Avalia expressão elemento-a-elemento sobre colunas de um DataFrame ───

    fn eval_col_expr(expr: &Expr, df: &DataFrame) -> Result<Vec<f64>> {
        match expr {
            Expr::Float(v) => {
                let n = df.n_rows();
                Ok(vec![*v; n])
            }
            Expr::Int(v) => {
                let n = df.n_rows();
                Ok(vec![*v as f64; n])
            }
            Expr::Bool(v) => {
                let n = df.n_rows();
                Ok(vec![if *v { 1.0 } else { 0.0 }; n])
            }
            Expr::Var(name) => {
                use greeners::Column;
                let col = df.get_column(name)
                    .map_err(|_| HayashiError::Runtime(format!("column '{name}' not found")))?;
                match col {
                    Column::Float(arr) => Ok(arr.to_vec()),
                    Column::Int(arr)   => Ok(arr.iter().map(|&x| x as f64).collect()),
                    _ => Err(HayashiError::Type(format!("column '{name}' is not numeric"))),
                }
            }
            Expr::Neg(inner) => {
                let vals = Self::eval_col_expr(inner, df)?;
                Ok(vals.into_iter().map(|x| -x).collect())
            }
            Expr::BinOp { op, lhs, rhs } => {
                let l = Self::eval_col_expr(lhs, df)?;
                let r = Self::eval_col_expr(rhs, df)?;
                if l.len() != r.len() {
                    return Err(HayashiError::Runtime("mismatched column lengths".into()));
                }
                Ok(l.into_iter().zip(r).map(|(a, b)| match op {
                    BinOp::Add  => a + b,
                    BinOp::Sub  => a - b,
                    BinOp::Mul  => a * b,
                    BinOp::Div  => a / b,
                    BinOp::Pow  => a.powf(b),
                    BinOp::Gt   => if a > b { 1.0 } else { 0.0 },
                    BinOp::Lt   => if a < b { 1.0 } else { 0.0 },
                    BinOp::GtEq => if a >= b { 1.0 } else { 0.0 },
                    BinOp::LtEq => if a <= b { 1.0 } else { 0.0 },
                    BinOp::Eq   => if (a - b).abs() < f64::EPSILON { 1.0 } else { 0.0 },
                    BinOp::Ne   => if (a - b).abs() >= f64::EPSILON { 1.0 } else { 0.0 },
                }).collect())
            }
            Expr::Call { func, args, .. } => {
                // funções escalares aplicadas elemento-a-elemento
                if args.len() == 1 {
                    let vals = Self::eval_col_expr(&args[0], df)?;
                    let f: fn(f64) -> f64 = match func.as_str() {
                        "log"  | "ln"   => f64::ln,
                        "log2"          => f64::log2,
                        "log10"         => f64::log10,
                        "exp"           => f64::exp,
                        "sqrt"          => f64::sqrt,
                        "abs"           => f64::abs,
                        "floor"         => f64::floor,
                        "ceil"          => f64::ceil,
                        "round"         => f64::round,
                        "sin"           => f64::sin,
                        "cos"           => f64::cos,
                        other => return Err(HayashiError::Runtime(
                            format!("unknown column function '{other}'")
                        )),
                    };
                    Ok(vals.into_iter().map(f).collect())
                } else if args.len() == 2 && func == "pow" {
                    let base = Self::eval_col_expr(&args[0], df)?;
                    let exp  = Self::eval_col_expr(&args[1], df)?;
                    Ok(base.into_iter().zip(exp).map(|(a, b)| a.powf(b)).collect())
                } else {
                    Err(HayashiError::Runtime(format!(
                        "function '{func}' not supported in generate expression"
                    )))
                }
            }
            _ => Err(HayashiError::Runtime(
                "expression type not supported in generate".into()
            )),
        }
    }

    // ── Executa statement ─────────────────────────────────────────────────────

    pub fn exec(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::Let { name, value } => {
                let val = self.eval_expr(value)?;
                self.env.set(name, val);
            }

            Stmt::Load { path, alias } => {
                let path_str = match self.eval_expr(path)? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("load requires a string path".into())),
                };

                // Para URLs: baixa para arquivo temporário e usa o caminho local
                let _tmp; // mantém o TempPath vivo durante a leitura
                let local_path: &str = if crate::io::fetch::is_url(&path_str) {
                    println!("Downloading '{}'…", path_str);
                    _tmp = crate::io::fetch::download_to_temp(&path_str)?;
                    _tmp.to_str().ok_or_else(|| HayashiError::Runtime("temp path is not UTF-8".into()))?
                } else {
                    &path_str
                };

                let (df, n_rows) = if local_path.ends_with(".dta") {
                    crate::io::dta::load_dta(local_path)?
                } else {
                    let df = DataFrame::from_csv(local_path)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                    let n = df.n_rows();
                    (df, n)
                };
                println!("Loaded '{}' → {alias} ({} rows)", path_str, n_rows);
                self.env.set(alias, Value::DataFrame(df));
            }

            Stmt::Predict { df, varname, model, kind } => {
                let mut df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df}' is not a DataFrame"))),
                };
                let model_val = self.eval_expr(model)?;

                let vals: Vec<f64> = match (&model_val, kind.as_str()) {
                    // OLS — fitted values
                    (Value::OlsResult(m), "xb") => {
                        m.x.dot(&m.result.params).to_vec()
                    }
                    // OLS — residuals
                    (Value::OlsResult(m), "residuals" | "resid" | "e") => {
                        m.residuals.to_vec()
                    }
                    // Logit/Probit — predicted probability
                    (Value::BinaryResult(m), "pr" | "xb") => {
                        m.result.predict_proba(&m.x).to_vec()
                    }
                    // Erros descritivos
                    (Value::OlsResult(_), k) => return Err(HayashiError::Runtime(
                        format!("unknown predict kind '{k}' for OLS — use: xb, residuals")
                    )),
                    (Value::BinaryResult(_), k) => return Err(HayashiError::Runtime(
                        format!("unknown predict kind '{k}' for logit/probit — use: pr")
                    )),
                    _ => return Err(HayashiError::Type(
                        "predict requires an OLS or binary model".into()
                    )),
                };

                let arr = ndarray::Array1::from(vals);
                df_val.insert(varname.clone(), arr)
                    .map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                println!("({} obs)  {df}.{varname} ({kind}) predicted", df_val.n_rows());
                self.env.set(df, Value::DataFrame(df_val));
            }

            Stmt::Count { df, cond } => {
                let df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df}' is not a DataFrame"))),
                };
                let n = if let Some(cond_expr) = cond {
                    let mask = Self::eval_col_expr(cond_expr, &df_val)?;
                    mask.iter().filter(|&&v| v != 0.0).count()
                } else {
                    df_val.n_rows()
                };
                println!("{n}");
            }

            Stmt::Replace { df, varname, expr, cond } => {
                let mut df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df}' is not a DataFrame"))),
                };
                let new_vals = Self::eval_col_expr(expr, &df_val)?;

                let final_vals: Vec<f64> = if let Some(cond_expr) = cond {
                    let mask = Self::eval_col_expr(cond_expr, &df_val)?;
                    // lê coluna original para preservar onde mask == 0
                    use greeners::Column;
                    let old_vals: Vec<f64> = match df_val.get_column(varname) {
                        Ok(Column::Float(arr)) => arr.to_vec(),
                        Ok(Column::Int(arr))   => arr.iter().map(|&v| v as f64).collect(),
                        _ => vec![f64::NAN; new_vals.len()],
                    };
                    let n_replaced = mask.iter().filter(|&&m| m != 0.0).count();
                    println!("({n_replaced} real changes made)");
                    mask.into_iter().zip(old_vals).zip(new_vals)
                        .map(|((m, old), new)| if m != 0.0 { new } else { old })
                        .collect()
                } else {
                    let n = new_vals.len();
                    println!("({n} real changes made)");
                    new_vals
                };

                let arr = ndarray::Array1::from(final_vals);
                df_val.insert(varname.clone(), arr)
                    .map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                self.env.set(df, Value::DataFrame(df_val));
            }

            Stmt::Generate { df, varname, expr } => {
                let mut df_val = match self.env.get(df) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => return Err(HayashiError::Runtime(format!("'{df}' is not a DataFrame"))),
                };
                let vals = Self::eval_col_expr(expr, &df_val)?;
                let arr = ndarray::Array1::from(vals);
                df_val.insert(varname.clone(), arr)
                    .map_err(|e: greeners::GreenersError| HayashiError::Runtime(e.to_string()))?;
                println!("({} obs)  {df}.{varname} generated", df_val.n_rows());
                self.env.set(df, Value::DataFrame(df_val));
            }

            Stmt::Print(expr) => {
                let val = self.eval_expr(expr)?;
                println!("{val}");
            }

            Stmt::Export { value, fmt, path } => {
                let val = self.eval_expr(value)?;
                let path_str = match self.eval_expr(path)? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("export path must be a string".into())),
                };

                use greeners::ExportableResult;

                match (val, fmt.as_str()) {
                    // ── DataFrame → CSV ───────────────────────────────────────
                    (Value::DataFrame(df), "csv" | "delimited") => {
                        df.to_csv(&path_str)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!("Exported DataFrame → '{path_str}' ({} rows)", df.n_rows());
                    }

                    // ── OLS → CSV / LaTeX / HTML ──────────────────────────────
                    (Value::OlsResult(m), "csv") => {
                        let content = m.result.to_csv();
                        std::fs::write(&path_str, &content)
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported OLS coefficients → '{path_str}'");
                    }
                    (Value::OlsResult(m), "latex" | "tex") => {
                        let content = m.result.to_latex();
                        std::fs::write(&path_str, &content)
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported OLS LaTeX table → '{path_str}'");
                    }
                    (Value::OlsResult(m), "html") => {
                        let content = m.result.to_html();
                        std::fs::write(&path_str, &content)
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported OLS HTML table → '{path_str}'");
                    }

                    // ── Outros modelos: texto plain ───────────────────────────
                    (Value::IvResult(r), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{r}"))
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported IV results → '{path_str}'");
                    }
                    (Value::BinaryResult(m), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{m}"))
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported logit/probit results → '{path_str}'");
                    }
                    (Value::PanelResult(r), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{r}"))
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported FE results → '{path_str}'");
                    }
                    (Value::ReResult(r), "txt" | "text") => {
                        std::fs::write(&path_str, format!("{r}"))
                            .map_err(|e| HayashiError::Io(e))?;
                        println!("Exported RE results → '{path_str}'");
                    }

                    (_, fmt) => return Err(HayashiError::Runtime(format!(
                        "unsupported export format '{fmt}' for this value type\n\
                         Available: DataFrame→csv  |  OLS→csv,latex,html  |  other models→txt"
                    ))),
                }
            }

            Stmt::Expr(expr) => {
                self.eval_expr(expr)?;
            }
        }
        Ok(())
    }
}
