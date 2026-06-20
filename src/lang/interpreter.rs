use std::collections::HashMap;
use std::rc::Rc;
use ndarray::{Array1, Array2};
use greeners::{DataFrame, Formula as GFormula, OLS, CovarianceType, IV, Logit, Probit, FixedEffects, RandomEffects};
use greeners::diagnostics::Diagnostics;
use greeners::specification_tests::SpecificationTests;
use crate::lang::ast::*;
use crate::lang::error::{HayashiError, Result};

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
