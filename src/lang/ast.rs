/// Econometric formula: y ~ x1 + x2 + C(region) | id + t
#[derive(Debug, Clone)]
pub struct Formula {
    pub lhs: String,
    pub rhs: Vec<RhsTerm>,
    pub fe: Vec<String>, // after |
}

/// Um termo no RHS de uma fórmula econométrica.
///
/// `Expr` carrega qualquer expressão do AST (variável, transformação, operação
/// aritmética, etc.).  A materialização em coluna numérica é feita pelo
/// interpreter via `eval_col_expr` antes de montar a design matrix — o Greeners
/// nunca vê strings como `"log(income):age"`.
///
/// `Categorical` preserva semântica de dummy-encoding: o Greeners ainda recebe
/// `C(colname)` para colunas que já existem no DataFrame, mas para expressões
/// compostas o interpreter as materializa primeiro.
///
/// `Interaction(lhs, rhs)` representa `lhs:rhs` (produto element-wise).
/// Cada lado é recursivamente um `RhsTerm`, o que permite `log(K):log(L)` ou
/// `I(x^2):C(region)`.
#[derive(Debug, Clone)]
pub enum RhsTerm {
    /// Qualquer expressão: variável simples, transformação, aritmética, etc.
    Expr(Box<Expr>),
    /// C(expr) — dummy-encoding; expr é normalmente Var mas pode ser composta.
    Categorical(Box<Expr>),
    /// lhs:rhs — interação (produto element-wise dos dois termos).
    Interaction(Box<RhsTerm>, Box<RhsTerm>),
}

impl RhsTerm {
    /// Constrói um termo simples a partir de um nome de variável.
    pub fn var(name: impl Into<String>) -> Self {
        RhsTerm::Expr(Box::new(Expr::Var(name.into())))
    }

    /// Retorna o nome da variável se o termo for exatamente `Expr(Var(name))`.
    pub fn as_var(&self) -> Option<&str> {
        if let RhsTerm::Expr(e) = self {
            if let Expr::Var(v) = e.as_ref() {
                return Some(v.as_str());
            }
        }
        None
    }

    /// Representação legível para nomes de coeficientes, diagnósticos, etc.
    pub fn display_name(&self) -> String {
        match self {
            RhsTerm::Expr(e) => expr_display(e),
            RhsTerm::Categorical(e) => format!("C({})", expr_display(e)),
            RhsTerm::Interaction(a, b) => format!("{}:{}", a.display_name(), b.display_name()),
        }
    }
}

/// Serializa uma Expr de forma compacta para nomes de colunas e fórmulas.
pub(crate) fn expr_display(e: &Expr) -> String {
    match e {
        Expr::Var(v) => v.clone(),
        Expr::Int(n) => n.to_string(),
        Expr::Float(f) => format!("{f}"),
        Expr::Bool(b) => b.to_string(),
        Expr::Str(s) => format!("\"{s}\""),
        Expr::Neg(inner) => format!("-{}", expr_display(inner)),
        Expr::Not(inner) => format!("!{}", expr_display(inner)),
        Expr::BinOp { op, lhs, rhs } => {
            let op_str = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Pow => "^",
                BinOp::Mod => "%",
                BinOp::Gt => ">",
                BinOp::Lt => "<",
                BinOp::GtEq => ">=",
                BinOp::LtEq => "<=",
                BinOp::Eq => "==",
                BinOp::Ne => "!=",
                BinOp::And => "&&",
                BinOp::Or => "||",
                BinOp::In => "in",
            };
            format!("({}{}{})", expr_display(lhs), op_str, expr_display(rhs))
        }
        Expr::Call { func, args, .. } => {
            let args_s: Vec<String> = args.iter().map(expr_display).collect();
            format!("{}({})", func, args_s.join(","))
        }
        // f-string: reconstruct as f"lit{expr}lit" for readable coefficient names
        Expr::FString(parts) => {
            let mut s = String::from("f\"");
            for part in parts {
                match part {
                    FStringPart::Lit(lit) => s.push_str(lit),
                    FStringPart::Interp { expr, fmt } => {
                        s.push('{');
                        s.push_str(&expr_display(expr));
                        if let Some(f) = fmt {
                            s.push(':');
                            s.push_str(f);
                        }
                        s.push('}');
                    }
                }
            }
            s.push('"');
            s
        }
        // time-series operators: L.price, L2.price, F.gdp, D.wage
        Expr::TsOp { op, var, n } => {
            let prefix = match op {
                TsOpKind::Lag => "L",
                TsOpKind::Lead => "F",
                TsOpKind::Diff => "D",
            };
            if *n == 1 {
                format!("{prefix}.{var}")
            } else {
                format!("{prefix}{n}.{var}")
            }
        }
        // if expression: if(cond,then,else)
        Expr::If {
            cond,
            then_expr,
            else_expr,
        } => {
            format!(
                "if({},{},{})",
                expr_display(cond),
                expr_display(then_expr),
                expr_display(else_expr)
            )
        }
        // indexing: obj[idx]
        Expr::Index { obj, idx } => format!("{}[{}]", expr_display(obj), expr_display(idx)),
        // list literal: [a,b,c]
        Expr::List(items) => {
            let s: Vec<String> = items.iter().map(expr_display).collect();
            format!("[{}]", s.join(","))
        }
        // field access: obj.field or obj.method(args)
        Expr::Field {
            obj, field, args, ..
        } => {
            if args.is_empty() {
                format!("{}.{}", expr_display(obj), field)
            } else {
                let args_s: Vec<String> = args.iter().map(expr_display).collect();
                format!("{}.{}({})", expr_display(obj), field, args_s.join(","))
            }
        }
        // remaining variants are not meaningful as column/coefficient names
        _ => "_".to_string(),
    }
}

/// Named options passed to estimators: cov=HC3, lags=4, ...
#[derive(Debug, Clone)]
pub struct Opt {
    pub name: String,
    pub value: Expr,
}

/// Binary operators for arithmetic/comparison/logical expressions
#[derive(Debug, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Gt,
    Lt,
    GtEq,
    LtEq,
    Eq,
    Ne,
    And,
    Or,
    In,
}

/// Time-series operator
#[derive(Debug, Clone)]
pub enum TsOpKind {
    Lag,
    Lead,
    Diff,
}

/// for loop iterator
#[derive(Debug, Clone)]
pub enum ForIter {
    Range(Expr, Expr),          // start..end   (exclusive, like Rust)
    RangeInclusive(Expr, Expr), // start..=end  (inclusive, like Rust)
    Items(Expr),                // list or variable
}

/// One segment of an interpolated string literal.
///
/// `f"hello {name:.2f}!"` is parsed at parse-time into:
/// `[Lit("hello "), Interp { expr: Var("name"), fmt: Some(".2f") }, Lit("!")]`
/// so the interpreter never needs to re-lex/re-parse at runtime.
#[derive(Debug, Clone)]
pub enum FStringPart {
    /// Literal text segment (already unescaped: `{{` → `{`, `}}` → `}`).
    Lit(String),
    /// Interpolated expression with an optional format specifier (e.g. `.2f`).
    Interp {
        expr: Box<Expr>,
        fmt: Option<String>,
    },
}

/// Language expressions
#[derive(Debug, Clone)]
pub enum Expr {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    FString(Vec<FStringPart>),
    Var(String),
    Formula(Formula),
    Nil,

    // arithmetic / comparison / logic: price * 1.5, mpg > 20, a && b
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    // unary negation: -price
    Neg(Box<Expr>),

    // boolean negation: !flag
    Not(Box<Expr>),

    // list literal: [1, 2, 3]  or  [ols(...), fe(...)]
    List(Vec<Expr>),

    // dict literal: {"key": value, "k2": v2}
    Dict(Vec<(Expr, Expr)>),

    // indexing: list[0] or dict["key"]
    Index {
        obj: Box<Expr>,
        idx: Box<Expr>,
    },

    // function/estimator call: ols(formula, df, cov=HC3)
    Call {
        func: String,
        args: Vec<Expr>,
        opts: Vec<Opt>,
    },

    // field access: model.summary()
    Field {
        obj: Box<Expr>,
        field: String,
        args: Vec<Expr>,
        opts: Vec<Opt>,
    },

    // closure: |x, y| x + y
    Closure {
        params: Vec<String>,
        body: Box<Expr>,
    },

    // closure application: expr |> |x| body
    Apply {
        func: Box<Expr>,
        args: Vec<Expr>,
    },

    // pipe chain: source |> f(x) |> g(y)
    // source = original LHS, expr = desugared call chain
    Pipe {
        source: Box<Expr>,
        expr: Box<Expr>,
    },

    // match expr { pattern => result, ... }
    Match {
        expr: Box<Expr>,
        arms: Vec<(Expr, Expr)>,
    },

    // if cond { expr } else { expr }  (expression, returns value)
    If {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },

    // time-series operators: L.price, L2.price, F.gdp, D.wage
    TsOp {
        op: TsOpKind,
        var: String,
        n: usize,
    },

    // obsolete functional form: quietly(expr)
    Quietly(Box<Expr>),

    // ranges as expressions: 1..5 → [1,2,3,4]  |  1..=5 → [1,2,3,4,5]
    Range(Box<Expr>, Box<Expr>),
    RangeInclusive(Box<Expr>, Box<Expr>),

    // expression block: { stmt; ...; expr }
    Block(Vec<Stmt>, Option<Box<Expr>>),
}

pub type Spanned = (Stmt, usize);

/// Language commands (statements)
#[derive(Debug, Clone)]
pub enum Stmt {
    // let name = expr (declares in current scope)
    Let {
        name: String,
        value: Expr,
    },

    // const name = expr (declares immutable in current scope)
    Const {
        name: String,
        value: Expr,
    },

    // name = expr (modifies existing variable in nearest scope)
    Assign {
        name: String,
        value: Expr,
    },

    // load "file.csv" as name [, sheet=Sheet1, table=t, query="..."]
    Load {
        path: Expr,
        alias: String,
        opts: Vec<Opt>,
    },

    // generate df newvar = expr
    // newvar may be an identifier or a dynamic string expression (e.g. f"ret_{t}")
    Generate {
        df: String,
        varname: Expr,
        expr: Expr,
    },

    // predict df newvar = model [, "kind"]
    // kind: "xb" (default) | "residuals" | "pr"
    Predict {
        df: String,
        varname: String,
        model: Expr,
        kind: Expr,
    },

    // print(expr, expr, ..., sep=" ", end="\n")
    Print(Vec<Expr>, Vec<Opt>),

    // export(expr, format, "file")
    Export {
        value: Expr,
        fmt: Expr,
        path: Expr,
    },

    // replace df varname = expr [if cond]
    Replace {
        df: String,
        varname: String,
        expr: Expr,
        cond: Option<Expr>,
    },

    // count df [if cond]
    Count {
        df: String,
        cond: Option<Expr>,
    },

    // tsset df timevar
    Tsset {
        df: String,
        t_var: String,
    },

    // if cond { ... } [else if cond { ... }]* [else { ... }]
    If {
        cond: Expr,
        then_body: Vec<Spanned>,
        else_body: Option<Vec<Spanned>>,
    },

    // for var in iter { ... }
    // for k, v in dict { ... } | for i, v in list { ... }
    For {
        var: String,
        var2: Option<String>,
        iter: ForIter,
        body: Vec<Spanned>,
    },

    // parallel for var in iter { ... } [, threads=N]
    // Like `for`, but iterations run concurrently across threads.
    // Each thread gets its own Interpreter; captured variables are Arc-cloned.
    // Results are collected in order into a List.
    // Optional `threads=N` limits the number of worker threads.
    ParallelFor {
        var: String,
        var2: Option<String>,
        iter: ForIter,
        body: Vec<Spanned>,
        threads: Option<Expr>,
    },

    // while cond { ... }
    While {
        cond: Expr,
        body: Vec<Spanned>,
    },

    // fn name(p1, p2, p3=default) { ## docstring ...; body }
    Fn {
        name: String,
        params: Vec<String>,
        defaults: Vec<Option<Expr>>,
        doc: Option<String>,
        body: Vec<Spanned>,
    },

    // return [expr]
    Return(Option<Expr>),

    // break / continue
    Break,
    Continue,

    // quietly on / quietly off
    QuietlyOn,
    QuietlyOff,

    // try { ... } catch e { ... } [finally { ... }]
    TryCatch {
        try_body: Vec<Spanned>,
        error_var: String,
        catch_body: Vec<Spanned>,
        finally_body: Vec<Spanned>,
    },

    // input df
    // Y X u
    // 4 1  3
    // ...
    // end
    Input {
        alias: String,
        headers: Vec<String>,
        rows: Vec<Vec<f64>>,
    },

    // display expr   (without parentheses, like Stata)
    Display(Expr),

    // expr standalone (e.g. test(model, white))
    Expr(Expr),

    // Standalone block { stmt* }
    Block(Vec<Spanned>),
}
