/// Econometric formula: y ~ x1 + x2 + C(region) | id + t
#[derive(Debug, Clone)]
pub struct Formula {
    pub lhs: String,
    pub rhs: Vec<RhsTerm>,
    pub fe: Vec<String>, // after |
}

#[derive(Debug, Clone)]
pub enum RhsTerm {
    Var(String),
    Categorical(String),       // C(var)
    Transform(String, String), // log(var), sqrt(var), I(expr)
    Interaction(String, String),
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

/// Language expressions
#[derive(Debug, Clone)]
pub enum Expr {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    FString(String),
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
    Generate {
        df: String,
        varname: String,
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

    // try { ... } catch e { ... }
    TryCatch {
        try_body: Vec<Spanned>,
        error_var: String,
        catch_body: Vec<Spanned>,
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
