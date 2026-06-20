/// Fórmula econométrica: y ~ x1 + x2 + C(region) | id + t
#[derive(Debug, Clone)]
pub struct Formula {
    pub lhs: String,
    pub rhs: Vec<RhsTerm>,
    pub fe: Vec<String>,   // após |
}

#[derive(Debug, Clone)]
pub enum RhsTerm {
    Var(String),
    Categorical(String),       // C(var)
    Transform(String, String), // log(var), sqrt(var), I(expr)
    Interaction(String, String),
}

/// Opções nomeadas passadas para estimadores: cov=HC3, lags=4, ...
#[derive(Debug, Clone)]
pub struct Opt {
    pub name: String,
    pub value: Expr,
}

/// Operadores binários para expressões aritméticas/comparação
#[derive(Debug, Clone)]
pub enum BinOp {
    Add, Sub, Mul, Div, Pow,
    Gt, Lt, GtEq, LtEq, Eq, Ne,
}

/// Operador de série temporal
#[derive(Debug, Clone)]
pub enum TsOpKind { Lag, Lead, Diff }

/// Expressões da linguagem
#[derive(Debug, Clone)]
pub enum Expr {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    Var(String),
    Formula(Formula),

    // aritmética / comparação: price * 1.5, mpg > 20
    BinOp { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },

    // negação unária: -price
    Neg(Box<Expr>),

    // chamada de função/estimador: ols(fórmula, df, cov=HC3)
    Call {
        func: String,
        args: Vec<Expr>,
        opts: Vec<Opt>,
    },

    // acesso a campo: model.summary()
    Field {
        obj: Box<Expr>,
        field: String,
        args: Vec<Expr>,
        opts: Vec<Opt>,
    },

    // operadores de série temporal: L.price, L2.price, F.gdp, D.wage
    TsOp { op: TsOpKind, var: String, n: usize },
}

/// Comandos (statements) da linguagem
#[derive(Debug, Clone)]
pub enum Stmt {
    // let nome = expr
    Let { name: String, value: Expr },

    // load "arquivo.csv" as nome
    Load { path: Expr, alias: String },

    // generate df newvar = expr
    Generate { df: String, varname: String, expr: Expr },

    // predict df newvar = model [, kind]
    // kind: xb (default) | residuals | pr
    Predict { df: String, varname: String, model: Expr, kind: String },

    // print(expr)
    Print(Expr),

    // export(expr, formato, "arquivo")
    Export { value: Expr, fmt: String, path: Expr },

    // replace df varname = expr [if cond]
    Replace { df: String, varname: String, expr: Expr, cond: Option<Expr> },

    // count df [if cond]
    Count { df: String, cond: Option<Expr> },

    // tsset df timevar
    Tsset { df: String, t_var: String },

    // expr standalone (ex: test(model, white))
    Expr(Expr),
}
