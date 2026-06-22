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

/// Operadores binários para expressões aritméticas/comparação/lógica
#[derive(Debug, Clone)]
pub enum BinOp {
    Add, Sub, Mul, Div, Pow,
    Gt, Lt, GtEq, LtEq, Eq, Ne,
    And, Or, In,
}

/// Operador de série temporal
#[derive(Debug, Clone)]
pub enum TsOpKind { Lag, Lead, Diff }

/// Iterador de loop for
#[derive(Debug, Clone)]
pub enum ForIter {
    Range(Expr, Expr),  // start..end  (exclusivo no topo, tipo Rust/Python)
    Items(Expr),         // lista ou variável
}

/// Expressões da linguagem
#[derive(Debug, Clone)]
pub enum Expr {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    FString(String),
    Var(String),
    Formula(Formula),

    // aritmética / comparação / lógica: price * 1.5, mpg > 20, a && b
    BinOp { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },

    // negação unária: -price
    Neg(Box<Expr>),

    // negação booleana: !flag
    Not(Box<Expr>),

    // lista literal: [1, 2, 3]  ou  [ols(...), fe(...)]
    List(Vec<Expr>),

    // dict literal: {"key": value, "k2": v2}
    Dict(Vec<(Expr, Expr)>),

    // indexação: lista[0] ou dict["key"]
    Index { obj: Box<Expr>, idx: Box<Expr> },

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

    // closure: |x, y| x + y
    Closure { params: Vec<String>, body: Box<Expr> },

    // match expr { pattern => result, ... }
    Match { expr: Box<Expr>, arms: Vec<(Expr, Expr)> },

    // if cond { expr } else { expr }  (expression, returns value)
    IfExpr { cond: Box<Expr>, then_expr: Box<Expr>, else_expr: Box<Expr> },

    // operadores de série temporal: L.price, L2.price, F.gdp, D.wage
    TsOp { op: TsOpKind, var: String, n: usize },
}

pub type Spanned = (Stmt, usize);

/// Comandos (statements) da linguagem
#[derive(Debug, Clone)]
pub enum Stmt {
    // let nome = expr (declara no escopo atual)
    Let { name: String, value: Expr },

    // const nome = expr (declara imutável no escopo atual)
    Const { name: String, value: Expr },

    // nome = expr (modifica variável existente no escopo mais próximo)
    Assign { name: String, value: Expr },

    // load "arquivo.csv" as nome [, sheet=Plan1, table=t, query="..."]
    Load { path: Expr, alias: String, opts: Vec<Opt> },

    // generate df newvar = expr
    Generate { df: String, varname: String, expr: Expr },

    // predict df newvar = model [, "kind"]
    // kind: "xb" (default) | "residuals" | "pr"
    Predict { df: String, varname: String, model: Expr, kind: Expr },

    // print(expr)
    Print(Expr),

    // export(expr, formato, "arquivo")
    Export { value: Expr, fmt: Expr, path: Expr },

    // replace df varname = expr [if cond]
    Replace { df: String, varname: String, expr: Expr, cond: Option<Expr> },

    // count df [if cond]
    Count { df: String, cond: Option<Expr> },

    // tsset df timevar
    Tsset { df: String, t_var: String },

    // if cond { ... } [else if cond { ... }]* [else { ... }]
    If {
        cond: Expr,
        then_body: Vec<Spanned>,
        else_body: Option<Vec<Spanned>>,
    },

    // for var in iter { ... }
    For {
        var: String,
        iter: ForIter,
        body: Vec<Spanned>,
    },

    // while cond { ... }
    While {
        cond: Expr,
        body: Vec<Spanned>,
    },

    // fn nome(p1, p2) { corpo }
    Fn {
        name: String,
        params: Vec<String>,
        body: Vec<Spanned>,
    },

    // return [expr]
    Return(Option<Expr>),

    // break / continue
    Break,
    Continue,

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

    // display expr   (sem parênteses, como Stata)
    Display(Expr),

    // expr standalone (ex: test(model, white))
    Expr(Expr),
}
