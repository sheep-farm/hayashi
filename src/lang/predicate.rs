//! Predicado `where=` para `load`.
//!
//! Reusa o lexer/parser do hayashi para parsear a string do `where`, depois
//! converte a `Expr` em um `RowPredicate` independente do interpreter. Isso
//! permite avaliar o predicado dentro dos loaders (CSV, DTA, Excel, Parquet)
//! sem precisar de `Interpreter`/`Environment`.
//!
//! Sintaxe suportada (operadores binários entre coluna e literal, com
//! `&&`, `||`, `!` e parênteses via gramática existente):
//!
//! ```text
//! col == literal | col != literal | col >  literal
//! col <  literal | col >= literal | col <= literal
//! col in [lit1, lit2, ...]
//! !pred | pred && pred | pred || pred
//! ```
//!
//! `literal` é `Int`, `Float`, `Str` ou `Bool`.

use crate::lang::ast::{BinOp, Expr};
use crate::lang::error::{HayashiError, Result};
use crate::lang::lexer::Lexer;
use crate::lang::parser::Parser;

#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

impl Literal {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Literal::Int(v) => Some(*v as f64),
            Literal::Float(v) => Some(*v),
            Literal::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            Literal::Str(_) => None,
        }
    }

    /// Literal embutido em SQL, corretamente escapado.
    fn to_sql_literal(&self) -> String {
        match self {
            Literal::Int(v) => format!("{v}"),
            Literal::Float(v) => format!("{v}"),
            Literal::Bool(v) => format!("{}", if *v { 1 } else { 0 }),
            Literal::Str(s) => format!("'{}'", s.replace('\'', "''")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RowPredicate {
    All,
    Eq(String, Literal),
    Ne(String, Literal),
    Gt(String, Literal),
    Lt(String, Literal),
    Ge(String, Literal),
    Le(String, Literal),
    In(String, Vec<Literal>),
    Not(Box<RowPredicate>),
    And(Vec<RowPredicate>),
    Or(Vec<RowPredicate>),
}

/// Acesso por nome de coluna durante a iteração de uma linha.
/// Cada loader implementa este trait para a sua estrutura de linha.
pub trait RowAccess {
    /// Valor numérico da coluna. `None` se a coluna não existir ou não for
    /// numérica; `Some(NaN)` se for numérica mas nula.
    fn get_f64(&self, col: &str) -> Option<f64>;
    /// Valor textual da coluna. `None` se a coluna não existir; `Some("")`
    /// se for string nula/vazia.
    fn get_str(&self, col: &str) -> Option<&str>;
}

/// Bounds (min/max) de uma coluna dentro de um row group do Parquet.
/// Usado pelo row group pruning — permite decidir se um row group pode
/// conter linhas que satisfazem o predicado, sem decodificar os dados.
pub trait GroupBounds {
    /// `(min, max)` numéricos da coluna neste row group, ou `None` se a
    /// coluna não existir, não tiver estatísticas, ou não for numérica.
    fn f64_bounds(&self, col: &str) -> Option<(f64, f64)>;
    /// `(min, max)` textuais da coluna neste row group, ou `None` se a
    /// coluna não existir, não tiver estatísticas, ou não for string.
    fn str_bounds(&self, col: &str) -> Option<(&str, &str)>;
}

impl RowPredicate {
    /// Parseia uma string `where="..."` usando o lexer/parser do hayashi.
    pub fn parse(s: &str) -> Result<RowPredicate> {
        let mut lexer = Lexer::new(s);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expr()?;
        Self::from_expr(&expr)
    }

    fn from_expr(e: &Expr) -> Result<RowPredicate> {
        match e {
            Expr::BinOp { op, lhs, rhs } => match op {
                BinOp::And => Ok(RowPredicate::And(vec![
                    Self::from_expr(lhs)?,
                    Self::from_expr(rhs)?,
                ])),
                BinOp::Or => Ok(RowPredicate::Or(vec![
                    Self::from_expr(lhs)?,
                    Self::from_expr(rhs)?,
                ])),
                BinOp::In => {
                    let col = extract_col(lhs).ok_or_else(|| {
                        HayashiError::Runtime(
                            "where: left-hand side of `in` must be a column name".into(),
                        )
                    })?;
                    let lits = extract_list_literals(rhs)?;
                    Ok(RowPredicate::In(col, lits))
                }
                BinOp::Eq | BinOp::Ne | BinOp::Gt | BinOp::Lt | BinOp::GtEq | BinOp::LtEq => {
                    let (col, lit) = match (extract_col(lhs), extract_col(rhs)) {
                        (Some(c), None) => (c, extract_lit(rhs)?),
                        (None, Some(_)) => {
                            return Err(HayashiError::Runtime(
                                "where: comparison must be `column OP literal`, \
                                 not `literal OP column`"
                                    .to_string(),
                            ));
                        }
                        (Some(_), Some(_)) => {
                            return Err(HayashiError::Runtime(
                                "where: cannot compare two columns".into(),
                            ));
                        }
                        (None, None) => {
                            return Err(HayashiError::Runtime(format!(
                                "where: comparison must involve a column name, \
                                 got `{}` OP `{}`",
                                expr_label(lhs),
                                expr_label(rhs)
                            )));
                        }
                    };
                    Ok(make_cmp(col, op, lit))
                }
                _ => Err(HayashiError::Runtime(format!(
                    "where: operator `{}` not supported in where clause",
                    binop_label(op)
                ))),
            },
            Expr::Not(inner) => Ok(RowPredicate::Not(Box::new(Self::from_expr(inner)?))),
            // true/false literais → predicado trivial (útil para debug)
            Expr::Bool(true) => Ok(RowPredicate::All),
            Expr::Bool(false) => Ok(RowPredicate::Not(Box::new(RowPredicate::All))),
            other => Err(HayashiError::Runtime(format!(
                "where: unsupported expression `{}`",
                expr_label(other)
            ))),
        }
    }

    /// Colunas referenciadas pelo predicado (para projeção mínima em parquet
    /// e para saber que colunas ler em CSV/DTA/Excel).
    pub fn referenced_columns(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_cols(&mut out);
        out
    }

    fn collect_cols(&self, out: &mut Vec<String>) {
        match self {
            RowPredicate::All => {}
            RowPredicate::Eq(c, _)
            | RowPredicate::Ne(c, _)
            | RowPredicate::Gt(c, _)
            | RowPredicate::Lt(c, _)
            | RowPredicate::Ge(c, _)
            | RowPredicate::Le(c, _)
            | RowPredicate::In(c, _) => {
                if !out.iter().any(|x| x == c) {
                    out.push(c.clone());
                }
            }
            RowPredicate::Not(p) => p.collect_cols(out),
            RowPredicate::And(ps) | RowPredicate::Or(ps) => {
                for p in ps {
                    p.collect_cols(out);
                }
            }
        }
    }

    /// Avalia o predicado contra uma linha concreta.
    /// Semântica SQL-like: valor nulo (NaN ou None) → comparação retorna `false`.
    pub fn evaluate(&self, row: &dyn RowAccess) -> bool {
        match self {
            RowPredicate::All => true,
            RowPredicate::Eq(c, lit) => cmp_eq(row, c, lit),
            RowPredicate::Ne(c, lit) => !is_null(row, c) && !cmp_eq(row, c, lit),
            RowPredicate::Gt(c, lit) => cmp_ord(row, c, lit, |a, b| a > b, |a, b| a > b),
            RowPredicate::Lt(c, lit) => cmp_ord(row, c, lit, |a, b| a < b, |a, b| a < b),
            RowPredicate::Ge(c, lit) => cmp_ord(row, c, lit, |a, b| a >= b, |a, b| a >= b),
            RowPredicate::Le(c, lit) => cmp_ord(row, c, lit, |a, b| a <= b, |a, b| a <= b),
            RowPredicate::In(c, lits) => {
                if is_null(row, c) {
                    return false;
                }
                // Se a coluna é numérica e todos os literais são numéricos,
                // comparar como f64. Senão, comparar como string.
                let col_f64 = row.get_f64(c);
                let all_num = lits.iter().all(|l| l.as_f64().is_some());
                match (col_f64, all_num) {
                    (Some(cv), true) => lits.iter().any(|l| cv == l.as_f64().unwrap_or(f64::NAN)),
                    _ => {
                        let Some(cs) = row.get_str(c) else {
                            return false;
                        };
                        lits.iter().any(|l| match l {
                            Literal::Str(s) => cs == s.as_str(),
                            _ => false,
                        })
                    }
                }
            }
            RowPredicate::Not(p) => !p.evaluate(row),
            RowPredicate::And(ps) => ps.iter().all(|p| p.evaluate(row)),
            RowPredicate::Or(ps) => ps.iter().any(|p| p.evaluate(row)),
        }
    }

    /// Avalia se um row group **pode** conter pelo menos uma linha que
    /// satisfaça o predicado, dados os bounds (min/max) das colunas.
    /// Retorna `true` em caso de dúvida (conservador — nunca poda um row
    /// group que poderia conter dados relevantes).
    ///
    /// Usado pelo row group pruning do Parquet antes de decodificar os dados.
    pub fn can_match(&self, bounds: &dyn GroupBounds) -> bool {
        match self {
            RowPredicate::All => true,
            RowPredicate::Eq(col, lit) => can_match_eq(bounds, col, lit),
            RowPredicate::Ne(col, lit) => can_match_ne(bounds, col, lit),
            RowPredicate::Gt(col, lit) => can_match_gt(bounds, col, lit),
            RowPredicate::Lt(col, lit) => can_match_lt(bounds, col, lit),
            RowPredicate::Ge(col, lit) => can_match_ge(bounds, col, lit),
            RowPredicate::Le(col, lit) => can_match_le(bounds, col, lit),
            RowPredicate::In(col, lits) => lits.iter().any(|lit| can_match_eq(bounds, col, lit)),
            // NOT: conservador — sempre keep, porque mesmo que o predicado
            // interno possa matchar todas as linhas, NOT matcharia nenhuma,
            // mas não temos garantia suficiente para podar.
            RowPredicate::Not(_) => true,
            // AND: todas as sub-condições devem poder matchar.
            RowPredicate::And(ps) => ps.iter().all(|p| p.can_match(bounds)),
            // OR: pelo menos uma sub-condição deve poder matchar.
            RowPredicate::Or(ps) => ps.iter().any(|p| p.can_match(bounds)),
        }
    }

    /// Cláusula SQL `WHERE ...` (sem a palavra-chave `WHERE`), com literais
    /// embutidos e escapados. Reaproveitada por SQLite e ODBC.
    pub fn to_sql(&self) -> String {
        match self {
            RowPredicate::All => "1=1".to_string(),
            RowPredicate::Eq(c, lit) => format!("{c} = {}", lit.to_sql_literal()),
            RowPredicate::Ne(c, lit) => format!("{c} <> {}", lit.to_sql_literal()),
            RowPredicate::Gt(c, lit) => format!("{c} > {}", lit.to_sql_literal()),
            RowPredicate::Lt(c, lit) => format!("{c} < {}", lit.to_sql_literal()),
            RowPredicate::Ge(c, lit) => format!("{c} >= {}", lit.to_sql_literal()),
            RowPredicate::Le(c, lit) => format!("{c} <= {}", lit.to_sql_literal()),
            RowPredicate::In(c, lits) => {
                let list: Vec<String> = lits.iter().map(|l| l.to_sql_literal()).collect();
                format!("{c} IN ({})", list.join(", "))
            }
            RowPredicate::Not(p) => format!("NOT ({})", p.to_sql()),
            RowPredicate::And(ps) => {
                let parts: Vec<String> = ps.iter().map(|p| format!("({})", p.to_sql())).collect();
                parts.join(" AND ")
            }
            RowPredicate::Or(ps) => {
                let parts: Vec<String> = ps.iter().map(|p| format!("({})", p.to_sql())).collect();
                parts.join(" OR ")
            }
        }
    }
}

// ── helpers ───────────────────────────────────────────────────────────────

fn make_cmp(col: String, op: &BinOp, lit: Literal) -> RowPredicate {
    match op {
        BinOp::Eq => RowPredicate::Eq(col, lit),
        BinOp::Ne => RowPredicate::Ne(col, lit),
        BinOp::Gt => RowPredicate::Gt(col, lit),
        BinOp::Lt => RowPredicate::Lt(col, lit),
        BinOp::GtEq => RowPredicate::Ge(col, lit),
        BinOp::LtEq => RowPredicate::Le(col, lit),
        _ => unreachable!("make_cmp chamado com op inválido"),
    }
}

fn extract_col(e: &Expr) -> Option<String> {
    if let Expr::Var(name) = e {
        Some(name.clone())
    } else {
        None
    }
}

fn extract_lit(e: &Expr) -> Result<Literal> {
    match e {
        Expr::Int(v) => Ok(Literal::Int(*v)),
        Expr::Float(v) => Ok(Literal::Float(*v)),
        Expr::Str(s) => Ok(Literal::Str(s.clone())),
        Expr::Bool(b) => Ok(Literal::Bool(*b)),
        Expr::Neg(inner) => match extract_lit(inner)? {
            Literal::Int(v) => Ok(Literal::Int(-v)),
            Literal::Float(v) => Ok(Literal::Float(-v)),
            _ => Err(HayashiError::Runtime(
                "where: unary minus only on numbers".into(),
            )),
        },
        other => Err(HayashiError::Runtime(format!(
            "where: expected a literal (number, string, bool), got `{}`",
            expr_label(other)
        ))),
    }
}

fn extract_list_literals(e: &Expr) -> Result<Vec<Literal>> {
    match e {
        Expr::List(items) => items.iter().map(extract_lit).collect(),
        Expr::RangeInclusive(a, b) => {
            let start = extract_lit(a)?.as_f64().ok_or_else(|| {
                HayashiError::Runtime("where: range bounds must be numbers".into())
            })?;
            let end = extract_lit(b)?.as_f64().ok_or_else(|| {
                HayashiError::Runtime("where: range bounds must be numbers".into())
            })?;
            let mut out = Vec::new();
            let mut i = start;
            while i <= end + 1e-9 {
                out.push(Literal::Float(i));
                i += 1.0;
            }
            Ok(out)
        }
        other => Err(HayashiError::Runtime(format!(
            "where: `in` expects a list literal `[...]`, got `{}`",
            expr_label(other)
        ))),
    }
}

fn is_null(row: &dyn RowAccess, col: &str) -> bool {
    match row.get_f64(col) {
        Some(v) if v.is_nan() => true,
        Some(_) => false,
        None => match row.get_str(col) {
            None => true,
            Some(s) => s.is_empty(),
        },
    }
}

fn cmp_eq(row: &dyn RowAccess, col: &str, lit: &Literal) -> bool {
    match lit {
        Literal::Str(s) => match row.get_str(col) {
            Some(cv) => cv == s.as_str(),
            None => match row.get_f64(col) {
                Some(cv) => s.parse::<f64>().map(|sv| cv == sv).unwrap_or(false),
                None => false,
            },
        },
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) => {
            let Some(lf) = lit.as_f64() else {
                return false;
            };
            match row.get_f64(col) {
                Some(cv) => !cv.is_nan() && cv == lf,
                None => match row.get_str(col) {
                    Some(cs) => cs.parse::<f64>().map(|cv| cv == lf).unwrap_or(false),
                    None => false,
                },
            }
        }
    }
}

fn cmp_ord(
    row: &dyn RowAccess,
    col: &str,
    lit: &Literal,
    fcmp: fn(f64, f64) -> bool,
    scmp: fn(&str, &str) -> bool,
) -> bool {
    match lit {
        Literal::Str(s) => match row.get_str(col) {
            Some(cv) => scmp(cv, s.as_str()),
            None => false,
        },
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) => {
            let Some(lf) = lit.as_f64() else {
                return false;
            };
            match row.get_f64(col) {
                Some(cv) => !cv.is_nan() && fcmp(cv, lf),
                None => match row.get_str(col) {
                    Some(cs) => cs.parse::<f64>().map(|cv| fcmp(cv, lf)).unwrap_or(false),
                    None => false,
                },
            }
        }
    }
}

// ── helpers para row group pruning (can_match) ───────────────────────────

/// `col == lit`: o row group pode conter `lit` se `min <= lit <= max`.
fn can_match_eq(bounds: &dyn GroupBounds, col: &str, lit: &Literal) -> bool {
    match lit {
        Literal::Str(s) => {
            if let Some((min, max)) = bounds.str_bounds(col) {
                s.as_str() >= min && s.as_str() <= max
            } else {
                true // sem bounds → conservador
            }
        }
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) => {
            if let Some(lf) = lit.as_f64() {
                if let Some((min, max)) = bounds.f64_bounds(col) {
                    lf >= min && lf <= max
                } else {
                    // tentar como string (coluna string com literal numérico)
                    true
                }
            } else {
                true
            }
        }
    }
}

/// `col != lit`: conservador — só podemos podar se todas as linhas
/// forem iguais a `lit` (min == max == lit). Caso raro; geralmente keep.
fn can_match_ne(bounds: &dyn GroupBounds, col: &str, lit: &Literal) -> bool {
    match lit {
        Literal::Str(s) => {
            if let Some((min, max)) = bounds.str_bounds(col) {
                // se min == max == lit, todas as linhas são iguais a lit → podar
                !(min == max && min == s.as_str())
            } else {
                true
            }
        }
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) => {
            if let Some(lf) = lit.as_f64() {
                if let Some((min, max)) = bounds.f64_bounds(col) {
                    !(min == max && min == lf)
                } else {
                    true
                }
            } else {
                true
            }
        }
    }
}

/// `col > lit`: pode matchar se `max > lit`.
fn can_match_gt(bounds: &dyn GroupBounds, col: &str, lit: &Literal) -> bool {
    match lit {
        Literal::Str(s) => {
            if let Some((_, max)) = bounds.str_bounds(col) {
                max > s.as_str()
            } else {
                true
            }
        }
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) => {
            if let Some(lf) = lit.as_f64() {
                if let Some((_, max)) = bounds.f64_bounds(col) {
                    max > lf
                } else {
                    true
                }
            } else {
                true
            }
        }
    }
}

/// `col < lit`: pode matchar se `min < lit`.
fn can_match_lt(bounds: &dyn GroupBounds, col: &str, lit: &Literal) -> bool {
    match lit {
        Literal::Str(s) => {
            if let Some((min, _)) = bounds.str_bounds(col) {
                min < s.as_str()
            } else {
                true
            }
        }
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) => {
            if let Some(lf) = lit.as_f64() {
                if let Some((min, _)) = bounds.f64_bounds(col) {
                    min < lf
                } else {
                    true
                }
            } else {
                true
            }
        }
    }
}

/// `col >= lit`: pode matchar se `max >= lit`.
fn can_match_ge(bounds: &dyn GroupBounds, col: &str, lit: &Literal) -> bool {
    match lit {
        Literal::Str(s) => {
            if let Some((_, max)) = bounds.str_bounds(col) {
                max >= s.as_str()
            } else {
                true
            }
        }
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) => {
            if let Some(lf) = lit.as_f64() {
                if let Some((_, max)) = bounds.f64_bounds(col) {
                    max >= lf
                } else {
                    true
                }
            } else {
                true
            }
        }
    }
}

/// `col <= lit`: pode matchar se `min <= lit`.
fn can_match_le(bounds: &dyn GroupBounds, col: &str, lit: &Literal) -> bool {
    match lit {
        Literal::Str(s) => {
            if let Some((min, _)) = bounds.str_bounds(col) {
                min <= s.as_str()
            } else {
                true
            }
        }
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) => {
            if let Some(lf) = lit.as_f64() {
                if let Some((min, _)) = bounds.f64_bounds(col) {
                    min <= lf
                } else {
                    true
                }
            } else {
                true
            }
        }
    }
}

fn binop_label(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Pow => "^",
        BinOp::Gt => ">",
        BinOp::Lt => "<",
        BinOp::GtEq => ">=",
        BinOp::LtEq => "<=",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::In => "in",
    }
}

fn expr_label(e: &Expr) -> String {
    match e {
        Expr::Var(name) => name.clone(),
        Expr::Int(v) => format!("{v}"),
        Expr::Float(v) => format!("{v}"),
        Expr::Str(s) => format!("\"{s}\""),
        Expr::Bool(b) => format!("{b}"),
        Expr::List(_) => "[...]".to_string(),
        _ => "<expr>".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRow {
        nums: std::collections::HashMap<String, f64>,
        strs: std::collections::HashMap<String, String>,
    }

    impl RowAccess for TestRow {
        fn get_f64(&self, col: &str) -> Option<f64> {
            self.nums.get(col).copied()
        }
        fn get_str(&self, col: &str) -> Option<&str> {
            self.strs.get(col).map(|s| s.as_str())
        }
    }

    fn row() -> TestRow {
        TestRow {
            nums: [
                ("price".to_string(), 100.0),
                ("volume".to_string(), 1_000_000.0),
            ]
            .into_iter()
            .collect(),
            strs: [
                ("ticker".to_string(), "AAPL".to_string()),
                ("sector".to_string(), "Tech".to_string()),
            ]
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn eq_string() {
        let p = RowPredicate::parse("ticker == \"AAPL\"").unwrap();
        assert!(p.evaluate(&row()));
        let p = RowPredicate::parse("ticker == \"MSFT\"").unwrap();
        assert!(!p.evaluate(&row()));
    }

    #[test]
    fn eq_number() {
        let p = RowPredicate::parse("price == 100").unwrap();
        assert!(p.evaluate(&row()));
    }

    #[test]
    fn gt_number() {
        let p = RowPredicate::parse("price > 50").unwrap();
        assert!(p.evaluate(&row()));
        let p = RowPredicate::parse("price > 200").unwrap();
        assert!(!p.evaluate(&row()));
    }

    #[test]
    fn and_or() {
        let p = RowPredicate::parse("ticker == \"AAPL\" && price > 50").unwrap();
        assert!(p.evaluate(&row()));
        let p = RowPredicate::parse("ticker == \"MSFT\" || price > 50").unwrap();
        assert!(p.evaluate(&row()));
    }

    #[test]
    fn not_pred() {
        let p = RowPredicate::parse("!(ticker == \"MSFT\")").unwrap();
        assert!(p.evaluate(&row()));
    }

    #[test]
    fn in_string_list() {
        let p = RowPredicate::parse("ticker in [\"AAPL\", \"MSFT\"]").unwrap();
        assert!(p.evaluate(&row()));
    }

    #[test]
    fn in_number_list() {
        let p = RowPredicate::parse("price in [50, 100, 200]").unwrap();
        assert!(p.evaluate(&row()));
    }

    #[test]
    fn to_sql_eq() {
        let p = RowPredicate::parse("ticker == \"AAPL\"").unwrap();
        assert_eq!(p.to_sql(), "ticker = 'AAPL'");
    }

    #[test]
    fn to_sql_in() {
        let p = RowPredicate::parse("price in [1, 2, 3]").unwrap();
        assert_eq!(p.to_sql(), "price IN (1, 2, 3)");
    }

    #[test]
    fn to_sql_string_escape() {
        let p = RowPredicate::parse("name == \"O'Brien\"").unwrap();
        assert_eq!(p.to_sql(), "name = 'O''Brien'");
    }

    #[test]
    fn rejects_literal_op_column() {
        let err = RowPredicate::parse("100 == price").unwrap_err();
        assert!(err.to_string().contains("column OP literal"));
    }

    #[test]
    fn rejects_arithmetic() {
        // `price + 1 == 100` → lhs não é coluna nem literal → erro claro.
        let err = RowPredicate::parse("price + 1 == 100").unwrap_err();
        assert!(err.to_string().contains("column"));
    }

    // ── Testes de row group pruning (can_match) ──────────────────────────

    struct TestBounds {
        f64: std::collections::HashMap<String, (f64, f64)>,
        str: std::collections::HashMap<String, (String, String)>,
    }

    impl GroupBounds for TestBounds {
        fn f64_bounds(&self, col: &str) -> Option<(f64, f64)> {
            self.f64.get(col).copied()
        }
        fn str_bounds(&self, col: &str) -> Option<(&str, &str)> {
            self.str.get(col).map(|(a, b)| (a.as_str(), b.as_str()))
        }
    }

    fn bounds() -> TestBounds {
        TestBounds {
            f64: [
                ("price".to_string(), (10.0, 500.0)),
                ("volume".to_string(), (0.0, 5_000_000.0)),
            ]
            .into_iter()
            .collect(),
            str: [("ticker".to_string(), ("AAA".to_string(), "ZZZ".to_string()))]
                .into_iter()
                .collect(),
        }
    }

    #[test]
    fn prune_eq_str_inside_bounds() {
        let p = RowPredicate::parse("ticker == \"AAPL\"").unwrap();
        assert!(p.can_match(&bounds())); // AAPL está entre AAA..ZZZ
    }

    #[test]
    fn prune_eq_str_outside_bounds() {
        let p = RowPredicate::parse("ticker == \"000\"").unwrap();
        assert!(!p.can_match(&bounds())); // 000 < AAA → poda
    }

    #[test]
    fn prune_eq_num_inside_bounds() {
        let p = RowPredicate::parse("price == 100").unwrap();
        assert!(p.can_match(&bounds())); // 10 <= 100 <= 500
    }

    #[test]
    fn prune_eq_num_outside_bounds() {
        let p = RowPredicate::parse("price == 999").unwrap();
        assert!(!p.can_match(&bounds())); // 999 > 500 → poda
    }

    #[test]
    fn prune_gt_within_bounds() {
        let p = RowPredicate::parse("price > 50").unwrap();
        assert!(p.can_match(&bounds())); // max=500 > 50
    }

    #[test]
    fn prune_gt_above_max() {
        let p = RowPredicate::parse("price > 600").unwrap();
        assert!(!p.can_match(&bounds())); // max=500 <= 600 → poda
    }

    #[test]
    fn prune_lt_below_min() {
        let p = RowPredicate::parse("price < 5").unwrap();
        assert!(!p.can_match(&bounds())); // min=10 >= 5 → poda
    }

    #[test]
    fn prune_and_one_side_cannot_match() {
        let p = RowPredicate::parse("ticker == \"AAPL\" && price > 600").unwrap();
        assert!(!p.can_match(&bounds())); // price > 600 impossível → poda
    }

    #[test]
    fn prune_or_one_side_can_match() {
        let p = RowPredicate::parse("ticker == \"000\" || price == 100").unwrap();
        assert!(p.can_match(&bounds())); // price == 100 é possível
    }

    #[test]
    fn prune_no_bounds_conservative() {
        let p = RowPredicate::parse("unknown_col == 42").unwrap();
        assert!(p.can_match(&bounds())); // sem bounds → keep
    }
}
