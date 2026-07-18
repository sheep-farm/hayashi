// ── Avaliação de operadores ───────────────────────────────────────────────────
use super::super::*;

/// Evaluates a scalar binary operator.
pub(in crate::lang::interpreter) fn eval_scalar_binop(
    op: &BinOp,
    l: Value,
    r: Value,
) -> Result<Value> {
    // Comparisons (works with any comparable type)
    match op {
        BinOp::Eq => {
            let eq = match (&l, &r) {
                (Value::Nil, Value::Nil) => true,
                (Value::Nil, _) | (_, Value::Nil) => false,
                (Value::Str(a), Value::Str(b)) => a == b,
                (Value::Bool(a), Value::Bool(b)) => a == b,
                _ => {
                    let a = value_as_f64(&l)?;
                    let b = value_as_f64(&r)?;
                    (a - b).abs() < f64::EPSILON
                }
            };
            return Ok(Value::Bool(eq));
        }
        BinOp::Ne => {
            let ne = match (&l, &r) {
                (Value::Nil, Value::Nil) => false,
                (Value::Nil, _) | (_, Value::Nil) => true,
                (Value::Str(a), Value::Str(b)) => a != b,
                (Value::Bool(a), Value::Bool(b)) => a != b,
                _ => {
                    let a = value_as_f64(&l)?;
                    let b = value_as_f64(&r)?;
                    (a - b).abs() >= f64::EPSILON
                }
            };
            return Ok(Value::Bool(ne));
        }
        _ => {}
    }

    // Arithmetic and numeric comparisons
    match (&l, &r) {
        // Int × Int → Int (for Add/Sub/Mul); Div/Pow → Float
        (Value::Int(a), Value::Int(b)) => match op {
            BinOp::Add => Ok(Value::Int(a + b)),
            BinOp::Sub => Ok(Value::Int(a - b)),
            BinOp::Mul => Ok(Value::Int(a * b)),
            BinOp::Div => Ok(Value::Float(*a as f64 / *b as f64)),
            BinOp::Mod => Ok(Value::Int(a % b)),
            BinOp::Pow => Ok(Value::Float((*a as f64).powf(*b as f64))),
            BinOp::Gt => Ok(Value::Bool(a > b)),
            BinOp::Lt => Ok(Value::Bool(a < b)),
            BinOp::GtEq => Ok(Value::Bool(a >= b)),
            BinOp::LtEq => Ok(Value::Bool(a <= b)),
            BinOp::And | BinOp::Or | BinOp::Eq | BinOp::Ne | BinOp::In => unreachable!(),
        },
        // Any Float → Float
        _ => {
            // String concatenation
            if let (BinOp::Add, Value::Str(a), Value::Str(b)) = (op, &l, &r) {
                return Ok(Value::Str(format!("{a}{b}")));
            }
            let a = value_as_f64(&l)?;
            let b = value_as_f64(&r)?;
            match op {
                BinOp::Add => Ok(Value::Float(a + b)),
                BinOp::Sub => Ok(Value::Float(a - b)),
                BinOp::Mul => Ok(Value::Float(a * b)),
                BinOp::Div => Ok(Value::Float(a / b)),
                BinOp::Mod => Ok(Value::Float(a % b)),
                BinOp::Pow => Ok(Value::Float(a.powf(b))),
                BinOp::Gt => Ok(Value::Bool(a > b)),
                BinOp::Lt => Ok(Value::Bool(a < b)),
                BinOp::GtEq => Ok(Value::Bool(a >= b)),
                BinOp::LtEq => Ok(Value::Bool(a <= b)),
                BinOp::And | BinOp::Or | BinOp::Eq | BinOp::Ne | BinOp::In => unreachable!(),
            }
        }
    }
}
