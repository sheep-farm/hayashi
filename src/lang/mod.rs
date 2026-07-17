// `Value` uses `Arc` for DataFrame/List/Dict/Series/UserFn (to enable `parallel for`)
// even though `Value` as a whole is not `Send+Sync` (model results use `Rc`).
#![allow(clippy::arc_with_non_send_sync)]

pub mod ast;
pub mod commands;
pub mod dap;
pub mod error;
pub mod help;
pub mod interpreter;
pub mod lexer;
pub mod parser;
pub mod plugin;
pub mod predicate;

use error::{HayashiError, Result};
use interpreter::Interpreter;
use lexer::Lexer;
use parser::Parser;

pub fn run_source(src: &str, interp: &mut Interpreter) -> Result<()> {
    run_source_with_path(src, interp, None)
}

pub fn run_source_with_path(
    src: &str,
    interp: &mut Interpreter,
    source_path: Option<&std::path::Path>,
) -> Result<()> {
    run_source_verbose(src, interp, false, source_path)
}

pub fn run_source_verbose(
    src: &str,
    interp: &mut Interpreter,
    verbose: bool,
    source_path: Option<&std::path::Path>,
) -> Result<()> {
    if let Some(path) = source_path {
        interp.set_current_source(path);
    }
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| annotate_error(src, &e))?;
    if verbose {
        eprintln!("[hayashi] {} tokens parsed", tokens.len());
    }
    let mut parser = Parser::new(tokens);
    let stmts = parser
        .parse_program()
        .map_err(|e| annotate_error(src, &e))?;
    if verbose {
        eprintln!("[hayashi] {} statements", stmts.len());
    }
    for (i, spanned) in stmts.iter().enumerate() {
        if verbose {
            eprintln!(
                "[hayashi] exec #{} (line {}): {:?}",
                i + 1,
                spanned.1,
                stmt_label(&spanned.0)
            );
        }
        interp.exec(spanned).map_err(|e| annotate_error(src, &e))?;
    }
    Ok(())
}

fn strip_runtime_prefixes(msg: &str) -> &str {
    let mut s = msg;
    loop {
        if let Some(rest) = s.strip_prefix("Runtime error: ") {
            s = rest;
        } else if let Some(rest) = s.strip_prefix("Type error: ") {
            s = rest;
        } else {
            break;
        }
    }
    s
}

fn extract_line_number(e: &HayashiError) -> Option<usize> {
    match e {
        HayashiError::Lex { line, .. } | HayashiError::Parse { line, .. } => Some(*line),
        HayashiError::Runtime(msg) | HayashiError::Type(msg) => {
            let clean = strip_runtime_prefixes(msg);
            if let Some(rest) = clean.strip_prefix("line ") {
                rest.split(':').next().and_then(|n| n.trim().parse().ok())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn error_message(e: &HayashiError) -> String {
    match e {
        HayashiError::Lex { line, msg } => format!("Lexer error at line {line}: {msg}"),
        HayashiError::Parse { line, msg } => format!("Parse error at line {line}: {msg}"),
        HayashiError::Runtime(msg) | HayashiError::Type(msg) => {
            let clean = strip_runtime_prefixes(msg);
            if let Some(rest) = clean.strip_prefix("line ") {
                if let Some(pos) = rest.find(':') {
                    let core = rest[pos + 1..].trim();
                    return core.to_string();
                }
            }
            clean.to_string()
        }
        _ => format!("{e}"),
    }
}

fn annotate_error(src: &str, e: &HayashiError) -> HayashiError {
    let line_num = match extract_line_number(e) {
        Some(n) => n,
        None => return e.clone(),
    };
    let lines: Vec<&str> = src.lines().collect();
    if line_num == 0 || line_num > lines.len() {
        return e.clone();
    }
    let line_src = lines[line_num - 1];
    let msg = error_message(e);
    let pad = " ".repeat(line_num.to_string().len());
    let preview = format!(
        "line {line_num}: {msg}\n  {line_num} │ {line_src}\n  {pad} │ {}",
        "^".repeat(line_src.trim().len())
    );
    HayashiError::Annotated(preview)
}

fn stmt_label(s: &ast::Stmt) -> &'static str {
    match s {
        ast::Stmt::Let { .. } => "let",
        ast::Stmt::Const { .. } => "const",
        ast::Stmt::Assign { .. } => "assign",
        ast::Stmt::Load { .. } => "load",
        ast::Stmt::Generate { .. } => "generate",
        ast::Stmt::Predict { .. } => "predict",
        ast::Stmt::Print(..) => "print",
        ast::Stmt::Export { .. } => "export",
        ast::Stmt::Replace { .. } => "replace",
        ast::Stmt::Count { .. } => "count",
        ast::Stmt::Tsset { .. } => "tsset",
        ast::Stmt::If { .. } => "if",
        ast::Stmt::For { .. } => "for",
        ast::Stmt::ParallelFor { .. } => "parallel for",
        ast::Stmt::While { .. } => "while",
        ast::Stmt::Fn { .. } => "fn",
        ast::Stmt::Return(_) => "return",
        ast::Stmt::Break => "break",
        ast::Stmt::Continue => "continue",
        ast::Stmt::QuietlyOn => "quietly on",
        ast::Stmt::QuietlyOff => "quietly off",
        ast::Stmt::TryCatch { .. } => "try/catch",
        ast::Stmt::Input { .. } => "input",
        ast::Stmt::Display(_) => "display",
        ast::Stmt::Expr(_) => "expr",
        ast::Stmt::Block(_) => "block",
    }
}
