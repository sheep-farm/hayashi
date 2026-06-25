pub mod ast;
pub mod error;
pub mod help;
pub mod interpreter;
pub mod lexer;
pub mod parser;

use error::Result;
use interpreter::Interpreter;
use lexer::Lexer;
use parser::Parser;

pub fn run_source(src: &str, interp: &mut Interpreter) -> Result<()> {
    run_source_verbose(src, interp, false)
}

pub fn run_source_verbose(src: &str, interp: &mut Interpreter, verbose: bool) -> Result<()> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize()?;
    if verbose {
        eprintln!("[hayashi] {} tokens parsed", tokens.len());
    }
    let mut parser = Parser::new(tokens);
    let stmts = parser.parse_program()?;
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
        interp.exec(spanned)?;
    }
    Ok(())
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
        ast::Stmt::While { .. } => "while",
        ast::Stmt::Fn { .. } => "fn",
        ast::Stmt::Return(_) => "return",
        ast::Stmt::Break => "break",
        ast::Stmt::Continue => "continue",
        ast::Stmt::TryCatch { .. } => "try/catch",
        ast::Stmt::Input { .. } => "input",
        ast::Stmt::Display(_) => "display",
        ast::Stmt::Expr(_) => "expr",
    }
}
