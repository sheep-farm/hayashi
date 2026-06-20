pub mod ast;
pub mod error;
pub mod interpreter;
pub mod lexer;
pub mod parser;

use error::Result;
use interpreter::Interpreter;
use lexer::Lexer;
use parser::Parser;

pub fn run_source(src: &str, interp: &mut Interpreter) -> Result<()> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let stmts = parser.parse_program()?;
    for stmt in &stmts {
        interp.exec(stmt)?;
    }
    Ok(())
}
