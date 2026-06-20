use thiserror::Error;

#[derive(Debug, Error)]
pub enum HayashiError {
    #[error("Lexer error at line {line}: {msg}")]
    Lex { line: usize, msg: String },

    #[error("Parse error at line {line}: {msg}")]
    Parse { line: usize, msg: String },

    #[error("Type error: {0}")]
    Type(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    // Sentinelas de controle de fluxo — capturadas internamente, nunca expostas ao usuário
    #[error("return")]
    Return,
    #[error("break")]
    Break,
    #[error("continue")]
    Continue,
}

pub type Result<T> = std::result::Result<T, HayashiError>;
