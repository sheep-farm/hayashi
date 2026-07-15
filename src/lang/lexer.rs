use crate::lang::error::{HayashiError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literais
    Ident(String),
    StringLit(String),
    FStringLit(String),
    DocString(String),
    Float(f64),
    Int(i64),
    Bool(bool),

    // Palavras-chave
    Let,
    Load,
    Print,
    Export,
    Generate,
    Predict,
    Replace,
    If,
    Else,
    For,
    Parallel,
    In,
    Count,
    Tsset,
    While,
    Fn,
    Return,
    Break,
    Continue,
    Quietly,
    Nil,

    // Time-series operators: L.x  L2.x  F.x  D.x
    TsLag(usize),
    TsLead(usize),
    TsDiff(usize),

    // Operators
    Eq,         // =
    EqEq,       // ==
    BangEq,     // !=
    Tilde,      // ~
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Caret,      // ^
    Percent,    // %
    StarStar,   // **
    Pipe,       // |
    Colon,      // :
    ColonColon, // ::
    Comma,      // ,
    Dot,        // .
    DotDot,     // ..
    DotDotEq,   // ..=
    Bang,       // !
    Lt,         // <
    LtEq,       // <=
    Gt,         // >
    GtEq,       // >=
    And,        // &&
    Or,         // ||
    FatArrow,   // =>
    PipeRight,  // |>
    PlusEq,     // +=
    MinusEq,    // -=
    StarEq,     // *=
    SlashEq,    // /=
    PercentEq,  // %=
    PlusPlus,   // ++
    MinusMinus, // --

    // Delimitadores
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,

    // Especiais
    Newline,
    Eof,
}

pub struct Lexer {
    src: Vec<char>,
    pos: usize,
    pub line: usize,
}

impl Lexer {
    pub fn new(src: &str) -> Self {
        Self {
            src: src.chars().collect(),
            pos: 0,
            line: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.src.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.src.get(self.pos).copied();
        if c == Some('\n') {
            self.line += 1;
        }
        self.pos += 1;
        c
    }

    fn skip_whitespace_inline(&mut self) {
        while matches!(self.peek(), Some(' ') | Some('\t') | Some('\r')) {
            self.advance();
        }
    }

    fn read_string(&mut self) -> Result<Token> {
        let line = self.line;
        let mut s = String::new();
        loop {
            match self.advance() {
                Some('"') => return Ok(Token::StringLit(s)),
                Some('\\') => match self.advance() {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    _ => s.push('\\'),
                },
                Some(c) => s.push(c),
                None => {
                    return Err(HayashiError::Lex {
                        line,
                        msg: "unterminated string".into(),
                    })
                }
            }
        }
    }

    fn read_number(&mut self, first: char) -> Token {
        let mut s = String::from(first);
        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.advance();
            } else if c == '.' && !is_float && self.peek2() != Some('.') {
                // only consume the dot if it is not ".." (range)
                is_float = true;
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        if is_float {
            Token::Float(s.parse().unwrap_or(0.0))
        } else {
            Token::Int(s.parse().unwrap_or(0))
        }
    }

    fn read_ident(&mut self, first: char) -> Token {
        let mut s = String::from(first);
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        match s.as_str() {
            "let" => Token::Let,
            "load" => Token::Load,
            "print" => Token::Print,
            "export" => Token::Export,
            "generate" => Token::Generate,
            "gen" => Token::Generate,
            "predict" => Token::Predict,
            "replace" => Token::Replace,
            "if" => Token::If,
            "else" => Token::Else,
            "for" => Token::For,
            "parallel" => Token::Parallel,
            "in" => Token::In,
            "count" => Token::Count,
            "tsset" => Token::Tsset,
            "while" => Token::While,
            "fn" => Token::Fn,
            "return" => Token::Return,
            "break" => Token::Break,
            "continue" => Token::Continue,
            "quietly" => Token::Quietly,
            "nil" => Token::Nil,
            "true" => Token::Bool(true),
            "false" => Token::Bool(false),
            _ => Token::Ident(s),
        }
    }

    // Converts Token::Ident("L"/"L2"/"F"/"D" etc.) into TsLag/TsLead/TsDiff
    // if the next char is '.'.  Consumes the dot.
    fn maybe_ts_op(&mut self, tok: Token) -> Token {
        let Token::Ident(ref s) = tok else { return tok };
        let mut chars = s.chars();
        let first = match chars.next() {
            Some(c @ ('L' | 'F' | 'D')) => c,
            _ => return tok,
        };
        let rest = chars.as_str();
        if !rest.is_empty() && !rest.chars().all(|c| c.is_ascii_digit()) {
            return tok; // e.g. "LEVEL" is not a ts operator
        }
        if self.peek() != Some('.') {
            return tok;
        }
        // do not consume if it is ".." (range) — ts op needs a name after "."
        // check if the char after "." is a letter or underscore
        if self.peek2().map(|c| c == '.').unwrap_or(false) {
            return tok;
        }
        self.advance(); // consume '.'
        let n: usize = if rest.is_empty() {
            1
        } else {
            rest.parse().unwrap_or(1)
        };
        match first {
            'L' => Token::TsLag(n),
            'F' => Token::TsLead(n),
            'D' => Token::TsDiff(n),
            _ => unreachable!(),
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<(Token, usize)>> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace_inline();
            let line = self.line;
            match self.advance() {
                None => {
                    tokens.push((Token::Eof, line));
                    break;
                }
                Some('#') => {
                    if self.peek() == Some('#') {
                        self.advance(); // segundo #
                        let mut s = String::new();
                        while !matches!(self.peek(), Some('\n') | None) {
                            s.push(self.advance().unwrap());
                        }
                        tokens.push((Token::DocString(s.trim().to_string()), line));
                    } else {
                        while !matches!(self.peek(), Some('\n') | None) {
                            self.advance();
                        }
                    }
                }
                Some('\n') => tokens.push((Token::Newline, line)),
                Some('"') => tokens.push((self.read_string()?, line)),
                Some(c) if c.is_ascii_digit() => tokens.push((self.read_number(c), line)),
                Some('f') if self.peek() == Some('"') => {
                    self.advance(); // consume "
                    match self.read_string()? {
                        Token::StringLit(s) => tokens.push((Token::FStringLit(s), line)),
                        _ => unreachable!(),
                    }
                }
                Some(c) if c.is_alphabetic() || c == '_' => {
                    let tok = self.read_ident(c);
                    let tok = self.maybe_ts_op(tok);
                    tokens.push((tok, line));
                }
                Some('=') => {
                    if self.peek() == Some('=') {
                        self.advance();
                        tokens.push((Token::EqEq, line));
                    } else if self.peek() == Some('>') {
                        self.advance();
                        tokens.push((Token::FatArrow, line));
                    } else {
                        tokens.push((Token::Eq, line));
                    }
                }
                Some('!') => {
                    if self.peek() == Some('=') {
                        self.advance();
                        tokens.push((Token::BangEq, line));
                    } else {
                        tokens.push((Token::Bang, line));
                    }
                }
                Some('<') => {
                    if self.peek() == Some('=') {
                        self.advance();
                        tokens.push((Token::LtEq, line));
                    } else {
                        tokens.push((Token::Lt, line));
                    }
                }
                Some('>') => {
                    if self.peek() == Some('=') {
                        self.advance();
                        tokens.push((Token::GtEq, line));
                    } else {
                        tokens.push((Token::Gt, line));
                    }
                }
                Some('&') => {
                    if self.peek() == Some('&') {
                        self.advance();
                    }
                    tokens.push((Token::And, line));
                }
                Some('|') => {
                    if self.peek() == Some('|') {
                        self.advance();
                        tokens.push((Token::Or, line));
                    } else if self.peek() == Some('>') {
                        self.advance();
                        tokens.push((Token::PipeRight, line));
                    } else {
                        tokens.push((Token::Pipe, line));
                    }
                }
                Some('.') => {
                    if self.peek() == Some('.') {
                        self.advance();
                        if self.peek() == Some('=') {
                            self.advance();
                            tokens.push((Token::DotDotEq, line));
                        } else {
                            tokens.push((Token::DotDot, line));
                        }
                    } else {
                        tokens.push((Token::Dot, line));
                    }
                }
                Some('~') => tokens.push((Token::Tilde, line)),
                Some('+') => {
                    if self.peek() == Some('=') {
                        self.advance();
                        tokens.push((Token::PlusEq, line));
                    } else if self.peek() == Some('+') {
                        self.advance();
                        tokens.push((Token::PlusPlus, line));
                    } else {
                        tokens.push((Token::Plus, line));
                    }
                }
                Some('-') => {
                    if self.peek() == Some('=') {
                        self.advance();
                        tokens.push((Token::MinusEq, line));
                    } else if self.peek() == Some('-') {
                        self.advance();
                        tokens.push((Token::MinusMinus, line));
                    } else {
                        tokens.push((Token::Minus, line));
                    }
                }
                Some('*') => {
                    if self.peek() == Some('*') {
                        self.advance();
                        tokens.push((Token::StarStar, line));
                    } else if self.peek() == Some('=') {
                        self.advance();
                        tokens.push((Token::StarEq, line));
                    } else {
                        tokens.push((Token::Star, line));
                    }
                }
                Some('/') => {
                    if self.peek() == Some('/') {
                        while !matches!(self.peek(), Some('\n') | None) {
                            self.advance();
                        }
                    } else if self.peek() == Some('=') {
                        self.advance();
                        tokens.push((Token::SlashEq, line));
                    } else {
                        tokens.push((Token::Slash, line));
                    }
                }
                Some('%') => {
                    if self.peek() == Some('=') {
                        self.advance();
                        tokens.push((Token::PercentEq, line));
                    } else {
                        tokens.push((Token::Percent, line));
                    }
                }
                Some('^') => tokens.push((Token::Caret, line)),
                Some(':') => {
                    if self.peek() == Some(':') {
                        self.advance();
                        tokens.push((Token::ColonColon, line));
                    } else {
                        tokens.push((Token::Colon, line));
                    }
                }
                Some(',') => tokens.push((Token::Comma, line)),
                Some('(') => tokens.push((Token::LParen, line)),
                Some(')') => tokens.push((Token::RParen, line)),
                Some('[') => tokens.push((Token::LBracket, line)),
                Some(']') => tokens.push((Token::RBracket, line)),
                Some('{') => tokens.push((Token::LBrace, line)),
                Some('}') => tokens.push((Token::RBrace, line)),
                Some(c) => {
                    return Err(HayashiError::Lex {
                        line,
                        msg: format!("unexpected character '{c}'"),
                    })
                }
            }
        }
        Ok(tokens)
    }
}
