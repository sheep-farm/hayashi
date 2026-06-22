use crate::lang::error::{HayashiError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literais
    Ident(String),
    StringLit(String),
    FStringLit(String),
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
    In,
    Count,
    Tsset,
    While,
    Fn,
    Return,
    Break,
    Continue,

    // Operadores de série temporal: L.x  L2.x  F.x  D.x
    TsLag(usize),
    TsLead(usize),
    TsDiff(usize),

    // Operadores
    Eq,       // =
    EqEq,     // ==
    BangEq,   // !=
    Tilde,    // ~
    Plus,     // +
    Minus,    // -
    Star,     // *
    Slash,    // /
    Caret,    // ^
    Pipe,     // |
    Colon,    // :
    Comma,    // ,
    Dot,      // .
    DotDot,   // ..
    Bang,     // !
    Lt,       // <
    LtEq,     // <=
    Gt,       // >
    GtEq,     // >=
    And,      // &&
    Or,       // ||

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
        Self { src: src.chars().collect(), pos: 0, line: 1 }
    }

    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.src.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.src.get(self.pos).copied();
        if c == Some('\n') { self.line += 1; }
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
                None => return Err(HayashiError::Lex { line, msg: "unterminated string".into() }),
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
                // só consome o ponto se não for ".." (range)
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
            "let"      => Token::Let,
            "load"     => Token::Load,
            "print"    => Token::Print,
            "export"   => Token::Export,
            "generate" => Token::Generate,
            "gen"      => Token::Generate,
            "predict"  => Token::Predict,
            "replace"  => Token::Replace,
            "if"       => Token::If,
            "else"     => Token::Else,
            "for"      => Token::For,
            "in"       => Token::In,
            "count"    => Token::Count,
            "tsset"    => Token::Tsset,
            "while"    => Token::While,
            "fn"       => Token::Fn,
            "return"   => Token::Return,
            "break"    => Token::Break,
            "continue" => Token::Continue,
            "true"     => Token::Bool(true),
            "false"    => Token::Bool(false),
            _          => Token::Ident(s),
        }
    }

    // Converte Token::Ident("L"/"L2"/"F"/"D" etc.) em TsLag/TsLead/TsDiff
    // se o próximo char for '.'.  Consome o ponto.
    fn maybe_ts_op(&mut self, tok: Token) -> Token {
        let Token::Ident(ref s) = tok else { return tok };
        let mut chars = s.chars();
        let first = match chars.next() {
            Some(c @ ('L' | 'F' | 'D')) => c,
            _ => return tok,
        };
        let rest = chars.as_str();
        if !rest.is_empty() && !rest.chars().all(|c| c.is_ascii_digit()) {
            return tok; // ex: "LEVEL" não é operador ts
        }
        if self.peek() != Some('.') {
            return tok;
        }
        // não consome se for ".." (range) — ts op precisa de nome após "."
        // verifica se o char depois de "." é letra ou underscore
        if self.peek2().map(|c| c == '.').unwrap_or(false) {
            return tok;
        }
        self.advance(); // consome '.'
        let n: usize = if rest.is_empty() { 1 } else { rest.parse().unwrap_or(1) };
        match first {
            'L' => Token::TsLag(n),
            'F' => Token::TsLead(n),
            'D' => Token::TsDiff(n),
            _   => unreachable!(),
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<(Token, usize)>> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace_inline();
            let line = self.line;
            match self.advance() {
                None => { tokens.push((Token::Eof, line)); break; }
                Some('#') => { while !matches!(self.peek(), Some('\n') | None) { self.advance(); } }
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
                    if self.peek() == Some('=') { self.advance(); tokens.push((Token::EqEq, line)); }
                    else { tokens.push((Token::Eq, line)); }
                }
                Some('!') => {
                    if self.peek() == Some('=') { self.advance(); tokens.push((Token::BangEq, line)); }
                    else { tokens.push((Token::Bang, line)); }
                }
                Some('<') => {
                    if self.peek() == Some('=') { self.advance(); tokens.push((Token::LtEq, line)); }
                    else { tokens.push((Token::Lt, line)); }
                }
                Some('>') => {
                    if self.peek() == Some('=') { self.advance(); tokens.push((Token::GtEq, line)); }
                    else { tokens.push((Token::Gt, line)); }
                }
                Some('&') => {
                    if self.peek() == Some('&') { self.advance(); }
                    tokens.push((Token::And, line));
                }
                Some('|') => {
                    if self.peek() == Some('|') { self.advance(); tokens.push((Token::Or, line)); }
                    else { tokens.push((Token::Pipe, line)); }
                }
                Some('.') => {
                    if self.peek() == Some('.') { self.advance(); tokens.push((Token::DotDot, line)); }
                    else { tokens.push((Token::Dot, line)); }
                }
                Some('~') => tokens.push((Token::Tilde, line)),
                Some('+') => tokens.push((Token::Plus, line)),
                Some('-') => tokens.push((Token::Minus, line)),
                Some('*') => tokens.push((Token::Star, line)),
                Some('/') => {
                    if self.peek() == Some('/') {
                        // comentário de linha: consome até \n
                        while !matches!(self.peek(), Some('\n') | None) {
                            self.advance();
                        }
                    } else {
                        tokens.push((Token::Slash, line));
                    }
                }
                Some('^') => tokens.push((Token::Caret, line)),
                Some(':') => tokens.push((Token::Colon, line)),
                Some(',') => tokens.push((Token::Comma, line)),
                Some('(') => tokens.push((Token::LParen, line)),
                Some(')') => tokens.push((Token::RParen, line)),
                Some('[') => tokens.push((Token::LBracket, line)),
                Some(']') => tokens.push((Token::RBracket, line)),
                Some('{') => tokens.push((Token::LBrace, line)),
                Some('}') => tokens.push((Token::RBrace, line)),
                Some(c) => return Err(HayashiError::Lex { line, msg: format!("unexpected character '{c}'") }),
            }
        }
        Ok(tokens)
    }
}
