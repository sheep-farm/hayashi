use crate::lang::ast::*;
use crate::lang::error::{HayashiError, Result};
use crate::lang::lexer::Token;

pub struct Parser {
    tokens: Vec<(Token, usize)>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<(Token, usize)>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).map(|(t, _)| t).unwrap_or(&Token::Eof)
    }

    fn line(&self) -> usize {
        self.tokens.get(self.pos).map(|(_, l)| *l).unwrap_or(0)
    }

    fn advance(&mut self) -> &Token {
        let t = self.tokens.get(self.pos).map(|(t, _)| t).unwrap_or(&Token::Eof);
        self.pos += 1;
        t
    }

    fn expect_ident(&mut self) -> Result<String> {
        let line = self.line();
        match self.advance().clone() {
            Token::Ident(s) => Ok(s),
            t => Err(HayashiError::Parse { line, msg: format!("expected identifier, got {t:?}") }),
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        let line = self.line();
        let got = self.advance().clone();
        if &got == expected {
            Ok(())
        } else {
            Err(HayashiError::Parse { line, msg: format!("expected {expected:?}, got {got:?}") })
        }
    }

    fn skip_newlines(&mut self) {
        while self.peek() == &Token::Newline {
            self.advance();
        }
    }

    // ── Fórmula ──────────────────────────────────────────────────────────────

    fn parse_formula(&mut self, lhs: String) -> Result<Formula> {
        // consome ~
        self.advance();
        let mut rhs = Vec::new();
        let mut fe = Vec::new();
        let mut in_fe = false;

        loop {
            match self.peek().clone() {
                Token::Newline | Token::Eof | Token::RParen | Token::Comma => break,
                Token::Pipe => { self.advance(); in_fe = true; }
                Token::Plus | Token::Minus => { self.advance(); }
                Token::Ident(name) => {
                    self.advance();
                    if self.peek() == &Token::LParen {
                        // C(var) / log(var) / sqrt(var) / I(...)
                        self.advance();
                        let inner = self.expect_ident()?;
                        self.expect(&Token::RParen)?;
                        if in_fe {
                            fe.push(format!("{name}({inner})"));
                        } else {
                            let term = match name.as_str() {
                                "C" => RhsTerm::Categorical(inner),
                                f   => RhsTerm::Transform(f.to_string(), inner),
                            };
                            rhs.push(term);
                        }
                    } else if self.peek() == &Token::Colon {
                        // interação pura: x1:x2
                        self.advance();
                        let right = self.expect_ident()?;
                        rhs.push(RhsTerm::Interaction(name, right));
                    } else if self.peek() == &Token::Star {
                        // x1*x2 → x1 + x2 + x1:x2
                        self.advance();
                        let right = self.expect_ident()?;
                        rhs.push(RhsTerm::Var(name.clone()));
                        rhs.push(RhsTerm::Var(right.clone()));
                        rhs.push(RhsTerm::Interaction(name, right));
                    } else if in_fe {
                        fe.push(name);
                    } else {
                        rhs.push(RhsTerm::Var(name));
                    }
                }
                _ => { self.advance(); }
            }
        }
        Ok(Formula { lhs, rhs, fe })
    }

    // ── Expressão aritmética (Pratt parsing) ────────────────────────────────

    /// Ponto de entrada: expressão com comparações (menor precedência)
    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_additive()?;
        loop {
            let op = match self.peek() {
                Token::Gt    => BinOp::Gt,
                Token::Lt    => BinOp::Lt,
                Token::GtEq  => BinOp::GtEq,
                Token::LtEq  => BinOp::LtEq,
                Token::EqEq  => BinOp::Eq,
                Token::BangEq => BinOp::Ne,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_additive()?;
            lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Token::Plus  => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_multiplicative()?;
            lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_power()?;
        loop {
            let op = match self.peek() {
                Token::Star  => BinOp::Mul,
                Token::Slash => BinOp::Div,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_power()?;
            lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_power(&mut self) -> Result<Expr> {
        let base = self.parse_unary()?;
        if self.peek() == &Token::Caret {
            self.advance();
            let exp = self.parse_unary()?; // right-associative
            Ok(Expr::BinOp { op: BinOp::Pow, lhs: Box::new(base), rhs: Box::new(exp) })
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        if self.peek() == &Token::Minus {
            self.advance();
            let inner = self.parse_primary()?;
            return Ok(Expr::Neg(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        let line = self.line();
        match self.peek().clone() {
            Token::Float(f) => { self.advance(); Ok(Expr::Float(f)) }
            Token::Int(i)   => { self.advance(); Ok(Expr::Int(i)) }
            Token::Bool(b)  => { self.advance(); Ok(Expr::Bool(b)) }
            Token::StringLit(s) => { self.advance(); Ok(Expr::Str(s)) }

            // Agrupamento: (expr)
            Token::LParen => {
                self.advance();
                let inner = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }

            // Fórmula sem LHS: ~ z1 + z2
            Token::Tilde => {
                let formula = self.parse_formula(String::new())?;
                Ok(Expr::Formula(formula))
            }

            Token::Ident(name) => {
                self.advance();

                if self.peek() == &Token::Tilde {
                    let formula = self.parse_formula(name)?;
                    return Ok(Expr::Formula(formula));
                }

                if self.peek() == &Token::LParen {
                    self.advance();
                    let (args, opts) = self.parse_call_args()?;
                    self.expect(&Token::RParen)?;
                    let mut expr = Expr::Call { func: name, args, opts };

                    while self.peek() == &Token::Dot {
                        self.advance();
                        let field = self.expect_ident()?;
                        let (fargs, fopts) = if self.peek() == &Token::LParen {
                            self.advance();
                            let r = self.parse_call_args()?;
                            self.expect(&Token::RParen)?;
                            r
                        } else {
                            (vec![], vec![])
                        };
                        expr = Expr::Field { obj: Box::new(expr), field, args: fargs, opts: fopts };
                    }
                    return Ok(expr);
                }

                Ok(Expr::Var(name))
            }

            _ => Err(HayashiError::Parse { line, msg: format!("unexpected token {:?}", self.peek()) }),
        }
    }

    fn parse_call_args(&mut self) -> Result<(Vec<Expr>, Vec<Opt>)> {
        let mut args = Vec::new();
        let mut opts = Vec::new();

        while !matches!(self.peek(), Token::RParen | Token::Eof | Token::Newline) {
            // opt=value  ou  expr normal
            if let Token::Ident(name) = self.peek().clone() {
                // lookahead: é opt=val?
                if self.tokens.get(self.pos + 1).map(|(t, _)| t == &Token::Eq).unwrap_or(false) {
                    self.advance(); // nome
                    self.advance(); // =
                    // Identificador bare em posição de opção é átomo de string (ex: cov=HC3)
                    let val = if let Token::Ident(kw) = self.peek().clone() {
                        self.advance();
                        Expr::Str(kw)
                    } else {
                        // opções numéricas usam aritmética completa
                        self.parse_expr()?
                    };
                    opts.push(Opt { name, value: val });
                } else {
                    // Dentro de call args, fórmulas contêm '+' que não deve ser
                    // interpretado como adição — usamos parse_primary aqui.
                    // Só promovemos para parse_expr em contextos claramente aritméticos
                    // (ex: dentro do generate).
                    args.push(self.parse_primary()?);
                }
            } else {
                args.push(self.parse_primary()?);
            }

            if self.peek() == &Token::Comma { self.advance(); }
        }
        Ok((args, opts))
    }

    // ── Statement ────────────────────────────────────────────────────────────

    fn parse_stmt(&mut self) -> Result<Option<Stmt>> {
        self.skip_newlines();
        let line = self.line();

        match self.peek().clone() {
            Token::Eof => Ok(None),

            Token::Let => {
                self.advance();
                let name = self.expect_ident()?;
                self.expect(&Token::Eq)?;
                let value = self.parse_expr()?;
                Ok(Some(Stmt::Let { name, value }))
            }

            Token::Load => {
                self.advance();
                let path = self.parse_expr()?;
                // as nome (opcional)
                let alias = if let Token::Ident(kw) = self.peek().clone() {
                    if kw == "as" {
                        self.advance();
                        self.expect_ident()?
                    } else {
                        "df".to_string()
                    }
                } else {
                    "df".to_string()
                };
                Ok(Some(Stmt::Load { path, alias }))
            }

            Token::Print => {
                self.advance();
                self.expect(&Token::LParen)?;
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(Some(Stmt::Print(expr)))
            }

            Token::Export => {
                self.advance();
                self.expect(&Token::LParen)?;
                let value = self.parse_expr()?;
                self.expect(&Token::Comma)?;
                let fmt = self.expect_ident()?;
                self.expect(&Token::Comma)?;
                let path = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(Some(Stmt::Export { value, fmt, path }))
            }

            Token::Generate => {
                self.advance();
                let df      = self.expect_ident()?;
                let varname = self.expect_ident()?;
                self.expect(&Token::Eq)?;
                let expr = self.parse_expr()?;
                Ok(Some(Stmt::Generate { df, varname, expr }))
            }

            Token::Predict => {
                self.advance();
                let df      = self.expect_ident()?;
                let varname = self.expect_ident()?;
                self.expect(&Token::Eq)?;
                let model = self.parse_primary()?;
                // opcional: , kind
                let kind = if self.peek() == &Token::Comma {
                    self.advance();
                    self.expect_ident()?
                } else {
                    "xb".to_string()
                };
                Ok(Some(Stmt::Predict { df, varname, model, kind }))
            }

            Token::Ident(_) => {
                let expr = self.parse_expr()?;
                Ok(Some(Stmt::Expr(expr)))
            }

            t => Err(HayashiError::Parse { line, msg: format!("unexpected token at statement level: {t:?}") }),
        }
    }

    pub fn parse_program(&mut self) -> Result<Vec<Stmt>> {
        let mut stmts = Vec::new();
        loop {
            match self.parse_stmt()? {
                None => break,
                Some(s) => stmts.push(s),
            }
        }
        Ok(stmts)
    }
}
