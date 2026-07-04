use crate::lang::ast::*;
use crate::lang::error::{HayashiError, Result};
use crate::lang::lexer::Token;

pub struct Parser {
    tokens: Vec<(Token, usize)>,
    pos: usize,
    paren_depth: usize,
    bracket_depth: usize, // rastreia [ ]
    brace_depth: usize,   // rastreia { }
}

impl Parser {
    pub fn new(tokens: Vec<(Token, usize)>) -> Self {
        Self {
            tokens,
            pos: 0,
            paren_depth: 0,
            bracket_depth: 0,
            brace_depth: 0,
        }
    }

    fn peek(&mut self) -> &Token {
        if self.paren_depth > 0 || self.bracket_depth > 0 || self.brace_depth > 0 {
            while self.tokens.get(self.pos).map(|(t, _)| t) == Some(&Token::Newline) {
                self.pos += 1;
            }
        }
        self.tokens
            .get(self.pos)
            .map(|(t, _)| t)
            .unwrap_or(&Token::Eof)
    }

    fn peek_raw_at(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset).map(|(t, _)| t)
    }

    fn line(&self) -> usize {
        self.tokens.get(self.pos).map(|(_, l)| *l).unwrap_or(0)
    }

    /// Verifica se o próximo token não-Newline é `|>` (para continuação de pipe multi-linha).
    fn next_non_newline_is_pipe_right(&self) -> bool {
        let mut i = self.pos;
        while let Some((Token::Newline, _)) = self.tokens.get(i) {
            i += 1;
        }
        matches!(self.tokens.get(i), Some((Token::PipeRight, _)))
    }

    fn advance(&mut self) -> &Token {
        if self.paren_depth > 0 || self.bracket_depth > 0 || self.brace_depth > 0 {
            while self.tokens.get(self.pos).map(|(t, _)| t) == Some(&Token::Newline) {
                self.pos += 1;
            }
        }
        let t = self
            .tokens
            .get(self.pos)
            .map(|(t, _)| t)
            .unwrap_or(&Token::Eof);
        if t == &Token::LParen {
            self.paren_depth += 1;
        }
        if t == &Token::RParen && self.paren_depth > 0 {
            self.paren_depth -= 1;
        }
        if t == &Token::LBracket {
            self.bracket_depth += 1;
        }
        if t == &Token::RBracket && self.bracket_depth > 0 {
            self.bracket_depth -= 1;
        }
        // Nota: LBrace/RBrace NÃO são rastreados aqui.
        // O brace_depth é gerenciado manualmente apenas no parse_dict_expr
        // para evitar suprimir Newlines dentro de blocos if/for/fn.
        self.pos += 1;
        t
    }

    fn expect_ident(&mut self) -> Result<String> {
        let line = self.line();
        match self.advance().clone() {
            Token::Ident(s) => Ok(s),
            t => Err(HayashiError::Parse {
                line,
                msg: format!("expected identifier, got {t:?}"),
            }),
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        let line = self.line();
        let got = self.advance().clone();
        if &got == expected {
            Ok(())
        } else {
            Err(HayashiError::Parse {
                line,
                msg: format!("expected {expected:?}, got {got:?}"),
            })
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
                Token::Pipe => {
                    self.advance();
                    in_fe = true;
                }
                Token::Plus | Token::Minus => {
                    self.advance();
                }
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
                                f => RhsTerm::Transform(f.to_string(), inner),
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
                _ => {
                    self.advance();
                }
            }
        }
        Ok(Formula { lhs, rhs, fe })
    }

    // ── Expressão aritmética (Pratt parsing) ────────────────────────────────
    //
    // Precedência (do menor para o maior):
    //   or  ||
    //   and  &&
    //   comparison  > < >= <= == !=
    //   additive    + -
    //   multiplicative  * /
    //   power       ^
    //   unary       - !
    //   primary

    pub fn parse_expr(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_or()?;

        // Range operators: a..b (exclusivo) ou a..=b (inclusivo)
        if self.peek() == &Token::DotDotEq {
            self.advance();
            let rhs = self.parse_or()?;
            return Ok(Expr::RangeInclusive(Box::new(lhs), Box::new(rhs)));
        }
        if self.peek() == &Token::DotDot {
            self.advance();
            let rhs = self.parse_or()?;
            return Ok(Expr::Range(Box::new(lhs), Box::new(rhs)));
        }

        // Verifica se o próximo token não-Newline é |> para permitir pipe multi-linha
        if self.peek() != &Token::PipeRight && !self.next_non_newline_is_pipe_right() {
            return Ok(lhs);
        }
        let source = lhs.clone();
        while self.peek() == &Token::PipeRight || self.next_non_newline_is_pipe_right() {
            self.skip_newlines(); // consome Newlines antes do |>
            self.advance();
            let rhs = self.parse_or()?;
            lhs = match rhs {
                Expr::Call {
                    func,
                    mut args,
                    opts,
                } => {
                    let mut found = false;
                    for arg in args.iter_mut() {
                        if let Expr::Var(ref name) = arg {
                            if name == "_" {
                                *arg = lhs.clone();
                                found = true;
                            }
                        }
                    }
                    if !found {
                        args.insert(0, lhs);
                    }
                    Expr::Call { func, args, opts }
                }
                Expr::Var(name) => Expr::Call {
                    func: name,
                    args: vec![lhs],
                    opts: vec![],
                },
                closure @ Expr::Closure { .. } => Expr::Apply {
                    func: Box::new(closure),
                    args: vec![lhs],
                },
                _ => {
                    return Err(HayashiError::Parse {
                        line: self.line(),
                        msg: "|> right side must be a function call or closure".into(),
                    })
                }
            };
        }
        Ok(Expr::Pipe {
            source: Box::new(source),
            expr: Box::new(lhs),
        })
    }

    fn parse_or(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_and()?;
        while self.peek() == &Token::Or {
            self.advance();
            let rhs = self.parse_and()?;
            lhs = Expr::BinOp {
                op: BinOp::Or,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_comparison()?;
        while self.peek() == &Token::And {
            self.advance();
            let rhs = self.parse_comparison()?;
            lhs = Expr::BinOp {
                op: BinOp::And,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_additive()?;
        loop {
            let op = match self.peek() {
                Token::Gt => BinOp::Gt,
                Token::Lt => BinOp::Lt,
                Token::GtEq => BinOp::GtEq,
                Token::LtEq => BinOp::LtEq,
                Token::EqEq => BinOp::Eq,
                Token::BangEq => BinOp::Ne,
                Token::In => BinOp::In,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_additive()?;
            lhs = Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_multiplicative()?;
            lhs = Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_power()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_power()?;
            lhs = Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_power(&mut self) -> Result<Expr> {
        let base = self.parse_unary()?;
        if matches!(self.peek(), Token::Caret | Token::StarStar) {
            self.advance();
            let exp = self.parse_unary()?; // right-associative
            Ok(Expr::BinOp {
                op: BinOp::Pow,
                lhs: Box::new(base),
                rhs: Box::new(exp),
            })
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        match self.peek() {
            Token::Minus => {
                self.advance();
                let inner = self.parse_primary()?;
                Ok(Expr::Neg(Box::new(inner)))
            }
            Token::Bang => {
                self.advance();
                let inner = self.parse_primary()?;
                Ok(Expr::Not(Box::new(inner)))
            }
            _ => self.parse_postfix(),
        }
    }

    // Postfix: lida com v[idx] após parse_primary
    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.peek() == &Token::LBracket {
                self.advance();
                let idx = self.parse_expr()?;
                self.expect(&Token::RBracket)?;
                expr = Expr::Index {
                    obj: Box::new(expr),
                    idx: Box::new(idx),
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        let line = self.line();
        match self.peek().clone() {
            Token::Float(f) => {
                self.advance();
                Ok(Expr::Float(f))
            }
            Token::Int(i) => {
                self.advance();
                Ok(Expr::Int(i))
            }
            Token::Bool(b) => {
                self.advance();
                Ok(Expr::Bool(b))
            }
            Token::StringLit(s) => {
                self.advance();
                Ok(Expr::Str(s))
            }
            Token::Nil => {
                self.advance();
                Ok(Expr::Nil)
            }

            // if cond { then_expr } else { else_expr }  (expression)
            Token::If => {
                self.advance();
                let cond = self.parse_expr()?;
                self.expect(&Token::LBrace)?;
                let then_expr = self.parse_expr()?;
                self.expect(&Token::RBrace)?;
                if self.peek() != &Token::Else {
                    return Err(HayashiError::Parse {
                        line,
                        msg: "if-expression requires else branch".into(),
                    });
                }
                self.advance();
                self.expect(&Token::LBrace)?;
                let else_expr = self.parse_expr()?;
                self.expect(&Token::RBrace)?;
                Ok(Expr::If {
                    cond: Box::new(cond),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                })
            }
            Token::FStringLit(s) => {
                self.advance();
                Ok(Expr::FString(s))
            }

            // Agrupamento: (expr)
            Token::LParen => {
                self.advance();
                let inner = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }

            // Lista literal: [e1, e2, ...] — permite quebras de linha entre elementos
            Token::LBracket => {
                self.advance(); // avança LBracket (incrementa bracket_depth)
                self.skip_newlines();
                let mut items = Vec::new();
                // bracket_depth > 0 agora: peek() já pula Newlines automaticamente
                while !matches!(self.peek(), Token::RBracket | Token::Eof) {
                    items.push(self.parse_expr()?);
                    if self.peek() == &Token::Comma {
                        self.advance();
                        self.skip_newlines();
                    } else {
                        break;
                    }
                }
                self.expect(&Token::RBracket)?; // decrementa bracket_depth
                Ok(Expr::List(items))
            }

            // Dict literal: {"key": value, ...}  ou bloco expressão: { stmt; ...; expr }
            Token::LBrace => {
                if self.is_dict_literal() {
                    self.advance(); // consome LBrace
                                    // Incrementa manualmente: dentro do dict, Newlines são ignorados
                    self.brace_depth += 1;
                    let mut pairs = Vec::new();
                    while !matches!(self.peek(), Token::RBrace | Token::Eof) {
                        let key = self.parse_expr()?;
                        self.expect(&Token::Colon)?;
                        let val = self.parse_expr()?;
                        pairs.push((key, val));
                        if self.peek() == &Token::Comma {
                            self.advance();
                        }
                    }
                    self.brace_depth -= 1; // decrementa antes do RBrace
                    self.expect(&Token::RBrace)?;
                    Ok(Expr::Dict(pairs))
                } else {
                    self.parse_block_expr()
                }
            }

            // Fórmula sem LHS: ~ z1 + z2
            Token::Tilde => {
                let formula = self.parse_formula(String::new())?;
                Ok(Expr::Formula(formula))
            }

            // Match expression: match expr { pat => result, ... }
            Token::Ident(ref s) if s == "match" => {
                self.advance();
                let scrutinee = self.parse_expr()?;
                self.expect(&Token::LBrace)?;
                self.skip_newlines();
                let mut arms = Vec::new();
                while !matches!(self.peek(), Token::RBrace | Token::Eof) {
                    let pattern = self.parse_expr()?;
                    self.expect(&Token::FatArrow)?;
                    let result = self.parse_expr()?;
                    arms.push((pattern, result));
                    if self.peek() == &Token::Comma {
                        self.advance();
                    }
                    self.skip_newlines();
                }
                self.expect(&Token::RBrace)?;
                Ok(Expr::Match {
                    expr: Box::new(scrutinee),
                    arms,
                })
            }

            Token::Ident(name) => {
                self.advance();

                if self.peek() == &Token::ColonColon {
                    self.advance();
                    let member = self.expect_ident()?;
                    let qualified = format!("{name}::{member}");
                    if self.peek() == &Token::LParen {
                        self.advance();
                        let (args, opts) = self.parse_call_args()?;
                        self.expect(&Token::RParen)?;
                        return Ok(Expr::Call {
                            func: qualified,
                            args,
                            opts,
                        });
                    }
                    return Ok(Expr::Var(qualified));
                }

                if self.peek() == &Token::Tilde {
                    let formula = self.parse_formula(name)?;
                    return Ok(Expr::Formula(formula));
                }

                if self.peek() == &Token::LParen {
                    self.advance();
                    let (args, opts) = self.parse_call_args()?;
                    self.expect(&Token::RParen)?;
                    let mut expr = Expr::Call {
                        func: name,
                        args,
                        opts,
                    };

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
                        expr = Expr::Field {
                            obj: Box::new(expr),
                            field,
                            args: fargs,
                            opts: fopts,
                        };
                    }
                    return Ok(expr);
                }

                Ok(Expr::Var(name))
            }

            // Operadores de série temporal: L.price, L2.price, F.gdp, D.wage
            Token::TsLag(n) => {
                self.advance();
                let var = self.expect_ident()?;
                Ok(Expr::TsOp {
                    op: TsOpKind::Lag,
                    var,
                    n,
                })
            }
            Token::TsLead(n) => {
                self.advance();
                let var = self.expect_ident()?;
                Ok(Expr::TsOp {
                    op: TsOpKind::Lead,
                    var,
                    n,
                })
            }
            Token::TsDiff(n) => {
                self.advance();
                let var = self.expect_ident()?;
                Ok(Expr::TsOp {
                    op: TsOpKind::Diff,
                    var,
                    n,
                })
            }

            // Closure: |x, y| expr
            Token::Pipe => {
                self.advance();
                let mut params = Vec::new();
                while !matches!(self.peek(), Token::Pipe | Token::Eof) {
                    params.push(self.expect_ident()?);
                    if self.peek() == &Token::Comma {
                        self.advance();
                    }
                }
                self.expect(&Token::Pipe)?;
                let body = self.parse_expr()?;
                Ok(Expr::Closure {
                    params,
                    body: Box::new(body),
                })
            }

            // Keywords usadas como identificadores em contexto de expressão
            Token::Count
            | Token::Replace
            | Token::Load
            | Token::Export
            | Token::Print
            | Token::Predict
            | Token::Generate
            | Token::Return
            | Token::Break
            | Token::Continue
            | Token::In => {
                let name = match self.peek() {
                    Token::Count => "count",
                    Token::Replace => "replace",
                    Token::Load => "load",
                    Token::Export => "export",
                    Token::Print => "print",
                    Token::Predict => "predict",
                    Token::Generate => "generate",
                    Token::Return => "return",
                    Token::Break => "break",
                    Token::Continue => "continue",
                    Token::In => "in",
                    _ => "?",
                }
                .to_string();
                self.advance();
                if self.peek() == &Token::LParen {
                    self.advance();
                    let (args, opts) = self.parse_call_args()?;
                    self.expect(&Token::RParen)?;
                    Ok(Expr::Call {
                        func: name,
                        args,
                        opts,
                    })
                } else {
                    Ok(Expr::Var(name))
                }
            }

            _ => Err(HayashiError::Parse {
                line,
                msg: format!("unexpected token {:?}", self.peek()),
            }),
        }
    }

    fn is_kw_bare_arg(&mut self) -> bool {
        // Palavras-chave que NÃO podem ser identificadores isolados em
        // expressões (help(if), help(for), etc.). Outras como count/load/return
        // já são tratadas como Expr::Var pelo parse_primary.
        let is_kw = matches!(
            self.peek(),
            Token::If
                | Token::Else
                | Token::For
                | Token::While
                | Token::Fn
                | Token::Let
                | Token::Tsset
        );
        let next_is_terminator = self
            .tokens
            .get(self.pos + 1)
            .map(|(t, _)| matches!(t, Token::RParen | Token::Comma))
            .unwrap_or(false);
        is_kw && next_is_terminator
    }

    fn parse_call_args(&mut self) -> Result<(Vec<Expr>, Vec<Opt>)> {
        let mut args = Vec::new();
        let mut opts = Vec::new();

        while !matches!(self.peek(), Token::RParen | Token::Eof | Token::Newline) {
            // opt=value  ou  expr normal
            // Caso especial: keyword `if` usada como chave de opção (ex: mean(df, y, if=x==1))
            let is_kw_opt = matches!(
                self.peek(),
                Token::If
                    | Token::Else
                    | Token::Generate
                    | Token::For
                    | Token::In
                    | Token::Return
                    | Token::Break
                    | Token::Continue
                    | Token::Count
                    | Token::Replace
                    | Token::Load
                    | Token::Export
                    | Token::Print
                    | Token::Predict
            ) && self
                .tokens
                .get(self.pos + 1)
                .map(|(t, _)| t == &Token::Eq)
                .unwrap_or(false);
            if is_kw_opt {
                let kw_name = match self.peek() {
                    Token::If => "if",
                    Token::Else => "else",
                    Token::Generate => "gen",
                    Token::For => "for",
                    Token::In => "in",
                    Token::Return => "return",
                    Token::Break => "break",
                    Token::Continue => "continue",
                    Token::Count => "count",
                    Token::Replace => "replace",
                    Token::Load => "load",
                    Token::Export => "export",
                    Token::Print => "print",
                    Token::Predict => "predict",
                    _ => "?",
                }
                .to_string();
                self.advance(); // keyword
                self.advance(); // =
                let val = self.parse_expr()?;
                opts.push(Opt {
                    name: kw_name,
                    value: val,
                });
            } else if self.is_kw_bare_arg() {
                // Palavra-chave usada como argumento isolado (ex: help(if), help(for))
                let kw_name = match self.peek() {
                    Token::If => "if",
                    Token::Else => "else",
                    Token::For => "for",
                    Token::While => "while",
                    Token::Fn => "fn",
                    Token::Let => "let",
                    Token::Tsset => "tsset",
                    _ => "?",
                }
                .to_string();
                self.advance();
                args.push(Expr::Str(kw_name));
            } else if let Token::Ident(name) = self.peek().clone() {
                // lookahead: é opt=val?
                if self
                    .tokens
                    .get(self.pos + 1)
                    .map(|(t, _)| t == &Token::Eq)
                    .unwrap_or(false)
                {
                    self.advance(); // nome
                    self.advance(); // =
                    let val = self.parse_expr()?;
                    opts.push(Opt { name, value: val });
                } else {
                    args.push(self.parse_expr()?);
                }
            } else {
                args.push(self.parse_expr()?);
            }

            if self.peek() == &Token::Comma {
                self.advance();
            }
        }
        Ok((args, opts))
    }

    // ── Bloco { stmt* } ───────────────────────────────────────────────────────

    fn parse_block(&mut self) -> Result<Vec<Spanned>> {
        self.expect(&Token::LBrace)?;
        self.skip_newlines();
        let mut stmts = Vec::new();
        while !matches!(self.peek(), Token::RBrace | Token::Eof) {
            let line = self.line();
            if let Some(s) = self.parse_stmt()? {
                stmts.push((s, line));
            }
            self.skip_newlines();
        }
        self.expect(&Token::RBrace)?;
        Ok(stmts)
    }

    // ── Bloco expressão: { stmt; ...; expr } ───────────────────────────────────

    fn is_dict_literal(&mut self) -> bool {
        // Olha à frente do LBrace atual: {"key": ...} começa com StringLit seguido de ':'.
        matches!(self.tokens.get(self.pos + 1).map(|t| &t.0), Some(Token::StringLit(_)))
            && self.tokens.get(self.pos + 2).map(|t| &t.0) == Some(&Token::Colon)
    }

    fn parse_block_expr(&mut self) -> Result<Expr> {
        self.expect(&Token::LBrace)?;
        self.brace_depth += 1;
        self.skip_newlines();
        let mut stmts = Vec::new();
        while !matches!(self.peek(), Token::RBrace | Token::Eof) {
            if let Some(s) = self.parse_stmt()? {
                stmts.push(s);
            }
            self.skip_newlines();
        }
        self.brace_depth -= 1;
        self.expect(&Token::RBrace)?;
        // Se a última statement for uma expressão, ela é o valor de retorno do bloco.
        let final_expr = if let Some(Stmt::Expr(e)) = stmts.last() {
            let e = e.clone();
            stmts.pop();
            Some(Box::new(e))
        } else {
            None
        };
        Ok(Expr::Block(stmts, final_expr))
    }

    // ── Iterador do for ────────────────────────────────────────────────────────
    // Aceita:  start..end   (Range)   ou   expr   (Items — lista/var)

    fn parse_for_iter(&mut self) -> Result<ForIter> {
        match self.parse_expr()? {
            Expr::Range(start, end) => Ok(ForIter::Range(*start, *end)),
            Expr::RangeInclusive(start, end) => Ok(ForIter::RangeInclusive(*start, *end)),
            other => Ok(ForIter::Items(other)),
        }
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

            Token::Ident(ref s) if s == "const" => {
                self.advance();
                let name = self.expect_ident()?;
                self.expect(&Token::Eq)?;
                let value = self.parse_expr()?;
                Ok(Some(Stmt::Const { name, value }))
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
                // opções: , key=value, ...
                let mut opts = Vec::new();
                while *self.peek() == Token::Comma {
                    self.advance();
                    let key = self.expect_ident()?;
                    self.expect(&Token::Eq)?;
                    let val = match key.as_str() {
                        "sheet" | "table" => {
                            if let Token::Ident(s) = self.peek().clone() {
                                self.advance();
                                Expr::Str(s)
                            } else {
                                self.parse_expr()?
                            }
                        }
                        _ => self.parse_expr()?,
                    };
                    opts.push(Opt {
                        name: key,
                        value: val,
                    });
                }
                Ok(Some(Stmt::Load { path, alias, opts }))
            }

            Token::Print => {
                self.advance();
                self.expect(&Token::LParen)?;
                let mut exprs = Vec::new();
                let mut opts = Vec::new();
                if self.peek() != &Token::RParen {
                    loop {
                        if let Some(Token::Ident(name)) = self.peek_raw_at(0).cloned() {
                            if self.peek_raw_at(1) == Some(&Token::Eq) {
                                self.advance();
                                self.advance();
                                let val = self.parse_expr()?;
                                opts.push(Opt { name, value: val });
                                if self.peek() == &Token::Comma {
                                    self.advance();
                                    continue;
                                }
                                break;
                            }
                        }
                        exprs.push(self.parse_expr()?);
                        if self.peek() == &Token::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Token::RParen)?;
                Ok(Some(Stmt::Print(exprs, opts)))
            }

            Token::Export => {
                self.advance();
                self.expect(&Token::LParen)?;
                let value = self.parse_expr()?;
                self.expect(&Token::Comma)?;
                let fmt = self.parse_expr()?;
                self.expect(&Token::Comma)?;
                let path = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(Some(Stmt::Export { value, fmt, path }))
            }

            Token::Generate => {
                if self.peek_raw_at(1) == Some(&Token::LParen) {
                    // generate(df, col = expr) — function call form
                    let expr = self.parse_expr()?;
                    Ok(Some(Stmt::Expr(expr)))
                } else {
                    // generate df var = expr — Stata statement form
                    self.advance();
                    let df = self.expect_ident()?;
                    let varname = self.expect_ident()?;
                    self.expect(&Token::Eq)?;
                    let expr = self.parse_expr()?;
                    Ok(Some(Stmt::Generate { df, varname, expr }))
                }
            }

            Token::Predict => {
                self.advance();
                let df = self.expect_ident()?;
                let varname = self.expect_ident()?;
                self.expect(&Token::Eq)?;
                let model = self.parse_primary()?;
                let kind = if self.peek() == &Token::Comma {
                    self.advance();
                    self.parse_expr()?
                } else {
                    Expr::Str("xb".to_string())
                };
                Ok(Some(Stmt::Predict {
                    df,
                    varname,
                    model,
                    kind,
                }))
            }

            Token::Count => {
                self.advance();
                let df = self.expect_ident()?;
                let cond = if self.peek() == &Token::If {
                    self.advance();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                Ok(Some(Stmt::Count { df, cond }))
            }

            Token::Replace => {
                self.advance();
                let df = self.expect_ident()?;
                let varname = self.expect_ident()?;
                self.expect(&Token::Eq)?;
                let expr = self.parse_expr()?;
                // opcional: if cond_expr
                let cond = if self.peek() == &Token::If {
                    self.advance();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                Ok(Some(Stmt::Replace {
                    df,
                    varname,
                    expr,
                    cond,
                }))
            }

            Token::Tsset => {
                self.advance();
                let df = self.expect_ident()?;
                let t_var = self.expect_ident()?;
                Ok(Some(Stmt::Tsset { df, t_var }))
            }

            // ── if cond { ... } [else [if cond] { ... }] ─────────────────────
            Token::If => {
                self.advance();
                let cond = self.parse_expr()?;
                let then_body = self.parse_block()?;
                // else [if ...]
                let else_body = if self.peek() == &Token::Else {
                    self.advance();
                    if self.peek() == &Token::If {
                        let inner_line = self.line();
                        let inner = self.parse_stmt()?.ok_or_else(|| HayashiError::Parse {
                            line,
                            msg: "expected statement after 'else if'".into(),
                        })?;
                        Some(vec![(inner, inner_line)])
                    } else {
                        Some(self.parse_block()?)
                    }
                } else {
                    None
                };
                Ok(Some(Stmt::If {
                    cond,
                    then_body,
                    else_body,
                }))
            }

            // ── for var in iter { ... } ───────────────────────────────────────
            Token::For => {
                self.advance();
                let var = self.expect_ident()?;
                // espera "in"
                match self.advance().clone() {
                    Token::In => {}
                    t => {
                        return Err(HayashiError::Parse {
                            line,
                            msg: format!("expected 'in' after variável do for, got {t:?}"),
                        })
                    }
                }
                let iter = self.parse_for_iter()?;
                let body = self.parse_block()?;
                Ok(Some(Stmt::For { var, iter, body }))
            }

            // ── fn nome(p1, p2) { corpo } ─────────────────────────────────────
            Token::Fn => {
                self.advance();
                let name = self.expect_ident()?;
                self.expect(&Token::LParen)?;
                let mut params = Vec::new();
                while !matches!(self.peek(), Token::RParen | Token::Eof) {
                    params.push(self.expect_ident()?);
                    if self.peek() == &Token::Comma {
                        self.advance();
                    }
                }
                self.expect(&Token::RParen)?;
                let body = self.parse_block()?;
                Ok(Some(Stmt::Fn { name, params, body }))
            }

            // ── return [expr] ─────────────────────────────────────────────────
            Token::Return => {
                self.advance();
                let expr = if matches!(self.peek(), Token::Newline | Token::RBrace | Token::Eof) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                Ok(Some(Stmt::Return(expr)))
            }

            Token::Break => {
                self.advance();
                Ok(Some(Stmt::Break))
            }
            Token::Continue => {
                self.advance();
                Ok(Some(Stmt::Continue))
            }

            // ── while cond { ... } ────────────────────────────────────────────
            Token::While => {
                self.advance();
                let cond = self.parse_expr()?;
                let body = self.parse_block()?;
                Ok(Some(Stmt::While { cond, body }))
            }

            // ── input df \n header_row \n data_rows \n end ────────────────────
            Token::Ident(ref s) if s == "input" => {
                self.advance();
                let alias = self.expect_ident()?;
                self.skip_newlines();

                // Header: nomes das variáveis até newline
                let mut headers: Vec<String> = Vec::new();
                loop {
                    match self.peek().clone() {
                        Token::Newline | Token::Eof => break,
                        Token::Ident(h) => {
                            let h = h.clone();
                            self.advance();
                            headers.push(h);
                        }
                        _ => break,
                    }
                }
                self.skip_newlines();

                // Linhas de dados até "end"
                let mut rows: Vec<Vec<f64>> = Vec::new();
                'outer: loop {
                    self.skip_newlines();
                    // Detectar "end"
                    if let Token::Ident(ref s) = self.peek().clone() {
                        if s == "end" {
                            self.advance();
                            break 'outer;
                        }
                    }
                    if self.peek() == &Token::Eof {
                        break;
                    }

                    let mut row: Vec<f64> = Vec::new();
                    loop {
                        match self.peek().clone() {
                            Token::Newline | Token::Eof => break,
                            Token::Float(v) => {
                                self.advance();
                                row.push(v);
                            }
                            Token::Int(v) => {
                                let v = v as f64;
                                self.advance();
                                row.push(v);
                            }
                            Token::Minus => {
                                self.advance();
                                let v = match self.peek().clone() {
                                    Token::Float(v) => {
                                        self.advance();
                                        -v
                                    }
                                    Token::Int(v) => {
                                        self.advance();
                                        -(v as f64)
                                    }
                                    _ => {
                                        return Err(HayashiError::Parse {
                                            line,
                                            msg: "esperado número após '-'".into(),
                                        })
                                    }
                                };
                                row.push(v);
                            }
                            Token::Dot => {
                                self.advance();
                                row.push(f64::NAN);
                            } // . = missing
                            Token::Ident(ref s) | Token::StringLit(ref s) => {
                                let s = s.clone();
                                return Err(HayashiError::Parse {
                                    line,
                                    msg: format!(
                                        "input block só aceita valores numéricos — \
                                         '{s}' não é um número. \
                                         Use '.' para missing, ou load para arquivos com colunas de texto."
                                    ),
                                });
                            }
                            _ => {
                                // pular tokens desconhecidos até fim da linha
                                while !matches!(self.peek(), Token::Newline | Token::Eof) {
                                    self.advance();
                                }
                                break;
                            }
                        }
                    }
                    if !row.is_empty() {
                        rows.push(row);
                    }
                }
                Ok(Some(Stmt::Input {
                    alias,
                    headers,
                    rows,
                }))
            }

            // ── try { ... } catch e { ... } ──────────────────────────────────
            Token::Ident(ref s) if s == "try" => {
                self.advance();
                let try_body = self.parse_block()?;
                let catch_kw = match self.peek().clone() {
                    Token::Ident(s) if s == "catch" => {
                        self.advance();
                        true
                    }
                    _ => false,
                };
                if !catch_kw {
                    return Err(HayashiError::Parse {
                        line,
                        msg: "expected 'catch' after try block".into(),
                    });
                }
                let error_var = self.expect_ident()?;
                let catch_body = self.parse_block()?;
                Ok(Some(Stmt::TryCatch {
                    try_body,
                    error_var,
                    catch_body,
                }))
            }

            // ── display expr  (sem parênteses) ───────────────────────────────
            Token::Ident(ref s) if s == "display" || s == "di" => {
                self.advance();
                let expr = self.parse_expr()?;
                Ok(Some(Stmt::Display(expr)))
            }

            // ── scalar name = expr  (alias de let) ───────────────────────────
            Token::Ident(ref s) if s == "scalar" => {
                self.advance();
                let name = self.expect_ident()?;
                self.expect(&Token::Eq)?;
                let value = self.parse_expr()?;
                Ok(Some(Stmt::Let { name, value }))
            }

            // nome = expr (assignment sem let — modifica variável existente)
            Token::Ident(ref name)
                if self
                    .tokens
                    .get(self.pos + 1)
                    .map(|(t, _)| t == &Token::Eq)
                    .unwrap_or(false) =>
            {
                let name = name.clone();
                self.advance(); // ident
                self.advance(); // =
                let value = self.parse_expr()?;
                Ok(Some(Stmt::Assign { name, value }))
            }

            // compound assignment: +=  -=  *=  /=  %=
            Token::Ident(ref name)
                if self
                    .tokens
                    .get(self.pos + 1)
                    .map(|(t, _)| {
                        matches!(
                            t,
                            Token::PlusEq
                                | Token::MinusEq
                                | Token::StarEq
                                | Token::SlashEq
                                | Token::PercentEq
                        )
                    })
                    .unwrap_or(false) =>
            {
                let name = name.clone();
                self.advance(); // ident
                let op = match self.peek() {
                    Token::PlusEq => BinOp::Add,
                    Token::MinusEq => BinOp::Sub,
                    Token::StarEq => BinOp::Mul,
                    Token::SlashEq => BinOp::Div,
                    Token::PercentEq => BinOp::Mod,
                    _ => unreachable!(),
                };
                self.advance(); // op=
                let rhs = self.parse_expr()?;
                let value = Expr::BinOp {
                    op,
                    lhs: Box::new(Expr::Var(name.clone())),
                    rhs: Box::new(rhs),
                };
                Ok(Some(Stmt::Assign { name, value }))
            }

            Token::Ident(_) => {
                let expr = self.parse_expr()?;
                Ok(Some(Stmt::Expr(expr)))
            }

            Token::LBrace => {
                let body = self.parse_block()?;
                Ok(Some(Stmt::Block(body)))
            }

            Token::Int(_)
            | Token::Float(_)
            | Token::Bool(_)
            | Token::StringLit(_)
            | Token::FStringLit(_)
            | Token::LBracket
            | Token::LParen
            | Token::Minus
            | Token::Bang
            | Token::Pipe => {
                let expr = self.parse_expr()?;
                Ok(Some(Stmt::Expr(expr)))
            }

            t => Err(HayashiError::Parse {
                line,
                msg: format!("unexpected token at statement level: {t:?}"),
            }),
        }
    }

    pub fn parse_program(&mut self) -> Result<Vec<Spanned>> {
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            let line = self.line();
            match self.parse_stmt()? {
                None => break,
                Some(s) => stmts.push((s, line)),
            }
        }
        Ok(stmts)
    }
}
