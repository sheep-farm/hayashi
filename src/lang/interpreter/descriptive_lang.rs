use super::*;

/// set_seed, timer, quietly, capture, assert, preserve/restore, source, help,
/// describe, codebook, format, duplicates, label, correlate, summarize,
/// esttab/eststo. Extraído de `eval_call` (ver src/lang/interpreter.rs).
impl Interpreter {
    pub(super) fn eval_call_descriptive_lang(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        opt_map: &HashMap<String, Value>,
    ) -> Result<Option<Value>> {
        let result: Result<Value> = match func {
            // ── set_seed: reprodutibilidade ────────────────────────────────
            "set_seed" | "seed" | "setseed" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "set_seed(N) — define semente do RNG".into(),
                    ));
                }
                let s = match self.eval_expr(&args[0])? {
                    Value::Int(v) => v as u64,
                    Value::Float(v) => v as u64,
                    _ => return Err(HayashiError::Type("seed must be integer".into())),
                };
                self.rng_seed = Some(s);
                use rand::SeedableRng;
                self.rng = rand::rngs::StdRng::seed_from_u64(s);
                println!("set seed {s}");
                Ok(Value::Nil)
            }

            // ── timer: mede tempo de execução ─────────────────────────────
            "timer" | "time" | "bench" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "timer(expr) — mede tempo de avaliação".into(),
                    ));
                }
                let start = std::time::Instant::now();
                let result = self.eval_expr(&args[0])?;
                let elapsed = start.elapsed();
                println!("  elapsed: {:.4}s", elapsed.as_secs_f64());
                Ok(result)
            }

            // ── quietly: avalia expressão, suprime saída ──────────────────
            "quietly" | "quiet" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "quietly(expr) — avalia sem imprimir".into(),
                    ));
                }
                self.eval_expr(&args[0])?;
                Ok(Value::Nil)
            }

            // ── capture: avalia expressão, ignora erros ───────────────────
            "capture" | "cap" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "capture(expr) — avalia ignorando erros".into(),
                    ));
                }
                match self.eval_expr(&args[0]) {
                    Ok(v) => Ok(v),
                    Err(e) => {
                        eprintln!("(captured: {e})");
                        Ok(Value::Nil)
                    }
                }
            }

            // ── assert: erro se condição é falsa ──────────────────────────
            "assert" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "assert(cond [, msg]) — erro se condição falsa".into(),
                    ));
                }
                let val = self.eval_expr(&args[0])?;
                if !Self::value_as_bool(&val) {
                    let msg = if args.len() >= 2 {
                        match self.eval_expr(&args[1])? {
                            Value::Str(s) => s,
                            _ => "assertion failed".into(),
                        }
                    } else {
                        "assertion failed".into()
                    };
                    return Err(HayashiError::Runtime(msg));
                }
                Ok(Value::Nil)
            }

            // ── preserve/restore: salvar e restaurar estado de variáveis ───
            "preserve" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "preserve(df) — salva cópia do DataFrame".into(),
                    ));
                }
                let name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "preserve() requires a variable name".into(),
                        ))
                    }
                };
                let val = self
                    .env
                    .get(&name)
                    .ok_or_else(|| self.rt_err(format!("'{name}' not found")))?
                    .clone();
                self.preserved.insert(name.clone(), val);
                println!("preserve {name}");
                Ok(Value::Nil)
            }

            "restore" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "restore(df) — restaura DataFrame salvo".into(),
                    ));
                }
                let name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "restore() requires a variable name".into(),
                        ))
                    }
                };
                let val = self
                    .preserved
                    .remove(&name)
                    .ok_or_else(|| self.rt_err(format!("'{name}' was not preserved")))?;
                self.env.set(&name, val)?;
                println!("restore {name}");
                Ok(Value::Nil)
            }

            // ── source/do: executa script .hay no ambiente atual ─────────────
            "source" | "do" | "run" | "include" => {
                if args.is_empty() {
                    return Err(self.rt_err("source(\"script.hay\")"));
                }
                let path = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("source() requires a string path".into())),
                };
                let src = std::fs::read_to_string(&path)
                    .map_err(|e| self.rt_err(format!("cannot read '{path}': {e}")))?;
                println!("source {path}");
                crate::lang::run_source(&src, self)?;
                Ok(Value::Nil)
            }

            "import" | "require" => {
                if args.is_empty() {
                    return Err(self.rt_err("import(\"module_or_url\")"));
                }
                let module = match self.eval_expr(&args[0])? {
                    Value::Str(s) => s,
                    _ => return Err(HayashiError::Type("import() requires a string".into())),
                };

                if self.imported.contains(&module) {
                    return Ok(Some(Value::Nil));
                }

                let resolved = if crate::io::fetch::is_url(&module) {
                    let tmp = crate::io::fetch::download_to_temp(&module)?;
                    tmp.to_string_lossy().to_string()
                } else {
                    self.resolve_import(&module)?
                };

                let alias = match opt_map.get("as") {
                    Some(Value::Str(s)) => Some(s.clone()),
                    _ => None,
                };
                let only: Option<Vec<String>> = match opt_map.get("only") {
                    Some(Value::List(lst)) => Some(
                        lst.iter()
                            .filter_map(|v| match v {
                                Value::Str(s) => Some(s.clone()),
                                _ => None,
                            })
                            .collect(),
                    ),
                    _ => None,
                };

                let ns = alias.clone().unwrap_or_else(|| {
                    let base = module
                        .trim_end_matches(".hay")
                        .trim_end_matches(".wasm")
                        .trim_end_matches(".so")
                        .trim_end_matches(".dll")
                        .trim_end_matches(".dylib");
                    base.rsplit('/').next().unwrap_or(&module).to_string()
                });

                let is_wasm = resolved.ends_with(".wasm");
                let is_native = resolved.ends_with(".so")
                    || resolved.ends_with(".dll")
                    || resolved.ends_with(".dylib");

                if is_wasm {
                    #[cfg(not(feature = "wasm"))]
                    return Err(self.rt_err("import: WASM plugins require 'wasm' feature"));
                    #[cfg(feature = "wasm")]
                    {
                        use crate::lang::plugin::WasmPlugin;
                        let plugin = WasmPlugin::new(&resolved, &ns).map_err(|e| {
                            self.rt_err(format!("import: failed to load WASM plugin: {e}"))
                        })?;
                        self.plugins.insert(ns.clone(), Box::new(plugin));
                        self.imported.insert(module.clone());
                        return Ok(Some(Value::Nil));
                    }
                } else if is_native {
                    use crate::lang::plugin::RustNativePlugin;
                    let plugin = RustNativePlugin::new(&resolved, &ns).map_err(|e| {
                        self.rt_err(format!("import: failed to load native plugin: {e}"))
                    })?;
                    self.plugins.insert(ns.clone(), Box::new(plugin));
                    self.imported.insert(module.clone());
                    return Ok(Some(Value::Nil));
                }

                // Default script plugin (.hay) loading
                let src = std::fs::read_to_string(&resolved)
                    .map_err(|e| self.rt_err(format!("import: cannot read '{resolved}': {e}")))?;

                self.imported.insert(module.clone());

                let before: std::collections::HashSet<String> =
                    self.env.var_names().into_iter().collect();

                crate::lang::run_source(&src, self)?;

                let new_names: Vec<String> = self
                    .env
                    .var_names()
                    .into_iter()
                    .filter(|n| !before.contains(n))
                    .collect();

                if let Some(ref allowed) = only {
                    for name in &new_names {
                        if !allowed.contains(name) {
                            self.env.remove(name);
                        }
                    }
                } else {
                    for name in &new_names {
                        if let Some(val) = self.env.get(name).cloned() {
                            let qualified = format!("{ns}::{name}");
                            self.env.declare(&qualified, val).ok();
                            self.env.remove(name);
                        }
                    }
                }

                Ok(Value::Nil)
            }

            "plugin_path" => {
                if args.is_empty() {
                    if self.plugin_paths.is_empty() {
                        println!("plugin_path: (none)");
                    } else {
                        for p in &self.plugin_paths {
                            println!("  {p}");
                        }
                    }
                    return Ok(Some(Value::Nil));
                }
                for arg in args {
                    let path = match self.eval_expr(arg)? {
                        Value::Str(s) => s,
                        other => {
                            return Err(
                                self.type_err(format!("plugin_path: expected string, got {other}"))
                            )
                        }
                    };
                    if !self.plugin_paths.contains(&path) {
                        self.plugin_paths.push(path);
                    }
                }
                Ok(Value::Nil)
            }

            // ── help: sistema de ajuda inline ──────────────────────────────
            "help" => {
                let topic = if args.is_empty() {
                    String::new()
                } else {
                    match &args[0] {
                        Expr::Var(n) | Expr::Str(n) => n.clone(),
                        _ => String::new(),
                    }
                };
                if topic == "about" {
                    println!("{}", crate::lang::help::help_about());
                } else if topic == "license" {
                    println!("{}", crate::lang::help::help_license());
                } else {
                    match crate::lang::help::help_text(&topic) {
                        Some(h) => println!("{h}"),
                        None => {
                            if let Some(Value::UserFn(uf)) = self.env.get(&topic) {
                                if let Some(doc) = &uf.doc {
                                    println!("fn {}({})\n{}", topic, uf.params.join(", "), doc);
                                } else {
                                    println!("fn {}({})\n  (no docstring)", topic, uf.params.join(", "));
                                }
                            } else {
                                println!(
                                    "help: '{}' not documented. Type help() for full list.",
                                    topic
                                );
                            }
                        }
                    }
                }
                Ok(Value::Nil)
            }

            // ── describe ─────────────────────────────────────────────────────
            "describe" => {
                if args.len() != 1 {
                    return Err(HayashiError::Runtime("describe() takes 1 argument".into()));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => Some(n.clone()),
                    _ => None,
                };
                match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => {
                        println!("{}", df);
                        // mostrar labels se existirem
                        if let Some(ref name) = df_name {
                            if let Some(var_labels) = self.labels.get(name) {
                                if !var_labels.is_empty() {
                                    println!("\n  Labels:");
                                    let mut sorted: Vec<_> = var_labels.iter().collect();
                                    sorted.sort_by_key(|(k, _)| (*k).clone());
                                    for (var, lbl) in sorted {
                                        println!("    {:<20} {}", var, lbl);
                                    }
                                }
                            }
                        }
                        Ok(Value::Nil)
                    }
                    _ => Err(HayashiError::Type("describe() requires a DataFrame".into())),
                }
            }

            // ── codebook ─────────────────────────────────────────────────────
            "codebook" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "codebook(df [, var1, var2, ...])".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };

                let requested: Vec<String> = if args.len() > 1 {
                    self.resolve_var_list(&args[1..], &df)?
                } else {
                    let mut names = df.column_names();
                    names.sort();
                    names
                };

                let sep = "─".repeat(76);
                println!("\n{:═^76}", " Codebook ");

                for name in &requested {
                    use greeners::Column;
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                    println!("\n{sep}");
                    match col {
                        Column::Float(arr) => {
                            let total = arr.len();
                            let vals: Vec<f64> =
                                arr.iter().copied().filter(|x| x.is_finite()).collect();
                            let missing = total - vals.len();
                            let n = vals.len();
                            println!(
                                "  {:<20} type: float    obs: {}    missing: {}",
                                name, total, missing
                            );
                            if n > 0 {
                                let mean = vals.iter().sum::<f64>() / n as f64;
                                let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                                    / (n as f64 - 1.0).max(1.0);
                                let sd = var.sqrt();
                                let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                                let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let mut sorted = vals.clone();
                                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                let pctile = |p: f64| -> f64 {
                                    let idx = (p * (n - 1) as f64).round() as usize;
                                    sorted[idx.min(n - 1)]
                                };
                                let mut unique = sorted.clone();
                                unique.dedup();
                                println!(
                                    "  unique: {}    mean: {:.4}    sd: {:.4}",
                                    unique.len(),
                                    mean,
                                    sd
                                );
                                println!(
                                    "  min: {:.4}    p25: {:.4}    p50: {:.4}    p75: {:.4}    max: {:.4}",
                                    min,
                                    pctile(0.25),
                                    pctile(0.50),
                                    pctile(0.75),
                                    max
                                );
                            }
                        }
                        Column::Int(arr) => {
                            let total = arr.len();
                            let vals: Vec<f64> = arr.iter().map(|&x| x as f64).collect();
                            let n = vals.len();
                            println!("  {:<20} type: int      obs: {}    missing: 0", name, total);
                            if n > 0 {
                                let mean = vals.iter().sum::<f64>() / n as f64;
                                let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                                    / (n as f64 - 1.0).max(1.0);
                                let sd = var.sqrt();
                                let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                                let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let mut sorted = vals.clone();
                                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                let mut unique = sorted.clone();
                                unique.dedup();
                                println!(
                                    "  unique: {}    mean: {:.4}    sd: {:.4}",
                                    unique.len(),
                                    mean,
                                    sd
                                );
                                println!("  min: {:.0}    max: {:.0}", min, max);
                            }
                        }
                        Column::String(arr) => {
                            let total = arr.len();
                            let non_empty = arr.iter().filter(|s: &&String| !s.is_empty()).count();
                            let missing = total - non_empty;
                            let mut unique: Vec<&str> =
                                arr.iter().map(|s: &String| s.as_str()).collect();
                            unique.sort();
                            unique.dedup();
                            println!(
                                "  {:<20} type: string   obs: {}    missing: {}",
                                name, total, missing
                            );
                            println!("  unique: {}", unique.len());
                            if unique.len() <= 10 {
                                let examples: Vec<&str> = unique.iter().take(10).copied().collect();
                                println!("  values: {}", examples.join(", "));
                            } else {
                                let first5: Vec<&str> = unique.iter().take(5).copied().collect();
                                println!(
                                    "  values: {}, ... ({} more)",
                                    first5.join(", "),
                                    unique.len() - 5
                                );
                            }
                        }
                        Column::Bool(arr) => {
                            let total = arr.len();
                            let trues = arr.iter().filter(|&&b| b).count();
                            let falses = total - trues;
                            println!("  {:<20} type: bool     obs: {}    missing: 0", name, total);
                            println!("  true: {}    false: {}", trues, falses);
                        }
                        _ => {
                            println!("  {:<20} type: other", name);
                        }
                    }
                }
                println!("\n{sep}");
                println!();
                Ok(Value::Nil)
            }

            // ── format: formata valor numérico ──────────────────────────────
            "format" | "fmt" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "format(value, fmt_str) — Ex: format(3.14, \"%.2f\")".into(),
                    ));
                }
                let val = match self.eval_expr(&args[0])? {
                    Value::Float(f) => f,
                    Value::Int(i) => i as f64,
                    other => {
                        return Err(HayashiError::Type(format!(
                            "format(): primeiro argumento must be numeric, não {other}"
                        )))
                    }
                };
                let fmt_s = match self.eval_expr(&args[1])? {
                    Value::Str(s) => s,
                    _ => {
                        return Err(HayashiError::Type(
                            "format(): second argument must be string (ex: \"%.2f\")".into(),
                        ))
                    }
                };
                // parse "%.Nf" → N decimal places
                let decimals: usize = if fmt_s.starts_with("%.") && fmt_s.ends_with('f') {
                    fmt_s[2..fmt_s.len() - 1].parse().unwrap_or(4)
                } else if fmt_s.starts_with('%') && fmt_s.ends_with('f') {
                    // "%f" sem especificar decimais
                    6
                } else {
                    return Err(HayashiError::Runtime(format!(
                        "format(): string de formato '{fmt_s}' não reconhecida (use \"%.Nf\")"
                    )));
                };
                Ok(Value::Str(format!("{:.prec$}", val, prec = decimals)))
            }

            // ── duplicates: reportar/dropar/marcar duplicatas ────────────────
            "duplicates" => {
                if args.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "duplicates(df, var [, action=report|drop|tag])".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "duplicates(): primeiro argumento deve ser variable name".into(),
                        ))
                    }
                };
                let df = match self.env.get(&df_name) {
                    Some(Value::DataFrame(d)) => d.clone(),
                    _ => {
                        return Err(HayashiError::Runtime(format!(
                            "'{df_name}' is not a DataFrame"
                        )))
                    }
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "duplicates(): second argument must be nome de coluna".into(),
                        ))
                    }
                };
                let action = match opt_map.get("action") {
                    Some(Value::Str(s)) => s.clone(),
                    None => "report".into(),
                    _ => "report".into(),
                };

                let col = Self::get_col_f64(&df, &var_name)?;
                let n = col.len();

                // contar ocorrências de cada valor
                let mut counts: HashMap<i64, usize> = HashMap::new();
                for &v in col.iter() {
                    let key = v.to_bits() as i64;
                    *counts.entry(key).or_insert(0) += 1;
                }

                let n_dup: usize = counts.values().filter(|&&c| c > 1).map(|c| c - 1).sum();
                let n_unique = counts.len();

                match action.as_str() {
                    "report" => {
                        println!("duplicates report: {var_name}");
                        println!("  observações:    {n}");
                        println!("  valores únicos: {n_unique}");
                        println!("  duplicatas:     {n_dup}");
                        Ok(Value::Int(n_dup as i64))
                    }
                    "drop" => {
                        let mut seen: std::collections::HashSet<i64> =
                            std::collections::HashSet::new();
                        let keep: Vec<usize> = (0..n)
                            .filter(|&i| {
                                let key = col[i].to_bits() as i64;
                                seen.insert(key)
                            })
                            .collect();
                        let new_df = df
                            .iloc(Some(&keep), None)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!(
                            "duplicates drop: {n_dup} obs removidas, {} restantes",
                            new_df.n_rows()
                        );
                        self.env.set(&df_name, Value::DataFrame(Rc::new(new_df)))?;
                        Ok(Value::Nil)
                    }
                    "tag" => {
                        let dup_col: Vec<f64> = (0..n)
                            .map(|i| {
                                let key = col[i].to_bits() as i64;
                                *counts.get(&key).unwrap_or(&1) as f64
                            })
                            .collect();
                        let mut df_mut = df.clone();
                        let arr = ndarray::Array1::from(dup_col);
                        Rc::make_mut(&mut df_mut)
                            .insert("_dup".to_string(), arr)
                            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                        println!("duplicates tag: coluna _dup gerada ({n_dup} duplicatas)");
                        self.env.set(&df_name, Value::DataFrame(df_mut))?;
                        Ok(Value::Nil)
                    }
                    other => Err(HayashiError::Runtime(format!(
                        "duplicates(): action '{other}' desconhecida (report|drop|tag)"
                    ))),
                }
            }

            // ── label: armazena rótulos de variáveis ─────────────────────────
            "label" => {
                if args.len() < 3 {
                    return Err(HayashiError::Runtime(
                        "label(df, var, \"descrição\")".into(),
                    ));
                }
                let df_name = match &args[0] {
                    Expr::Var(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "label(): primeiro argumento deve ser nome do DataFrame".into(),
                        ))
                    }
                };
                let var_name = match &args[1] {
                    Expr::Var(n) | Expr::Str(n) => n.clone(),
                    _ => {
                        return Err(HayashiError::Type(
                            "label(): second argument must be variable name".into(),
                        ))
                    }
                };
                let description = match self.eval_expr(&args[2])? {
                    Value::Str(s) => s,
                    _ => {
                        return Err(HayashiError::Type(
                            "label(): terceiro argumento must be string".into(),
                        ))
                    }
                };
                self.labels
                    .entry(df_name.clone())
                    .or_default()
                    .insert(var_name.clone(), description.clone());
                println!("label {df_name}.{var_name} = \"{description}\"");
                Ok(Value::Nil)
            }

            // ── correlate ────────────────────────────────────────────────────
            "correlate" | "corr" | "pwcorr" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "correlate() requires a DataFrame as first argument".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };

                // variáveis pedidas ou todas as numéricas
                let names: Vec<String> = if args.len() > 1 {
                    self.resolve_var_list(&args[1..], &df)?
                } else {
                    use greeners::Column;
                    let mut ns: Vec<String> = df
                        .column_names()
                        .into_iter()
                        .filter(|n| {
                            matches!(df.get_column(n), Ok(Column::Float(_)) | Ok(Column::Int(_)))
                        })
                        .collect();
                    ns.sort();
                    ns
                };

                if names.len() < 2 {
                    return Err(HayashiError::Runtime(
                        "correlate() needs at least 2 numeric variables".into(),
                    ));
                }

                let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                let sub = df
                    .select(&refs)
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;
                let mat = sub
                    .corr()
                    .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                // corr() ordena colunas alfabeticamente — sincronizar com a matriz
                let mut sorted_names = names.clone();
                sorted_names.sort();

                let col_w = 10usize;
                let row_label_w = 16usize;
                let trunc = |s: &str, w: usize| {
                    if s.len() > w {
                        s[..w].to_string()
                    } else {
                        s.to_string()
                    }
                };

                // cabeçalho
                print!("{:>width$} |", "", width = row_label_w);
                for name in &sorted_names {
                    print!(" {:>width$}", trunc(name, col_w), width = col_w);
                }
                println!();
                println!(
                    "{}-+{}",
                    "-".repeat(row_label_w),
                    "-".repeat((col_w + 1) * sorted_names.len())
                );

                // p-value: t = r*sqrt(n-2)/sqrt(1-r²), df=n-2
                let show_stars =
                    func == "pwcorr" || matches!(opt_map.get("star"), Some(Value::Bool(true)));
                let n_obs = df.n_rows() as f64;
                let corr_pval = |r: f64| -> f64 {
                    if n_obs <= 2.0 || (1.0 - r * r) <= 0.0 {
                        return 1.0;
                    }
                    let t = r * (n_obs - 2.0).sqrt() / (1.0 - r * r).sqrt();
                    t_pvalue_two(t, n_obs - 2.0)
                };
                let star = |p: f64| -> &str {
                    if p < 0.01 {
                        "***"
                    } else if p < 0.05 {
                        "**"
                    } else if p < 0.10 {
                        "*"
                    } else {
                        ""
                    }
                };

                for (i, row_name) in sorted_names.iter().enumerate() {
                    print!(
                        "{:>width$} |",
                        trunc(row_name, row_label_w),
                        width = row_label_w
                    );
                    for j in 0..=i {
                        let r = mat[[i, j]];
                        if show_stars && i != j {
                            let s = star(corr_pval(r));
                            print!(" {:>7.4}{:<3}", r, s);
                        } else {
                            print!(" {:>10.4}", r);
                        }
                    }
                    println!();
                }
                if show_stars {
                    println!("* p<0.10  ** p<0.05  *** p<0.01");
                }
                println!();
                Ok(Value::Nil)
            }

            // ── summarize ────────────────────────────────────────────────────
            "summarize" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime(
                        "summarize() requires a DataFrame as first argument".into(),
                    ));
                }
                let df = match self.eval_expr(&args[0])? {
                    Value::DataFrame(df) => df,
                    other => return Err(self.type_mismatch("DataFrame", &other)),
                };

                let requested: Vec<String> = if args.len() > 1 {
                    self.resolve_var_list(&args[1..], &df)?
                } else {
                    let mut names = df.column_names();
                    names.sort();
                    names
                };

                let detail = matches!(opt_map.get("detail"), Some(Value::Bool(true)))
                    || matches!(opt_map.get("d"), Some(Value::Bool(true)));
                let quiet = self.capturing;

                if !quiet {
                    println!(
                        "\n{:<16} {:>9}  {:>7}  {:>12} {:>12} {:>12} {:>12}",
                        "Variable", "Obs", "Missing", "Mean", "Std. Dev.", "Min", "Max"
                    );
                    println!("{}", "-".repeat(91));
                }

                let mut result_dicts: Vec<(String, HashMap<String, Value>)> = Vec::new();

                for name in &requested {
                    use greeners::Column;
                    let col = df
                        .get_column(name)
                        .map_err(|e| HayashiError::Runtime(e.to_string()))?;

                    let (n_total, n_missing, vals): (usize, usize, Vec<f64>) = match col {
                        Column::Float(arr) => {
                            let total = arr.len();
                            let vals: Vec<f64> =
                                arr.iter().copied().filter(|x| x.is_finite()).collect();
                            let missing = total - vals.len();
                            (total, missing, vals)
                        }
                        Column::Int(arr) => {
                            let vals: Vec<f64> = arr.iter().map(|&x| x as f64).collect();
                            (vals.len(), 0, vals)
                        }
                        _ => {
                            if !quiet {
                                println!("{:<16} {:>9}  {:>7}", name, "(non-numeric)", "");
                            }
                            continue;
                        }
                    };

                    let n = vals.len();
                    if n == 0 {
                        if !quiet {
                            println!("{:<16} {:>9}  {:>7}  (all missing)", name, 0, n_total);
                        }
                        continue;
                    }

                    let mean = vals.iter().sum::<f64>() / n as f64;
                    let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / (n as f64 - 1.0).max(1.0);
                    let sd = variance.sqrt();
                    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    if !quiet {
                        let miss_str = if n_missing > 0 {
                            format!("{}", n_missing)
                        } else {
                            String::new()
                        };
                        println!(
                            "{:<16} {:>9}  {:>7}  {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                            name, n, miss_str, mean, sd, min, max
                        );
                    }

                    let mut d = HashMap::new();
                    d.insert("N".into(), Value::Int(n as i64));
                    d.insert("missing".into(), Value::Int(n_missing as i64));
                    d.insert("mean".into(), Value::Float(mean));
                    d.insert("sd".into(), Value::Float(sd));
                    d.insert("min".into(), Value::Float(min));
                    d.insert("max".into(), Value::Float(max));
                    d.insert("variance".into(), Value::Float(variance));

                    if detail {
                        let mut sorted = vals.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let pctile = |p: f64| -> f64 {
                            let idx = (p * (n - 1) as f64).round() as usize;
                            sorted[idx.min(n - 1)]
                        };
                        let p1 = pctile(0.01);
                        let p5 = pctile(0.05);
                        let p10 = pctile(0.10);
                        let p25 = pctile(0.25);
                        let p50 = pctile(0.50);
                        let p75 = pctile(0.75);
                        let p90 = pctile(0.90);
                        let p95 = pctile(0.95);
                        let p99 = pctile(0.99);
                        let skew = if n > 2 {
                            let m3 = vals.iter().map(|x| ((x - mean) / sd).powi(3)).sum::<f64>();
                            m3 * n as f64 / ((n - 1) as f64 * (n - 2) as f64)
                        } else {
                            f64::NAN
                        };
                        let kurt = if n > 3 {
                            let m4 = vals.iter().map(|x| ((x - mean) / sd).powi(4)).sum::<f64>()
                                / n as f64;
                            m4
                        } else {
                            f64::NAN
                        };
                        if !quiet {
                            println!("         Percentiles:");
                            println!("          1%  {:>10.4}       Skewness  {:>10.4}", p1, skew);
                            println!("          5%  {:>10.4}       Kurtosis  {:>10.4}", p5, kurt);
                            println!("         10%  {:>10.4}", p10);
                            println!(
                                "         25%  {:>10.4}       Variance  {:>10.4}",
                                p25, variance
                            );
                            println!("         50%  {:>10.4}", p50);
                            println!("         75%  {:>10.4}", p75);
                            println!("         90%  {:>10.4}", p90);
                            println!("         95%  {:>10.4}", p95);
                            println!("         99%  {:>10.4}", p99);
                        }
                        d.insert("p1".into(), Value::Float(p1));
                        d.insert("p5".into(), Value::Float(p5));
                        d.insert("p10".into(), Value::Float(p10));
                        d.insert("p25".into(), Value::Float(p25));
                        d.insert("p50".into(), Value::Float(p50));
                        d.insert("p75".into(), Value::Float(p75));
                        d.insert("p90".into(), Value::Float(p90));
                        d.insert("p95".into(), Value::Float(p95));
                        d.insert("p99".into(), Value::Float(p99));
                        d.insert("skewness".into(), Value::Float(skew));
                        d.insert("kurtosis".into(), Value::Float(kurt));
                    }
                    result_dicts.push((name.clone(), d));
                }
                if !quiet {
                    println!();
                }

                if quiet {
                    if result_dicts.len() == 1 {
                        let (_, d) = result_dicts.into_iter().next().unwrap();
                        Ok(Value::Dict(Rc::new(d)))
                    } else {
                        let mut outer = HashMap::new();
                        for (name, d) in result_dicts {
                            outer.insert(name, Value::Dict(Rc::new(d)));
                        }
                        Ok(Value::Dict(Rc::new(outer)))
                    }
                } else {
                    Ok(Value::Nil)
                }
            }

            // ── esttab ───────────────────────────────────────────────────────
            // ── eststo: acumula modelo para esttab posterior ──────────────
            "eststo" | "est_store" => {
                if args.is_empty() {
                    return Err(HayashiError::Runtime("eststo(model)".into()));
                }
                let val = self.eval_expr(&args[0])?;
                let n = self.stored_models.len() + 1;
                self.stored_models.push(val);
                println!(
                    "eststo: modelo {n} armazenado ({} total)",
                    self.stored_models.len()
                );
                Ok(Value::Nil)
            }

            "estclear" => {
                let n = self.stored_models.len();
                self.stored_models.clear();
                println!("estclear: {n} modelos removidos");
                Ok(Value::Nil)
            }

            "esttab" => {
                // sem args → usa modelos acumulados via eststo
                let use_stored = args.is_empty();
                if use_stored && self.stored_models.is_empty() {
                    return Err(HayashiError::Runtime(
                        "esttab() requires models — pass as args or use eststo() first".into(),
                    ));
                }

                let fmt = match opt_map.get("fmt") {
                    Some(Value::Str(s)) => s.clone(),
                    None => "txt".to_string(),
                    _ => return Err(HayashiError::Type("fmt= must be a string".into())),
                };
                let out_path = match opt_map.get("path") {
                    Some(Value::Str(s)) => Some(s.clone()),
                    None => None,
                    _ => return Err(HayashiError::Type("path= must be a string".into())),
                };

                // (nome_variável, coef, se_opt, pval_opt)
                type CoefRow = (String, f64, Option<f64>, Option<f64>);
                // (label, coefs, n_obs, fit_stats)
                struct ModelInfo {
                    label: String,
                    coefs: Vec<CoefRow>,
                    n: usize,
                    r2: Option<f64>,
                    adj_r2: Option<f64>,
                    #[allow(dead_code)]
                    ll: Option<f64>,
                }

                // parseia CSV do OlsResult: variable,coef,se,t,p
                let parse_csv = |csv: &str| -> Vec<CoefRow> {
                    let mut rows = Vec::new();
                    let mut first = true;
                    for line in csv.lines() {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        if first {
                            first = false;
                            continue;
                        } // cabeçalho
                        let f: Vec<&str> = line.splitn(6, ',').collect();
                        if f.len() >= 5 {
                            let raw = f[0].trim().trim_matches('"');
                            let name = if raw == "const" {
                                "_cons".to_string()
                            } else {
                                raw.to_string()
                            };
                            let coef = f[1].trim().parse::<f64>().unwrap_or(f64::NAN);
                            let se = f[2].trim().parse::<f64>().unwrap_or(f64::NAN);
                            let p = f[4].trim().parse::<f64>().unwrap_or(1.0);
                            rows.push((name, coef, Some(se), Some(p)));
                        }
                    }
                    rows
                };

                let stars = |p: Option<f64>| match p {
                    Some(p) if p < 0.01 => "***",
                    Some(p) if p < 0.05 => "**",
                    Some(p) if p < 0.10 => "*",
                    _ => "",
                };

                let extract_std = |label: &str,
                                   vnames: &Option<Vec<String>>,
                                   params: &ndarray::Array1<f64>,
                                   se: &ndarray::Array1<f64>,
                                   pv: &ndarray::Array1<f64>,
                                   n: usize|
                 -> ModelInfo {
                    let k = params.len();
                    let fb: Vec<String> = (0..k).map(|i| format!("x{i}")).collect();
                    let nm = vnames.as_ref().unwrap_or(&fb);
                    let coefs: Vec<CoefRow> = nm
                        .iter()
                        .zip(params.iter())
                        .zip(se.iter())
                        .zip(pv.iter())
                        .map(|(((n, &c), &s), &p)| (n.clone(), c, Some(s), Some(p)))
                        .collect();
                    ModelInfo {
                        label: label.to_string(),
                        coefs,
                        n,
                        r2: None,
                        adj_r2: None,
                        ll: None,
                    }
                };

                let mut models: Vec<ModelInfo> = Vec::new();
                let model_vals: Vec<Value> = if use_stored {
                    self.stored_models.clone()
                } else {
                    let mut vals = Vec::new();
                    for a in args {
                        let v = self.eval_expr(a)?;
                        if let Value::List(items) = v {
                            vals.extend(items.iter().cloned());
                        } else {
                            vals.push(v);
                        }
                    }
                    vals
                };
                for val in model_vals {
                    match val {
                        Value::OlsResult(m) => {
                            use greeners::ExportableResult;
                            let coefs = parse_csv(&m.result.to_csv());
                            let n = m.residuals.len();
                            let cov_label = match &m.result.cov_type {
                                CovarianceType::NonRobust => "",
                                CovarianceType::HC1 => " (robust)",
                                CovarianceType::HC2 => " (HC2)",
                                CovarianceType::HC3 => " (HC3)",
                                CovarianceType::HC4 => " (HC4)",
                                CovarianceType::NeweyWest(l) => {
                                    let _ = l;
                                    " (NW)"
                                }
                                CovarianceType::Clustered(_) => " (cluster)",
                                CovarianceType::ClusteredTwoWay(_, _) => " (2w-cluster)",
                            };
                            models.push(ModelInfo {
                                label: format!("OLS{cov_label}"),
                                coefs,
                                n,
                                r2: Some(m.result.r_squared),
                                adj_r2: Some(m.result.adj_r_squared),
                                ll: Some(m.result.log_likelihood),
                            });
                        }
                        Value::BinaryResult(bm) => {
                            let label = if bm.kind == "logit" {
                                "Logit"
                            } else {
                                "Probit"
                            }
                            .to_string();
                            let n = bm.x.nrows();
                            models.push(extract_std(
                                &label,
                                &bm.result.variable_names,
                                &bm.result.params,
                                &bm.result.std_errors,
                                &bm.result.p_values,
                                n,
                            ));
                        }
                        Value::IvResult(r) => {
                            models.push(extract_std(
                                "IV/2SLS",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::PoissonResult(r) => {
                            models.push(extract_std(
                                "Poisson",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::NegBinResult(r) => {
                            models.push(extract_std(
                                "NegBin",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::OrderedResult(r) => {
                            let mut info = extract_std(
                                &r.model_name,
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            );
                            for (i, (&thr, &thr_se)) in r
                                .thresholds
                                .iter()
                                .zip(r.threshold_std_errors.iter())
                                .enumerate()
                            {
                                info.coefs.push((
                                    format!("_cut{}", i + 1),
                                    thr,
                                    Some(thr_se),
                                    None,
                                ));
                            }
                            models.push(info);
                        }
                        Value::TobitResult(r) => {
                            let mut info = extract_std(
                                "Tobit",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            );
                            info.coefs.push(("_sigma".into(), r.sigma, None, None));
                            models.push(info);
                        }
                        Value::HeckmanResult(r) => {
                            let mut info = extract_std(
                                "Heckman",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            );
                            let dz = if r.delta_se > 0.0 {
                                r.delta / r.delta_se
                            } else {
                                f64::NAN
                            };
                            let dp = if dz.is_finite() {
                                t_pvalue_two(dz, r.n_selected as f64)
                            } else {
                                f64::NAN
                            };
                            info.coefs.push((
                                "_lambda".into(),
                                r.delta,
                                Some(r.delta_se),
                                Some(dp),
                            ));
                            models.push(info);
                        }
                        Value::PanelResult(r) => {
                            models.push(extract_std(
                                "FE",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::ReResult(r) => {
                            models.push(extract_std(
                                "RE",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                0,
                            ));
                        }
                        Value::AbResult(r) => {
                            models.push(extract_std(
                                "AB-GMM",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::GmmResult(r) => {
                            let names: Option<Vec<String>> =
                                Some((0..r.params.len()).map(|i| format!("x{i}")).collect());
                            models.push(extract_std(
                                "GMM",
                                &names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::SysGmmResult(r) => {
                            models.push(extract_std(
                                "SysGMM",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs_fd,
                            ));
                        }
                        Value::PcseResult(r) => {
                            models.push(extract_std(
                                "PCSE",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::PanelGlsResult(r) => {
                            let label = match r.panels {
                                greeners::panel::GlsPanels::Hetero => "XTGLS-H",
                                greeners::panel::GlsPanels::Correlated => "XTGLS-C",
                            };
                            models.push(extract_std(
                                label,
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::FE2SLSResult(r) => {
                            models.push(extract_std(
                                "FE-IV",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::QuantileResult(r) => {
                            let label = format!("QReg(τ={:.2})", r.tau);
                            models.push(extract_std(
                                &label,
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                0,
                            ));
                        }
                        Value::CoxResult(r) => {
                            models.push(extract_std(
                                "CoxPH",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::RlmResult(r) => {
                            models.push(extract_std(
                                "RLM",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::GeeResult(r) => {
                            // GEE usa SE robusto (sandwich) por padrão
                            models.push(extract_std(
                                "GEE",
                                &r.variable_names,
                                &r.params,
                                &r.robust_se,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::BetaResult(r) => {
                            models.push(extract_std(
                                "BetaReg",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::GlmResult(r) => {
                            let family_name = format!("GLM({:?})", r.family);
                            models.push(extract_std(
                                &family_name,
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::LowessResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta lowess — use predict para extrair valores suavizados".into()
                            ));
                        }
                        Value::PcaResult(_) | Value::FactorResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta PCA/Factor — use print() para ver cargas e variância explicada".into()
                            ));
                        }
                        Value::ConditionalResult(r) => {
                            models.push(extract_std(
                                &r.model_name,
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::MarkovResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta Markov Switching — use print() para ver parâmetros por regime".into()
                            ));
                        }
                        Value::GlsarResult(r) => {
                            models.push(extract_std(
                                "GLSAR",
                                &r.variable_names,
                                &r.params,
                                &r.std_errors,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::MixedResult(r) => {
                            // esttab exibe apenas efeitos fixos do MixedLM
                            models.push(extract_std(
                                "MixedLM",
                                &r.variable_names,
                                &r.fixed_effects,
                                &r.fixed_se,
                                &r.p_values,
                                r.n_obs,
                            ));
                        }
                        Value::ZeroInflatedResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta zip/zinb (duas equações) — use print()"
                                    .into(),
                            ));
                        }
                        Value::SurResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta sur (múltiplas equações) — use print()"
                                    .into(),
                            ));
                        }
                        Value::RollingResult(_) | Value::RecursiveLSResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta rolling/recursive — coeficientes variam ao longo do tempo; use print()".into()
                            ));
                        }
                        Value::MNLogitResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta mlogit (múltiplas equações) — use print()"
                                    .into(),
                            ));
                        }
                        Value::DidResult(_) | Value::KMResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta did/km — resultado tem formato próprio; use print()".into()
                            ));
                        }
                        Value::RdResult(_) | Value::SynthResult(_) | Value::PsmResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta estimadores causais (rd, psm, synth) — use print()".into()
                            ));
                        }
                        Value::VarmaResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta VARMA (coeficientes matriciais) — use print()".into()
                            ));
                        }
                        Value::DecompResult(_) | Value::MstlResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta decomposição sazonal — use print()".into(),
                            ));
                        }
                        Value::UCResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta UCM (parâmetros de variância, não β) — use print()".into()
                            ));
                        }
                        Value::GamResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta GAM (termos smooth não têm tabela β padrão) — use print()".into()
                            ));
                        }
                        Value::MiceResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta MICE (múltiplos datasets) — estime modelo em cada dataset e use Rubin's rules".into()
                            ));
                        }
                        Value::MSARResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta Markov-AR (parâmetros por regime) — use print()".into()
                            ));
                        }
                        Value::SVarResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta SVAR (matrizes A/B estruturais) — use print()".into()
                            ));
                        }
                        Value::ThreeSLSResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta 3SLS (múltiplas equações) — use print()"
                                    .into(),
                            ));
                        }
                        Value::DFMResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta DFM (fatores latentes) — use print()".into(),
                            ));
                        }
                        Value::EtsResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta ETS (parâmetros de suavização) — use print()".into()
                            ));
                        }
                        Value::ThresholdResult(_) => {
                            return Err(HayashiError::Runtime(
                                "esttab() not supporta panel threshold (dois regimes) — use print()".into()
                            ));
                        }
                        _ => {
                            return Err(HayashiError::Type(
                                "esttab(): tipo de modelo not supportado — use print()".into(),
                            ))
                        }
                    }
                }

                // união dos nomes de variáveis na ordem de primeira ocorrência
                let mut all_vars: Vec<String> = Vec::new();
                let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                for mi in &models {
                    let coefs = &mi.coefs;
                    for (nm, _, _, _) in coefs {
                        if seen.insert(nm.clone()) {
                            all_vars.push(nm.clone());
                        }
                    }
                }

                let n_models = models.len();
                let col_w = 16usize;
                let label_w = all_vars.iter().map(|s| s.len()).max().unwrap_or(8).max(12) + 2;
                let total_w = label_w + n_models * (col_w + 1);

                // monta conteúdo (txt ou latex)
                let mut buf = String::new();

                if fmt == "latex" || fmt == "tex" {
                    buf.push_str("\\begin{tabular}{l");
                    for _ in 0..n_models {
                        buf.push('r');
                    }
                    buf.push_str("}\n\\hline\\hline\n");
                    // cabeçalho
                    buf.push_str(" &");
                    for (i, mi) in models.iter().enumerate() {
                        let label = &mi.label;
                        buf.push_str(&format!(" ({}) {}", i + 1, label));
                        if i + 1 < n_models {
                            buf.push('&');
                        }
                    }
                    buf.push_str(" \\\\\n\\hline\n");

                    for var in &all_vars {
                        if var == "_cons" {
                            continue;
                        } // _cons vai no final
                        buf.push_str(&var.to_string());
                        for mi in &models {
                            let coefs = &mi.coefs;
                            let row = coefs.iter().find(|(nm, _, _, _)| nm == var);
                            match row {
                                Some((_, c, _, p)) => {
                                    buf.push_str(&format!(" & {:.4}{}", c, stars(*p)))
                                }
                                None => buf.push_str(" &"),
                            }
                        }
                        buf.push_str(" \\\\\n");
                        // SE linha
                        let has_se = models.iter().any(|mi| {
                            mi.coefs
                                .iter()
                                .find(|(nm, _, _, _)| nm == var)
                                .and_then(|(_, _, se, _)| *se)
                                .is_some()
                        });
                        if has_se {
                            buf.push(' ');
                            for mi in &models {
                                let coefs = &mi.coefs;
                                let row = coefs.iter().find(|(nm, _, _, _)| nm == var);
                                match row.and_then(|(_, _, se, _)| *se) {
                                    Some(se) => buf.push_str(&format!(" & ({:.4})", se)),
                                    None => buf.push_str(" &"),
                                }
                            }
                            buf.push_str(" \\\\\n");
                        }
                    }
                    // _cons no final
                    if all_vars.iter().any(|v| v == "_cons") {
                        buf.push_str("Constant");
                        for mi in &models {
                            let coefs = &mi.coefs;
                            let row = coefs.iter().find(|(nm, _, _, _)| nm == "_cons");
                            match row {
                                Some((_, c, _, p)) => {
                                    buf.push_str(&format!(" & {:.4}{}", c, stars(*p)))
                                }
                                None => buf.push_str(" &"),
                            }
                        }
                        buf.push_str(" \\\\\n");
                        let has_se = models.iter().any(|mi| {
                            mi.coefs
                                .iter()
                                .find(|(nm, _, _, _)| nm == "_cons")
                                .and_then(|(_, _, se, _)| *se)
                                .is_some()
                        });
                        if has_se {
                            buf.push(' ');
                            for mi in &models {
                                let coefs = &mi.coefs;
                                let row = coefs.iter().find(|(nm, _, _, _)| nm == "_cons");
                                match row.and_then(|(_, _, se, _)| *se) {
                                    Some(se) => buf.push_str(&format!(" & ({:.4})", se)),
                                    None => buf.push_str(" &"),
                                }
                            }
                            buf.push_str(" \\\\\n");
                        }
                    }
                    buf.push_str("\\hline\nN");
                    for mi in &models {
                        buf.push_str(&format!(" & {}", mi.n));
                    }
                    buf.push_str(" \\\\\n");
                    if models.iter().any(|mi| mi.r2.is_some()) {
                        buf.push_str("$R^2$");
                        for mi in &models {
                            match mi.r2 {
                                Some(v) => buf.push_str(&format!(" & {:.4}", v)),
                                None => buf.push_str(" &"),
                            }
                        }
                        buf.push_str(" \\\\\n");
                    }
                    if models.iter().any(|mi| mi.adj_r2.is_some()) {
                        buf.push_str("Adj. $R^2$");
                        for mi in &models {
                            match mi.adj_r2 {
                                Some(v) => buf.push_str(&format!(" & {:.4}", v)),
                                None => buf.push_str(" &"),
                            }
                        }
                        buf.push_str(" \\\\\n");
                    }
                    buf.push_str("\\hline\\hline\n\\end{tabular}\n");
                    buf.push_str("\\footnotesize{* p$<$0.10, ** p$<$0.05, *** p$<$0.01}\n");
                } else {
                    // ── ASCII txt ─────────────────────────────────────────────
                    let sep = "─".repeat(total_w);

                    // cabeçalho: numeração
                    let mut line = format!("{:<lw$}", "", lw = label_w);
                    for i in 0..n_models {
                        line.push_str(&format!(" {:>cw$}", format!("({})", i + 1), cw = col_w));
                    }
                    buf.push_str(&format!("{line}\n"));

                    // cabeçalho: labels
                    let mut line = format!("{:<lw$}", "", lw = label_w);
                    for mi in &models {
                        line.push_str(&format!(" {:>cw$}", mi.label, cw = col_w));
                    }
                    buf.push_str(&format!("{line}\n"));
                    buf.push_str(&format!("{sep}\n"));

                    let print_var = |var: &str, buf: &mut String| {
                        // linha de coeficientes
                        let display_name = if var == "_cons" { "Constant" } else { var };
                        let mut line = format!("{:<lw$}", display_name, lw = label_w);
                        for mi in &models {
                            let coefs = &mi.coefs;
                            let row = coefs.iter().find(|(nm, _, _, _)| nm == var);
                            match row {
                                Some((_, c, _, p)) => {
                                    let s = stars(*p);
                                    let cell = format!("{:.4}{}", c, s);
                                    line.push_str(&format!(" {:>cw$}", cell, cw = col_w));
                                }
                                None => line.push_str(&format!(" {:>cw$}", "", cw = col_w)),
                            }
                        }
                        buf.push_str(&format!("{line}\n"));

                        // linha de erros padrão
                        let has_se = models.iter().any(|mi| {
                            mi.coefs
                                .iter()
                                .find(|(nm, _, _, _)| nm == var)
                                .and_then(|(_, _, se, _)| *se)
                                .is_some()
                        });
                        if has_se {
                            let mut line = format!("{:<lw$}", "", lw = label_w);
                            for mi in &models {
                                let coefs = &mi.coefs;
                                let row = coefs.iter().find(|(nm, _, _, _)| nm == var);
                                match row.and_then(|(_, _, se, _)| *se) {
                                    Some(se) => line.push_str(&format!(
                                        " {:>cw$}",
                                        format!("({:.4})", se),
                                        cw = col_w
                                    )),
                                    None => line.push_str(&format!(" {:>cw$}", "", cw = col_w)),
                                }
                            }
                            buf.push_str(&format!("{line}\n"));
                        }
                    };

                    for var in &all_vars {
                        if var == "_cons" {
                            continue;
                        }
                        print_var(var, &mut buf);
                    }
                    if all_vars.iter().any(|v| v == "_cons") {
                        print_var("_cons", &mut buf);
                    }

                    buf.push_str(&format!("{sep}\n"));
                    let mut line = format!("{:<lw$}", "N", lw = label_w);
                    for mi in &models {
                        line.push_str(&format!(" {:>cw$}", mi.n, cw = col_w));
                    }
                    buf.push_str(&format!("{line}\n"));
                    if models.iter().any(|mi| mi.r2.is_some()) {
                        let mut line = format!("{:<lw$}", "R²", lw = label_w);
                        for mi in &models {
                            match mi.r2 {
                                Some(v) => line.push_str(&format!(" {:>cw$.4}", v, cw = col_w)),
                                None => line.push_str(&format!(" {:>cw$}", "", cw = col_w)),
                            }
                        }
                        buf.push_str(&format!("{line}\n"));
                    }
                    if models.iter().any(|mi| mi.adj_r2.is_some()) {
                        let mut line = format!("{:<lw$}", "Adj. R²", lw = label_w);
                        for mi in &models {
                            match mi.adj_r2 {
                                Some(v) => line.push_str(&format!(" {:>cw$.4}", v, cw = col_w)),
                                None => line.push_str(&format!(" {:>cw$}", "", cw = col_w)),
                            }
                        }
                        buf.push_str(&format!("{line}\n"));
                    }
                    buf.push_str(&format!("{sep}\n"));
                    buf.push_str("* p<0.10  ** p<0.05  *** p<0.01\n");
                }

                if let Some(path) = out_path {
                    std::fs::write(&path, &buf).map_err(|e| HayashiError::Io(e.to_string()))?;
                    println!("Exported table → '{path}'");
                } else {
                    print!("\n{buf}");
                }

                Ok(Value::Nil)
            }

            _ => return Ok(None),
        };
        result.map(Some)
    }
}
