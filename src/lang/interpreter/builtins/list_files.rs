use super::super::*;
impl Interpreter {
    pub(super) fn list_files(
        &mut self,
        func: &str,
        args: &[Expr],
        _opts: &[Opt],
        _opt_map: &HashMap<String, Value>,
    ) -> Result<Value> {
        match func {
            "list_files" => {
                let dir = if args.is_empty() {
                    ".".to_string()
                } else {
                    match self.eval_expr(&args[0])? {
                        Value::Str(s) => s,
                        _ => return Err(self.type_err("list_files: directory must be a string")),
                    }
                };
                let pattern = if args.len() > 1 {
                    match self.eval_expr(&args[1])? {
                        Value::Str(s) => Some(s),
                        _ => return Err(self.type_err("list_files: pattern must be a string")),
                    }
                } else {
                    None
                };

                let entries = std::fs::read_dir(&dir)
                    .map_err(|e| self.rt_err(format!("list_files: cannot read '{dir}': {e}")))?;

                let mut files: Vec<String> = Vec::new();
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if let Some(ref pat) = pattern {
                            if !name.contains(pat) {
                                continue;
                            }
                        }
                        files.push(path.to_string_lossy().to_string());
                    }
                }
                files.sort();
                Ok(Value::List(Arc::new(
                    files.into_iter().map(Value::Str).collect(),
                )))
            }
            _ => unreachable!(),
        }
    }
}
