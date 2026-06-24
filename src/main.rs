mod io;
mod lang;

use lang::interpreter::Interpreter;
use rustyline::completion::Completer;
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hinter, HistoryHinter};
use rustyline::validate::Validator;
use rustyline::{Context, Editor, Helper};
use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

// ── REPL Helper (tab completion, syntax highlighting, history hints) ─────────

const KEYWORDS: &[&str] = &[
    "let",
    "const",
    "fn",
    "if",
    "else",
    "for",
    "in",
    "while",
    "return",
    "break",
    "continue",
    "match",
    "try",
    "catch",
    "import",
    "source",
    "display",
    "input",
    "end",
    "generate",
    "replace",
    "load",
    "export",
    "predict",
    "xtset",
    "tsset",
    "ols",
    "reg",
    "fe",
    "re",
    "iv",
    "logit",
    "probit",
    "poisson",
    "nbreg",
    "tobit",
    "qreg",
    "rlm",
    "lasso",
    "ridge",
    "elasticnet",
    "garch",
    "arima",
    "var",
    "vecm",
    "hausman",
    "fmb",
    "portsort",
    "doublesort",
    "esttab",
    "eststo",
    "estclear",
    "test",
    "nlcom",
    "lincom",
    "margins",
    "bootstrap",
    "estat",
    "vif",
    "reset",
    "jb",
    "condnum",
    "coefplot",
    "summarize",
    "describe",
    "tabulate",
    "correlate",
    "pwcorr",
    "ttest",
    "ci",
    "list",
    "count",
    "sort",
    "filter",
    "drop",
    "keep",
    "rename",
    "collapse",
    "append",
    "merge",
    "reshape",
    "winsor",
    "tabgen",
    "encode",
    "recode",
    "duplicates",
    "label",
    "preserve",
    "restore",
    "quietly",
    "capture",
    "assert",
    "push",
    "pop",
    "len",
    "keys",
    "values",
    "map",
    "select",
    "unique",
    "flatten",
    "mean",
    "sum",
    "min",
    "max",
    "std",
    "abs",
    "sqrt",
    "log",
    "exp",
    "help",
    "timer",
    "set_seed",
    "format",
    "typeof",
    "drop_collinear",
    "true",
    "false",
    "nil",
];

struct HayHelper {
    vars: Rc<RefCell<Vec<String>>>,
    hinter: HistoryHinter,
}

impl HayHelper {
    fn new(vars: Rc<RefCell<Vec<String>>>) -> Self {
        Self {
            vars,
            hinter: HistoryHinter {},
        }
    }

    fn completions_for(&self, word: &str) -> Vec<String> {
        let mut matches: Vec<String> = KEYWORDS
            .iter()
            .filter(|k| k.starts_with(word))
            .map(|k| k.to_string())
            .collect();
        for v in self.vars.borrow().iter() {
            if v.starts_with(word) && !matches.contains(v) {
                matches.push(v.clone());
            }
        }
        matches.sort();
        matches
    }
}

impl Completer for HayHelper {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<String>)> {
        let start = line[..pos]
            .rfind(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|i| i + 1)
            .unwrap_or(0);
        let word = &line[start..pos];
        if word.is_empty() {
            return Ok((pos, vec![]));
        }
        Ok((start, self.completions_for(word)))
    }
}

impl Highlighter for HayHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        let mut out = String::with_capacity(line.len() * 2);
        let mut chars = line.chars().peekable();
        let mut word = String::new();

        let flush_word = |word: &mut String, out: &mut String| {
            if word.is_empty() {
                return;
            }
            let w = word.as_str();
            if KEYWORDS.contains(&w) {
                out.push_str("\x1b[1;34m");
                out.push_str(w);
                out.push_str("\x1b[0m");
            } else if w.parse::<f64>().is_ok() {
                out.push_str("\x1b[33m");
                out.push_str(w);
                out.push_str("\x1b[0m");
            } else {
                out.push_str(w);
            }
            word.clear();
        };

        while let Some(c) = chars.next() {
            if c == '"' {
                flush_word(&mut word, &mut out);
                out.push_str("\x1b[32m\"");
                for c2 in chars.by_ref() {
                    out.push(c2);
                    if c2 == '"' {
                        break;
                    }
                }
                out.push_str("\x1b[0m");
            } else if (c == '/' && chars.peek() == Some(&'/')) || (c == '#' && word.is_empty()) {
                flush_word(&mut word, &mut out);
                out.push_str("\x1b[90m");
                out.push(c);
                for c2 in chars.by_ref() {
                    out.push(c2);
                }
                out.push_str("\x1b[0m");
            } else if c.is_alphanumeric() || c == '_' || c == '.' {
                word.push(c);
            } else {
                flush_word(&mut word, &mut out);
                out.push(c);
            }
        }
        flush_word(&mut word, &mut out);
        Cow::Owned(out)
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Owned(format!("\x1b[1;32m{}\x1b[0m", prompt))
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(format!("\x1b[90m{}\x1b[0m", hint))
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        true
    }
}

impl Hinter for HayHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &Context<'_>) -> Option<String> {
        self.hinter.hint(line, pos, ctx)
    }
}

impl Validator for HayHelper {}
impl Helper for HayHelper {}

const VERSION: &str = env!("CARGO_PKG_VERSION");
const HISTORY_FILE: &str = ".hay_history";

fn main() {
    const STACK_SIZE: usize = 32 * 1024 * 1024;
    let handler = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(run)
        .expect("failed to spawn main thread");
    if let Err(e) = handler.join() {
        if let Some(msg) = e.downcast_ref::<&str>() {
            eprintln!("fatal: {msg}");
        } else if let Some(msg) = e.downcast_ref::<String>() {
            eprintln!("fatal: {msg}");
        }
        std::process::exit(1);
    }
}

fn run() {
    let args: Vec<String> = std::env::args().collect();

    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    let args_clean: Vec<&str> = args
        .iter()
        .map(String::as_str)
        .filter(|a| *a != "--verbose" && *a != "-v")
        .collect();

    match args_clean.get(1).copied() {
        Some("--version") | Some("-V") => {
            println!("hay {VERSION}");
            return;
        }
        Some("--help") | Some("-h") => {
            print_help();
            return;
        }
        Some("-") => {
            use std::io::Read;
            let mut src = String::new();
            std::io::stdin()
                .read_to_string(&mut src)
                .expect("failed to read stdin");
            let mut interp = Interpreter::new();
            interp.load_plugins();
            if let Err(e) = lang::run_source_verbose(&src, &mut interp, verbose) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
            return;
        }
        Some("install") => {
            let pkg = args_clean.get(2).unwrap_or_else(|| {
                eprintln!("Usage: hay install user/repo");
                std::process::exit(1);
            });
            pkg_install(pkg);
            return;
        }
        Some("remove") | Some("uninstall") => {
            let pkg = args_clean.get(2).unwrap_or_else(|| {
                eprintln!("Usage: hay remove package_name");
                std::process::exit(1);
            });
            pkg_remove(pkg);
            return;
        }
        Some("list") | Some("packages") => {
            pkg_list();
            return;
        }
        Some(path) if !path.starts_with('-') => {
            run_script(path, verbose);
            return;
        }
        Some(unknown) => {
            eprintln!("hay: unknown argument '{unknown}'");
            eprintln!("Usage: hay [script.hay | - | install | remove | list]");
            std::process::exit(1);
        }
        None => {}
    }

    run_repl();
}

fn run_script(path: &str, verbose: bool) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("hay: cannot read '{path}': {e}");
            std::process::exit(1);
        }
    };
    let mut interp = Interpreter::new();
    interp.load_plugins();
    if let Err(e) = lang::run_source_verbose(&src, &mut interp, verbose) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn brace_depth(s: &str) -> i32 {
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut prev = '\0';
    for c in s.chars() {
        if c == '"' && prev != '\\' {
            in_string = !in_string;
        }
        if !in_string {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        prev = c;
    }
    depth
}

fn run_repl() {
    println!("Hayashi {VERSION}  — Applied Econometrics Language");
    println!("In honor of Fumio Hayashi. Type 'exit' or Ctrl-D to quit.\n");

    let mut interp = Interpreter::new();
    interp.load_plugins();

    let vars: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let helper = HayHelper::new(vars.clone());
    let mut rl = Editor::new().expect("failed to init readline");
    rl.set_helper(Some(helper));
    let _ = rl.load_history(HISTORY_FILE);

    let mut buf = String::new();
    let mut depth: i32 = 0;

    loop {
        *vars.borrow_mut() = interp.env.var_names();
        let prompt = if depth > 0 { "      > " } else { "hay> " };
        match rl.readline(prompt) {
            Ok(line) => {
                let trimmed = line.trim();
                if buf.is_empty() {
                    if trimmed.is_empty() {
                        continue;
                    }
                    if trimmed == "exit" || trimmed == "quit" {
                        break;
                    }
                }

                // input block: acumula até "end"
                let in_input = buf.lines().any(|l| {
                    let t = l.trim();
                    t.starts_with("input ") && !buf.contains("\nend")
                });
                if in_input {
                    buf.push('\n');
                    buf.push_str(&line);
                    if trimmed == "end" {
                        let _ = rl.add_history_entry(buf.trim());
                        match lang::run_source(&buf, &mut interp) {
                            Ok(()) => {}
                            Err(e) => eprintln!("error: {e}"),
                        }
                        buf.clear();
                        depth = 0;
                    }
                    continue;
                }

                buf.push_str(trimmed);
                buf.push('\n');
                depth += brace_depth(trimmed);

                if depth <= 0 {
                    depth = 0;
                    let source = buf.trim().to_string();
                    if !source.is_empty() {
                        let _ = rl.add_history_entry(&source);
                        match lang::run_source(&source, &mut interp) {
                            Ok(()) => {}
                            Err(e) => eprintln!("error: {e}"),
                        }
                    }
                    buf.clear();
                }
            }
            Err(ReadlineError::Interrupted) => {
                if !buf.is_empty() {
                    buf.clear();
                    depth = 0;
                    println!("(input cancelled)");
                } else {
                    println!("(Ctrl-C — use 'exit' to quit)");
                }
            }
            Err(ReadlineError::Eof) => {
                println!("exit");
                break;
            }
            Err(e) => {
                eprintln!("readline error: {e}");
                break;
            }
        }
    }

    let _ = rl.save_history(HISTORY_FILE);
}

fn print_help() {
    println!("Hayashi {VERSION}  — Applied Econometrics Language");
    println!();
    println!("USAGE:");
    println!("    hay              Start interactive REPL (multi-line)");
    println!("    hay script.hay    Run a script file");
    println!("    hay --version    Print version");
    println!();
    println!("ESTIMATORS:");
    println!("    ols/reg  logit  probit  iv  poisson  nbreg  tobit  qreg");
    println!("    fe  re  ab  sysgmm  pcse  xtgls  heckman  cox");
    println!("    lasso  ridge  elasticnet  garch  arima  var  vecm");
    println!();
    println!("POST-ESTIMATION:");
    println!("    test  nlcom  margins  bootstrap  esttab  estat  predict");
    println!();
    println!("DATA:");
    println!("    load (csv/tsv/json/dta/xlsx/parquet/sqlite/odbc)");
    println!("    generate  replace  drop  keep  dropna  encode  winsor");
    println!("    summarize  tabulate  ttest  correlate  list  describe");
    println!();
    println!("PACKAGES:");
    println!("    hay install user/repo    Install from GitHub");
    println!("    hay remove  name         Uninstall a package");
    println!("    hay list                 List installed packages");
    println!();
    println!("In REPL, type help() for full command list or help(cmd) for details.");
}

fn packages_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::Path::new(&home).join(".hay").join("packages")
}

fn pkg_install(spec: &str) {
    let (user, repo) = if let Some(pos) = spec.find('/') {
        (&spec[..pos], &spec[pos + 1..])
    } else {
        eprintln!("hay install: expected 'user/repo', got '{spec}'");
        std::process::exit(1);
    };

    let dest = packages_dir().join(user).join(repo);
    if dest.exists() {
        println!("{user}/{repo}: already installed at {}", dest.display());
        println!("  use 'hay remove {user}/{repo}' first to reinstall");
        return;
    }

    let api_url = format!("https://api.github.com/repos/{user}/{repo}/contents/");
    println!("Fetching {user}/{repo}...");

    let resp = match ureq::get(&api_url)
        .set("User-Agent", "hay")
        .set("Accept", "application/vnd.github.v3+json")
        .call()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("hay install: cannot reach GitHub API: {e}");
            std::process::exit(1);
        }
    };

    let body: String = resp.into_string().unwrap_or_default();
    let entries: Vec<GhEntry> = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("hay install: cannot parse GitHub response: {e}");
            std::process::exit(1);
        }
    };

    let dominated = |name: &str| -> bool {
        let lower = name.to_lowercase();
        lower.ends_with(".hay")
            || lower == "readme.md"
            || lower == "readme"
            || lower == "readme.txt"
            || lower == "license"
            || lower == "license.md"
            || lower == "license.txt"
            || lower == "licence"
            || lower == "licence.md"
    };

    let files: Vec<&GhEntry> = entries
        .iter()
        .filter(|e| e.r#type == "file" && e.download_url.is_some() && dominated(&e.name))
        .collect();

    let n_hy = files.iter().filter(|e| e.name.ends_with(".hay")).count();
    if n_hy == 0 {
        eprintln!("hay install: no .hay files found in {user}/{repo}");
        std::process::exit(1);
    }

    std::fs::create_dir_all(&dest).unwrap_or_else(|e| {
        eprintln!("hay install: cannot create {}: {e}", dest.display());
        std::process::exit(1);
    });

    let mut installed = 0;
    for file in &files {
        let url = file.download_url.as_ref().unwrap();
        print!("  {} ... ", file.name);
        match ureq::get(url).call() {
            Ok(resp) => {
                let content = resp.into_string().unwrap_or_default();
                let path = dest.join(&file.name);
                if std::fs::write(&path, &content).is_ok() {
                    println!("ok");
                    installed += 1;
                } else {
                    println!("write error");
                }
            }
            Err(e) => println!("download error: {e}"),
        }
    }

    println!(
        "Installed {user}/{repo}: {installed} file(s) → {}",
        dest.display()
    );
    println!("  use: import(\"{user}/{repo}/module\")");
}

fn pkg_remove(spec: &str) {
    let dest = if let Some(pos) = spec.find('/') {
        packages_dir().join(&spec[..pos]).join(&spec[pos + 1..])
    } else {
        packages_dir().join(spec)
    };
    if !dest.exists() {
        eprintln!("hay remove: package '{spec}' not installed");
        std::process::exit(1);
    }
    std::fs::remove_dir_all(&dest).unwrap_or_else(|e| {
        eprintln!("hay remove: cannot remove {}: {e}", dest.display());
        std::process::exit(1);
    });
    // remove empty user dir
    if let Some(parent) = dest.parent() {
        let _ = std::fs::remove_dir(parent);
    }
    println!("Removed {spec}");
}

fn pkg_list() {
    let dir = packages_dir();
    if !dir.is_dir() {
        println!("No packages installed.");
        return;
    }
    let mut found = false;
    let mut users: Vec<_> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .collect()
        })
        .unwrap_or_default();
    users.sort_by_key(|e: &std::fs::DirEntry| e.file_name());

    for user_entry in &users {
        let user = user_entry.file_name();
        let mut repos: Vec<_> = std::fs::read_dir(user_entry.path())
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .collect()
            })
            .unwrap_or_default();
        repos.sort_by_key(|e: &std::fs::DirEntry| e.file_name());

        for repo_entry in &repos {
            if !found {
                println!("Installed packages (~/.hay/packages/):\n");
                found = true;
            }
            let repo = repo_entry.file_name();
            let n_hy = std::fs::read_dir(repo_entry.path())
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("hy"))
                        .count()
                })
                .unwrap_or(0);
            println!(
                "  {}/{}  ({} file{})",
                user.to_string_lossy(),
                repo.to_string_lossy(),
                n_hy,
                if n_hy == 1 { "" } else { "s" }
            );
        }
    }
    if !found {
        println!("No packages installed.");
    }
}

#[derive(serde::Deserialize)]
struct GhEntry {
    name: String,
    r#type: String,
    download_url: Option<String>,
}
