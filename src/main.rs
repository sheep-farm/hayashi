mod io;
mod lang;

use lang::interpreter::Interpreter;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const HISTORY_FILE: &str = ".hayashi_history";

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    let args_clean: Vec<&str> = args.iter()
        .map(String::as_str)
        .filter(|a| *a != "--verbose" && *a != "-v")
        .collect();

    match args_clean.get(1).copied() {
        Some("--version") | Some("-V") => {
            println!("hayashi {VERSION}");
            return;
        }
        Some("--help") | Some("-h") => {
            print_help();
            return;
        }
        Some("-") => {
            use std::io::Read;
            let mut src = String::new();
            std::io::stdin().read_to_string(&mut src).expect("failed to read stdin");
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
                eprintln!("Usage: hayashi install user/repo");
                std::process::exit(1);
            });
            pkg_install(pkg);
            return;
        }
        Some("remove") | Some("uninstall") => {
            let pkg = args_clean.get(2).unwrap_or_else(|| {
                eprintln!("Usage: hayashi remove package_name");
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
            eprintln!("hayashi: unknown argument '{unknown}'");
            eprintln!("Usage: hayashi [script.hy | - | install | remove | list]");
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
            eprintln!("hayashi: cannot read '{path}': {e}");
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
        if c == '"' && prev != '\\' { in_string = !in_string; }
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
    let mut rl = DefaultEditor::new().expect("failed to init readline");
    let _ = rl.load_history(HISTORY_FILE);

    let mut buf = String::new();
    let mut depth: i32 = 0;

    loop {
        let prompt = if depth > 0 { "      > " } else { "hayashi> " };
        match rl.readline(prompt) {
            Ok(line) => {
                let trimmed = line.trim();
                if buf.is_empty() {
                    if trimmed.is_empty() { continue; }
                    if trimmed == "exit" || trimmed == "quit" { break; }
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
    println!("    hayashi              Start interactive REPL (multi-line)");
    println!("    hayashi script.hy    Run a script file");
    println!("    hayashi --version    Print version");
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
    println!("    hayashi install user/repo    Install from GitHub");
    println!("    hayashi remove  name         Uninstall a package");
    println!("    hayashi list                 List installed packages");
    println!();
    println!("In REPL, type help() for full command list or help(cmd) for details.");
}

fn packages_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::Path::new(&home).join(".hayashi").join("packages")
}

fn pkg_install(spec: &str) {
    let (user, repo) = if let Some(pos) = spec.find('/') {
        (&spec[..pos], &spec[pos+1..])
    } else {
        eprintln!("hayashi install: expected 'user/repo', got '{spec}'");
        std::process::exit(1);
    };

    let dest = packages_dir().join(repo);
    if dest.exists() {
        println!("{repo}: already installed at {}", dest.display());
        println!("  use 'hayashi remove {repo}' first to reinstall");
        return;
    }

    let api_url = format!("https://api.github.com/repos/{user}/{repo}/contents/");
    println!("Fetching {user}/{repo}...");

    let resp = match ureq::get(&api_url)
        .set("User-Agent", "hayashi")
        .set("Accept", "application/vnd.github.v3+json")
        .call()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("hayashi install: cannot reach GitHub API: {e}");
            std::process::exit(1);
        }
    };

    let body: String = resp.into_string().unwrap_or_default();
    let entries: Vec<GhEntry> = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("hayashi install: cannot parse GitHub response: {e}");
            std::process::exit(1);
        }
    };

    let hy_files: Vec<&GhEntry> = entries.iter()
        .filter(|e| e.name.ends_with(".hy") && e.r#type == "file" && e.download_url.is_some())
        .collect();

    if hy_files.is_empty() {
        eprintln!("hayashi install: no .hy files found in {user}/{repo}");
        std::process::exit(1);
    }

    std::fs::create_dir_all(&dest).unwrap_or_else(|e| {
        eprintln!("hayashi install: cannot create {}: {e}", dest.display());
        std::process::exit(1);
    });

    let mut installed = 0;
    for file in &hy_files {
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

    println!("Installed {repo}: {installed} file(s) → {}", dest.display());
    println!("  use: import(\"{repo}/module\")");
}

fn pkg_remove(name: &str) {
    let dest = packages_dir().join(name);
    if !dest.exists() {
        eprintln!("hayashi remove: package '{name}' not installed");
        std::process::exit(1);
    }
    std::fs::remove_dir_all(&dest).unwrap_or_else(|e| {
        eprintln!("hayashi remove: cannot remove {}: {e}", dest.display());
        std::process::exit(1);
    });
    println!("Removed {name}");
}

fn pkg_list() {
    let dir = packages_dir();
    if !dir.is_dir() {
        println!("No packages installed.");
        return;
    }
    let mut entries: Vec<_> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd.filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect(),
        Err(_) => { println!("No packages installed."); return; }
    };
    if entries.is_empty() {
        println!("No packages installed.");
        return;
    }
    entries.sort_by_key(|e| e.file_name());
    println!("Installed packages (~/.hayashi/packages/):\n");
    for entry in entries {
        let name = entry.file_name();
        let n_files = std::fs::read_dir(entry.path())
            .map(|rd| rd.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("hy"))
                .count())
            .unwrap_or(0);
        println!("  {:<20} ({} file{})", name.to_string_lossy(), n_files, if n_files == 1 { "" } else { "s" });
    }
}

#[derive(serde::Deserialize)]
struct GhEntry {
    name: String,
    r#type: String,
    download_url: Option<String>,
}
