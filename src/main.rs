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
        Some(path) if !path.starts_with('-') => {
            run_script(path, verbose);
            return;
        }
        Some(unknown) => {
            eprintln!("hayashi: unknown argument '{unknown}'");
            eprintln!("Usage: hayashi [script.hy | -]");
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
    println!("    load (csv/tsv/json/dta/xlsx/xls/ods/sqlite)");
    println!("    generate  replace  drop  keep  dropna  encode  winsor");
    println!("    summarize  tabulate  ttest  correlate  list  describe");
    println!();
    println!("In REPL, type help() for full command list or help(cmd) for details.");
}
