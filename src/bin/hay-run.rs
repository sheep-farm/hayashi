//! Headless script runner for Hayashi.
//!
//! This binary runs a `.hay` script and exits. It has no REPL, no DAP, no
//! network/update/package commands, and no interactive features. It is meant
//! for SaaS/CI/benchmark deployments where only `hay <script>` is needed.

use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: hay-run <script.hay>");
        process::exit(if args.len() < 2 { 1 } else { 0 });
    }

    let path = &args[1];
    if path == "-" {
        run_stdin();
    } else {
        run_file(path);
    }
}

fn run_file(path: &str) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("hay-run: cannot read '{path}': {e}");
            process::exit(1);
        }
    };

    let mut interp = hayashi_lang::Interpreter::new();
    interp.load_plugins();

    let p = std::path::Path::new(path);
    if let Err(e) = hayashi_lang::lang::run_source_with_path(&src, &mut interp, Some(p)) {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

fn run_stdin() {
    use std::io::Read;

    let mut src = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut src) {
        eprintln!("hay-run: failed to read stdin: {e}");
        process::exit(1);
    }

    let mut interp = hayashi_lang::Interpreter::new();
    interp.load_plugins();

    if let Err(e) = hayashi_lang::lang::run_source(&src, &mut interp) {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
