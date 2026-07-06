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
    "date",
    "datetime",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "dow",
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
    let yes = args.iter().any(|a| a == "--yes" || a == "-y");

    let args_clean: Vec<&str> = args
        .iter()
        .map(String::as_str)
        .filter(|a| *a != "--verbose" && *a != "-v" && *a != "--yes" && *a != "-y")
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
                eprintln!("Usage: hay install user/repo [version] [-y]");
                std::process::exit(1);
            });
            let version = args_clean.get(3).copied();
            pkg_install_internal(pkg, version, yes);
            return;
        }
        Some("update") => {
            let pkg_opt = args_clean.get(2).copied();
            pkg_update(pkg_opt, yes);
            return;
        }
        Some("check-plugin") => {
            let pkg_opt = args_clean.get(2).copied();
            pkg_check_plugin(pkg_opt);
            return;
        }
        Some("validate") => {
            run_validation(&args_clean[2..]);
            return;
        }
        Some("dist-update") => {
            dist_update(&args_clean[2..]);
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
            eprintln!(
                "Usage: hay [script.hay | - | install | remove | list | update | check-plugin | validate | dist-update]"
            );
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

/// Runs the empirical validation programme by invoking `validation/run.py`.
fn run_validation(args: &[&str]) {
    // Prefer the validation directory relative to the current working directory
    // (typical for development and CI), then fall back to the executable's
    // directory (typical for a self-contained installation).
    let (run_py, hay_dir) = std::env::current_dir()
        .ok()
        .map(|d| d.join("validation").join("run.py"))
        .filter(|p| p.exists())
        .map(|p| {
            let base = p.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf());
            (p, base)
        })
        .or_else(|| {
            let exe = std::env::current_exe().ok()?;
            let bin_dir = exe.parent()?;
            let candidate = bin_dir.join("validation").join("run.py");
            if candidate.exists() {
                let base = Some(bin_dir.to_path_buf());
                Some((candidate, base))
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            eprintln!("hay: validation programme not found");
            eprintln!(
                "       Looked for ./validation/run.py and <hay-binary>/validation/run.py."
            );
            eprintln!(
                "       Run this from a checkout/installation that includes the validation/ directory."
            );
            std::process::exit(1);
        });

    let python = if let Some(venv) = hay_dir.as_deref().and_then(|p| p.parent()).map(|p| {
        p.join("validation")
            .join(".venv")
            .join("bin")
            .join("python")
    }) {
        if venv.exists() {
            venv
        } else if std::process::Command::new("python")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            std::path::PathBuf::from("python")
        } else {
            std::path::PathBuf::from("python3")
        }
    } else {
        std::path::PathBuf::from("python3")
    };

    let mut cmd = std::process::Command::new(python);
    cmd.arg(&run_py);
    cmd.args(args);
    if let Some(dir) = hay_dir {
        cmd.current_dir(dir);
    }
    let status = cmd
        .status()
        .expect("hay: failed to spawn validation runner");

    std::process::exit(status.code().unwrap_or(1));
}

/// Print dist-update subcommand help.
fn dist_update_help() {
    println!("Usage: hay dist-update [--help] [--check]");
    println!();
    println!("Options:");
    println!("  --help, -h  Show this help message and exit");
    println!("  --check     Report whether a newer release is available without");
    println!("              downloading or replacing the current binary");
}

/// Check GitHub for the latest release and return the version string if it is
/// newer than the current one. Returns Ok(None) when already up to date.
fn check_latest_release() -> Result<Option<String>, String> {
    let release_url = "https://api.github.com/repos/sheep-farm/hayashi/releases/latest";
    let release_resp = ureq::get(release_url)
        .set("User-Agent", "hay")
        .set("Accept", "application/vnd.github.v3+json")
        .call()
        .map_err(|e| format!("cannot fetch latest release: {e}"))?;

    let release_body: String = release_resp.into_string().unwrap_or_default();
    let release: GhRelease = serde_json::from_str(&release_body)
        .map_err(|e| format!("cannot parse release payload: {e}"))?;

    let remote_version = release.tag_name.trim_start_matches('v').to_string();
    if !is_newer_version(&remote_version, VERSION) {
        return Ok(None);
    }
    Ok(Some(remote_version))
}

/// Download and replace the current binary with the given release version.
fn dist_update_install(remote_version: &str) {
    let target = current_target_triple();
    let (asset_ext, archive_cmd) = dist_asset_kind();
    let asset_name = format!("hay-v{remote_version}-{target}.{asset_ext}");

    let release_url = "https://api.github.com/repos/sheep-farm/hayashi/releases/latest";
    let release_resp = match ureq::get(release_url)
        .set("User-Agent", "hay")
        .set("Accept", "application/vnd.github.v3+json")
        .call()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("hay dist-update: cannot fetch latest release: {e}");
            std::process::exit(1);
        }
    };

    let release_body: String = release_resp.into_string().unwrap_or_default();
    let release: GhRelease = serde_json::from_str(&release_body).unwrap_or_else(|e| {
        eprintln!("hay dist-update: cannot parse release payload: {e}");
        std::process::exit(1);
    });

    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .unwrap_or_else(|| {
            eprintln!("hay dist-update: no asset found for {asset_name}");
            std::process::exit(1);
        });

    let exe_path = std::env::current_exe().unwrap_or_else(|e| {
        eprintln!("hay dist-update: cannot locate current executable: {e}");
        std::process::exit(1);
    });

    let tmp_dir = std::env::temp_dir().join(format!("hay-dist-update-{remote_version}"));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap_or_else(|e| {
        eprintln!("hay dist-update: cannot create temp dir: {e}");
        std::process::exit(1);
    });

    let archive_path = tmp_dir.join(&asset_name);
    println!("hay dist-update: downloading {} ...", asset.name);
    match ureq::get(&asset.browser_download_url).call() {
        Ok(resp) => {
            let mut reader = resp.into_reader();
            let mut file = std::fs::File::create(&archive_path).unwrap();
            std::io::copy(&mut reader, &mut file).unwrap_or_else(|e| {
                eprintln!("hay dist-update: download failed: {e}");
                std::process::exit(1);
            });
        }
        Err(e) => {
            eprintln!("hay dist-update: download failed: {e}");
            std::process::exit(1);
        }
    }

    let extract_dir = tmp_dir.join("extract");
    std::fs::create_dir_all(&extract_dir).unwrap();
    if let Err(e) = archive_cmd(&archive_path, &extract_dir) {
        eprintln!("hay dist-update: cannot extract archive: {e}");
        std::process::exit(1);
    }

    let new_bin = find_extracted_bin(&extract_dir).unwrap_or_else(|| {
        eprintln!("hay dist-update: no hay binary found in downloaded archive");
        std::process::exit(1);
    });

    println!("hay dist-update: replacing {} ...", exe_path.display());

    let is_windows = std::env::consts::OS == "windows";
    let backup_path = exe_path.with_extension(if is_windows { "exe.old" } else { "old" });
    let _ = std::fs::remove_file(&backup_path);

    std::fs::rename(&exe_path, &backup_path).unwrap_or_else(|e| {
        eprintln!("hay dist-update: cannot backup current executable: {e}");
        std::process::exit(1);
    });

    std::fs::copy(&new_bin, &exe_path).unwrap_or_else(|e| {
        eprintln!("hay dist-update: cannot install new binary: {e}");
        let _ = std::fs::rename(&backup_path, &exe_path);
        std::process::exit(1);
    });

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&exe_path).unwrap().permissions();
        perms.set_mode(0o755);
        let _ = std::fs::set_permissions(&exe_path, perms);
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);

    if is_windows {
        println!("hay dist-update: installed {remote_version}. Please restart hay.");
    } else {
        println!("hay dist-update: installed {remote_version}. Run `hay --version` to verify.");
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DistUpdateMode {
    Help,
    Check,
    Install,
}

/// Parse dist-update arguments into a safe mode or fail on unknown flags.
fn parse_dist_update_args(argv: &[&str]) -> Result<DistUpdateMode, String> {
    if argv.iter().any(|a| *a == "--help" || *a == "-h") {
        return Ok(DistUpdateMode::Help);
    }

    let mut check = false;
    for arg in argv {
        if *arg == "--check" {
            check = true;
        } else if arg.starts_with('-') {
            return Err(format!("unknown flag '{arg}'"));
        } else {
            return Err(format!("unexpected positional argument '{arg}'"));
        }
    }

    if check {
        Ok(DistUpdateMode::Check)
    } else {
        Ok(DistUpdateMode::Install)
    }
}

/// Parse dist-update arguments and dispatch to help, check, or install mode.
fn dist_update(argv: &[&str]) {
    match parse_dist_update_args(argv) {
        Ok(DistUpdateMode::Help) => {
            dist_update_help();
        }
        Ok(DistUpdateMode::Check) => {
            println!("hay dist-update: current version {VERSION}");
            match check_latest_release() {
                Ok(Some(remote_version)) => {
                    println!("hay dist-update: newer release {remote_version} available");
                }
                Ok(None) => {
                    println!("hay dist-update: already up to date");
                }
                Err(e) => {
                    eprintln!("hay dist-update: {e}");
                    std::process::exit(1);
                }
            }
        }
        Ok(DistUpdateMode::Install) => {
            println!("hay dist-update: current version {VERSION}");
            match check_latest_release() {
                Ok(Some(remote_version)) => {
                    println!("hay dist-update: newer release {remote_version} available");
                    dist_update_install(&remote_version);
                }
                Ok(None) => {
                    println!("hay dist-update: already up to date");
                }
                Err(e) => {
                    eprintln!("hay dist-update: {e}");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("hay dist-update: {e}");
            dist_update_help();
            std::process::exit(1);
        }
    }
}

type Extractor = fn(&std::path::Path, &std::path::Path) -> Result<(), String>;

/// Returns the archive extension and an extractor closure for the current platform.
fn dist_asset_kind() -> (&'static str, Extractor) {
    match std::env::consts::OS {
        "windows" => (
            "zip",
            |archive: &std::path::Path, dest: &std::path::Path| {
                let status = std::process::Command::new("powershell")
                    .args([
                        "-Command",
                        "Expand-Archive",
                        "-Path",
                        &archive.to_string_lossy(),
                        "-DestinationPath",
                        &dest.to_string_lossy(),
                        "-Force",
                    ])
                    .status()
                    .map_err(|e| e.to_string())?;
                if status.success() {
                    Ok(())
                } else {
                    Err("Expand-Archive failed".to_string())
                }
            },
        ),
        _ => (
            "tar.gz",
            |archive: &std::path::Path, dest: &std::path::Path| {
                let status = std::process::Command::new("tar")
                    .args([
                        "xzf",
                        &archive.to_string_lossy(),
                        "-C",
                        &dest.to_string_lossy(),
                    ])
                    .status()
                    .map_err(|e| e.to_string())?;
                if status.success() {
                    Ok(())
                } else {
                    Err("tar extraction failed".to_string())
                }
            },
        ),
    }
}

/// Searches the extracted archive for the new hay binary.
fn find_extracted_bin(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let name = if std::env::consts::OS == "windows" {
        "hay.exe"
    } else {
        "hay"
    };
    let mut queue = vec![dir.to_path_buf()];
    while let Some(current) = queue.pop() {
        if let Ok(entries) = std::fs::read_dir(&current) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    queue.push(path);
                } else if path.file_name().map(|n| n == name).unwrap_or(false) {
                    return Some(path);
                }
            }
        }
    }
    None
}

/// Compares two semantic versions, ignoring pre-release labels so that
/// 0.2.7 > 0.2.7-dev and 0.2.7 > 0.2.6. Returns true if remote is newer.
fn is_newer_version(remote: &str, current: &str) -> bool {
    fn parse(v: &str) -> (Vec<u32>, Option<&str>) {
        let v = v.trim_start_matches('v');
        let (num, pre) = v
            .split_once('-')
            .map(|(n, p)| (n, Some(p)))
            .unwrap_or((v, None));
        let nums: Vec<u32> = num
            .split('.')
            .filter_map(|s| s.parse::<u32>().ok())
            .collect();
        (nums, pre)
    }

    let (r_nums, r_pre) = parse(remote);
    let (c_nums, c_pre) = parse(current);

    for (a, b) in r_nums.iter().zip(c_nums.iter()) {
        match a.cmp(b) {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {}
        }
    }

    if r_nums.len() != c_nums.len() {
        return r_nums.len() > c_nums.len();
    }

    // A release without a pre-release tag is newer than one with it.
    match (r_pre, c_pre) {
        (None, Some(_)) => true,
        (Some(_), None) => false,
        (None, None) => false,
        (Some(a), Some(b)) => a > b,
    }
}

/// Calcula a profundidade de delimitadores abertos numa linha para o REPL.
/// Conta {, [, ( como +1 e }, ], ) como -1, ignorando o interior de strings.
fn open_depth(s: &str) -> i32 {
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut prev = '\0';
    for c in s.chars() {
        if c == '"' && prev != '\\' {
            in_string = !in_string;
        }
        if !in_string {
            match c {
                '{' | '[' | '(' => depth += 1,
                '}' | ']' | ')' => depth -= 1,
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
        let is_continuation = depth > 0 || buf.trim_end().ends_with("|>");
        let prompt = if is_continuation { "      > " } else { "hay> " };
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

                // input block: accumulate until "end"
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
                // depth tracks unclosed delimiters: {}, [], ()
                depth += open_depth(trimmed);

                // Keep accumulating if:
                // (a) there are open delimiters (depth > 0), OR
                // (b) the buffer (without trailing spaces) ends with |>
                let buf_trimmed = buf.trim_end();
                let trailing_pipe = buf_trimmed.ends_with("|>");
                if depth > 0 || trailing_pipe {
                    continue;
                }

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
    println!("    hay install user/repo    Install from GitHub (-y to bypass overwrite prompt)");
    println!("    hay remove  user/repo    Uninstall a package");
    println!("    hay list                 List installed packages");
    println!("    hay update [user/repo]   Update package(s) (-y to bypass prompt)");
    println!("    hay check-plugin [name]  Check integrity/version with remote repository");
    println!("    hay validate [options]   Run the empirical validation programme (R/Python)");
    println!("    hay dist-update          Check and install the latest hay release from GitHub");
    println!();
    println!("In REPL, type help() for full command list or help(cmd) for details.");
}

fn packages_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::Path::new(&home).join(".hay").join("packages")
}

fn is_pkg_installed(user: &str, repo: &str) -> Option<std::path::PathBuf> {
    let dir = packages_dir().join(user).join(repo);
    if dir.exists() && dir.is_dir() {
        return Some(dir);
    }
    let ext = current_target_ext();
    let file = packages_dir().join(user).join(format!("{repo}.{ext}"));
    if file.exists() && file.is_file() {
        return Some(file);
    }
    None
}

#[allow(dead_code)]
fn pkg_install(spec: &str) {
    pkg_install_internal(spec, None, false);
}

fn pkg_install_internal(spec: &str, version: Option<&str>, force_overwrite: bool) {
    let (user, repo) = if let Some(pos) = spec.find('/') {
        (&spec[..pos], &spec[pos + 1..])
    } else {
        eprintln!("hay install: expected 'user/repo', got '{spec}'");
        std::process::exit(1);
    };

    let version_tag = version.and_then(|v| {
        let trimmed = v.trim();
        if trimmed.is_empty() || trimmed == "latest" {
            None
        } else if trimmed.starts_with('v') {
            Some(trimmed.to_string())
        } else {
            Some(format!("v{trimmed}"))
        }
    });

    if !force_overwrite {
        if let Some(installed_path) = is_pkg_installed(user, repo) {
            print!(
                "Package {}/{} is already installed at {}. Overwrite? (y/N): ",
                user,
                repo,
                installed_path.display()
            );
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_ok() {
                let trimmed = input.trim().to_lowercase();
                if trimmed != "y" && trimmed != "yes" {
                    println!("Installation cancelled.");
                    return;
                }
            } else {
                println!("Installation cancelled.");
                return;
            }
        }
    }

    let dest = packages_dir().join(user).join(repo);
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
        println!("No .hay scripts found. Checking for native/WASM releases...");
        let release_url = match &version_tag {
            Some(tag) => format!("https://api.github.com/repos/{user}/{repo}/releases/tags/{tag}"),
            None => format!("https://api.github.com/repos/{user}/{repo}/releases/latest"),
        };

        let release_resp = match ureq::get(&release_url)
            .set("User-Agent", "hay")
            .set("Accept", "application/vnd.github.v3+json")
            .call()
        {
            Ok(r) => r,
            Err(e) => {
                if let Some(tag) = &version_tag {
                    eprintln!("hay install: release {tag} not found for {user}/{repo}: {e}");
                } else {
                    eprintln!(
                        "hay install: no scripts or native releases found for {user}/{repo}: {e}"
                    );
                }
                std::process::exit(1);
            }
        };

        let release_body: String = release_resp.into_string().unwrap_or_default();
        let release: GhRelease = serde_json::from_str(&release_body).unwrap_or_else(|e| {
            eprintln!("hay install: cannot parse release payload: {e}");
            std::process::exit(1);
        });

        let target = current_target_triple();
        let ext = current_target_ext();

        let matching_asset = release
            .assets
            .iter()
            .find(|asset| asset.name.contains(target) && asset.name.ends_with(ext));

        if let Some(asset) = matching_asset {
            println!("Found binary release for {target}: {}", asset.name);
            let parent_dir = packages_dir().join(user);
            std::fs::create_dir_all(&parent_dir).unwrap();
            let dest_file = parent_dir.join(format!("{repo}.{ext}"));

            print!("Downloading {} ... ", asset.name);
            match ureq::get(&asset.browser_download_url).call() {
                Ok(resp) => {
                    let mut reader = resp.into_reader();
                    let mut out_file = std::fs::File::create(&dest_file).unwrap();
                    if std::io::copy(&mut reader, &mut out_file).is_ok() {
                        println!("ok");

                        let meta = PkgMetadata {
                            user: user.to_string(),
                            repo: repo.to_string(),
                            version: release.tag_name.clone(),
                            installed_at: chrono::Utc::now().to_rfc3339(),
                            pkg_type: "native".to_string(),
                        };
                        write_pkg_metadata(&meta);

                        println!(
                            "Successfully installed native plugin {user}/{repo} at {}",
                            dest_file.display()
                        );
                        println!("  use: import(\"{user}/{repo}\")");
                        return;
                    } else {
                        println!("write error");
                    }
                }
                Err(e) => println!("download error: {e}"),
            }
        } else {
            eprintln!("hay install: no compatible release asset found for {target}");
            std::process::exit(1);
        }
        return;
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

    let commit_url = format!("https://api.github.com/repos/{user}/{repo}/commits");
    let mut version = "unknown".to_string();
    if let Ok(c_resp) = ureq::get(&commit_url)
        .set("User-Agent", "hay")
        .set("Accept", "application/vnd.github.v3+json")
        .call()
    {
        if let Ok(c_body) = c_resp.into_string() {
            if let Ok(commits) = serde_json::from_str::<Vec<GhCommitInfo>>(&c_body) {
                if let Some(first) = commits.first() {
                    version = first.sha.clone();
                }
            }
        }
    }

    let meta = PkgMetadata {
        user: user.to_string(),
        repo: repo.to_string(),
        version,
        installed_at: chrono::Utc::now().to_rfc3339(),
        pkg_type: "script".to_string(),
    };
    write_pkg_metadata(&meta);

    println!(
        "Installed {user}/{repo}: {installed} file(s) → {}",
        dest.display()
    );
    println!("  use: import(\"{user}/{repo}/module\")");
}

fn pkg_remove(spec: &str) {
    let (user, repo) = if let Some(pos) = spec.find('/') {
        (&spec[..pos], &spec[pos + 1..])
    } else {
        eprintln!("hay remove: expected 'user/repo', got '{spec}'");
        std::process::exit(1);
    };

    let dir = packages_dir().join(user).join(repo);
    let ext = current_target_ext();
    let file = packages_dir().join(user).join(format!("{repo}.{ext}"));
    let meta_file = pkg_metadata_path(user, repo);

    let mut removed = false;

    if dir.exists() && dir.is_dir() {
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| {
            eprintln!("hay remove: cannot remove {}: {e}", dir.display());
            std::process::exit(1);
        });
        removed = true;
    }

    if file.exists() && file.is_file() {
        std::fs::remove_file(&file).unwrap_or_else(|e| {
            eprintln!("hay remove: cannot remove {}: {e}", file.display());
            std::process::exit(1);
        });
        removed = true;
    }

    if meta_file.exists() {
        let _ = std::fs::remove_file(&meta_file);
    }

    if !removed {
        eprintln!("hay remove: package '{spec}' not installed");
        std::process::exit(1);
    }

    let user_dir = packages_dir().join(user);
    if user_dir.exists() {
        let _ = std::fs::remove_dir(&user_dir);
    }

    println!("Removed {spec}");
}

fn migrate_legacy_packages() {
    let dir = packages_dir();
    if !dir.is_dir() {
        return;
    }
    if let Ok(users) = std::fs::read_dir(&dir) {
        for user_entry in users.filter_map(|e| e.ok()) {
            if user_entry.path().is_dir() {
                let user = user_entry.file_name().to_string_lossy().to_string();
                if let Ok(repos) = std::fs::read_dir(user_entry.path()) {
                    for repo_entry in repos.filter_map(|e| e.ok()) {
                        let path = repo_entry.path();
                        let repo_name = repo_entry.file_name().to_string_lossy().to_string();

                        if path.is_dir() {
                            let metadata_file = pkg_metadata_path(&user, &repo_name);
                            if !metadata_file.exists() {
                                let meta = PkgMetadata {
                                    user: user.clone(),
                                    repo: repo_name,
                                    version: "unknown".to_string(),
                                    installed_at: "unknown".to_string(),
                                    pkg_type: "script".to_string(),
                                };
                                write_pkg_metadata(&meta);
                            }
                        } else if path.is_file() {
                            let ext = path
                                .extension()
                                .and_then(|x| x.to_str())
                                .unwrap_or("")
                                .to_lowercase();
                            if matches!(ext.as_str(), "so" | "dll" | "dylib" | "wasm") {
                                let clean_name =
                                    repo_name.trim_end_matches(&format!(".{ext}")).to_string();
                                let metadata_file = pkg_metadata_path(&user, &clean_name);
                                if !metadata_file.exists() {
                                    let meta = PkgMetadata {
                                        user: user.clone(),
                                        repo: clean_name,
                                        version: "unknown".to_string(),
                                        installed_at: "unknown".to_string(),
                                        pkg_type: "native".to_string(),
                                    };
                                    write_pkg_metadata(&meta);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn get_installed_packages() -> Vec<PkgMetadata> {
    let mut pkgs = Vec::new();
    let dir = packages_dir();
    if !dir.is_dir() {
        return pkgs;
    }
    if let Ok(users) = std::fs::read_dir(&dir) {
        for user_entry in users.filter_map(|e| e.ok()) {
            if user_entry.path().is_dir() {
                if let Ok(repos) = std::fs::read_dir(user_entry.path()) {
                    for repo_entry in repos.filter_map(|e| e.ok()) {
                        let path = repo_entry.path();
                        if path.is_file() {
                            let name = repo_entry.file_name().to_string_lossy().to_string();
                            if name.ends_with(".metadata.json") {
                                if let Ok(content) = std::fs::read_to_string(&path) {
                                    if let Ok(meta) = serde_json::from_str::<PkgMetadata>(&content)
                                    {
                                        pkgs.push(meta);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    pkgs
}

fn check_pkg_integrity(meta: &PkgMetadata) -> Result<(String, bool), String> {
    if meta.pkg_type == "native" {
        let release_url = format!(
            "https://api.github.com/repos/{}/{}/releases/latest",
            meta.user, meta.repo
        );
        let resp = ureq::get(&release_url)
            .set("User-Agent", "hay")
            .set("Accept", "application/vnd.github.v3+json")
            .call()
            .map_err(|e| e.to_string())?;

        let body: String = resp.into_string().map_err(|e| e.to_string())?;
        let release: GhRelease = serde_json::from_str(&body).map_err(|e| e.to_string())?;

        let up_to_date = meta.version == release.tag_name;
        Ok((release.tag_name, up_to_date))
    } else {
        let commit_url = format!(
            "https://api.github.com/repos/{}/{}/commits",
            meta.user, meta.repo
        );
        let resp = ureq::get(&commit_url)
            .set("User-Agent", "hay")
            .set("Accept", "application/vnd.github.v3+json")
            .call()
            .map_err(|e| e.to_string())?;

        let body: String = resp.into_string().map_err(|e| e.to_string())?;
        let commits: Vec<GhCommitInfo> = serde_json::from_str(&body).map_err(|e| e.to_string())?;

        if let Some(first) = commits.first() {
            let up_to_date = meta.version == first.sha;
            Ok((first.sha.clone(), up_to_date))
        } else {
            Err("No commits found in remote repository".to_string())
        }
    }
}

fn pkg_check_plugin(spec_opt: Option<&str>) {
    migrate_legacy_packages();

    if let Some(spec) = spec_opt {
        let (user, repo) = if let Some(pos) = spec.find('/') {
            (&spec[..pos], &spec[pos + 1..])
        } else {
            eprintln!("hay check-plugin: expected 'user/repo', got '{spec}'");
            std::process::exit(1);
        };

        match read_pkg_metadata(user, repo) {
            Some(meta) => {
                println!("Checking {}/{} ...", user, repo);
                match check_pkg_integrity(&meta) {
                    Ok((remote_ver, up_to_date)) => {
                        if up_to_date {
                            println!(
                                "  {}/{} is UP TO DATE (version {})",
                                user, repo, meta.version
                            );
                        } else {
                            println!(
                                "  {}/{} has updates available (local: {}, remote: {})",
                                user, repo, meta.version, remote_ver
                            );
                        }
                    }
                    Err(e) => {
                        println!("  {}/{} check failed: {}", user, repo, e);
                    }
                }
            }
            None => {
                eprintln!("hay check-plugin: package '{spec}' not installed");
                std::process::exit(1);
            }
        }
    } else {
        let pkgs = get_installed_packages();
        if pkgs.is_empty() {
            println!("No packages installed.");
            return;
        }

        println!("Checking installed packages:");
        for meta in pkgs {
            println!("Checking {}/{} ...", meta.user, meta.repo);
            match check_pkg_integrity(&meta) {
                Ok((remote_ver, up_to_date)) => {
                    if up_to_date {
                        println!(
                            "  {}/{} is UP TO DATE (version {})",
                            meta.user, meta.repo, meta.version
                        );
                    } else {
                        println!(
                            "  {}/{} has updates available (local: {}, remote: {})",
                            meta.user, meta.repo, meta.version, remote_ver
                        );
                    }
                }
                Err(e) => {
                    println!("  {}/{} check failed: {}", meta.user, meta.repo, e);
                }
            }
        }
    }
}

fn pkg_update(spec_opt: Option<&str>, auto_confirm: bool) {
    migrate_legacy_packages();

    if let Some(spec) = spec_opt {
        let (user, repo) = if let Some(pos) = spec.find('/') {
            (&spec[..pos], &spec[pos + 1..])
        } else {
            eprintln!("hay update: expected 'user/repo', got '{spec}'");
            std::process::exit(1);
        };

        let meta = match read_pkg_metadata(user, repo) {
            Some(m) => m,
            None => {
                eprintln!("hay update: package '{spec}' not installed");
                std::process::exit(1);
            }
        };

        println!("Checking updates for {}/{} ...", user, repo);
        match check_pkg_integrity(&meta) {
            Ok((remote_ver, up_to_date)) => {
                if up_to_date {
                    println!(
                        "Package {}/{} is already up to date (version {}).",
                        user, repo, meta.version
                    );
                    return;
                }

                let confirm = if auto_confirm {
                    true
                } else {
                    print!(
                        "Update package {}/{} to {}? (y/N): ",
                        user, repo, remote_ver
                    );
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    let mut input = String::new();
                    if std::io::stdin().read_line(&mut input).is_ok() {
                        let trimmed = input.trim().to_lowercase();
                        trimmed == "y" || trimmed == "yes"
                    } else {
                        false
                    }
                };

                if confirm {
                    pkg_install_internal(spec, None, true);
                } else {
                    println!("Update cancelled.");
                }
            }
            Err(e) => {
                println!("Check failed for {}/{}: {}", user, repo, e);
                let confirm = if auto_confirm {
                    true
                } else {
                    print!("Attempt update for {}/{} anyway? (y/N): ", user, repo);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    let mut input = String::new();
                    if std::io::stdin().read_line(&mut input).is_ok() {
                        let trimmed = input.trim().to_lowercase();
                        trimmed == "y" || trimmed == "yes"
                    } else {
                        false
                    }
                };
                if confirm {
                    pkg_install_internal(spec, None, true);
                } else {
                    println!("Update cancelled.");
                }
            }
        }
    } else {
        let pkgs = get_installed_packages();
        if pkgs.is_empty() {
            println!("No packages installed.");
            return;
        }

        println!("Checking all packages for updates vistas...");
        for meta in pkgs {
            println!("Checking {}/{} ...", meta.user, meta.repo);
            match check_pkg_integrity(&meta) {
                Ok((remote_ver, up_to_date)) => {
                    if up_to_date {
                        println!("  {}/{} is up to date.", meta.user, meta.repo);
                    } else {
                        let confirm = if auto_confirm {
                            true
                        } else {
                            print!(
                                "  Update package {}/{} to {}? (y/N): ",
                                meta.user, meta.repo, remote_ver
                            );
                            use std::io::Write;
                            let _ = std::io::stdout().flush();
                            let mut input = String::new();
                            if std::io::stdin().read_line(&mut input).is_ok() {
                                let trimmed = input.trim().to_lowercase();
                                trimmed == "y" || trimmed == "yes"
                            } else {
                                false
                            }
                        };
                        if confirm {
                            pkg_install_internal(
                                &format!("{}/{}", meta.user, meta.repo),
                                None,
                                true,
                            );
                        }
                    }
                }
                Err(e) => {
                    println!("  {}/{} check failed: {}", meta.user, meta.repo, e);
                    let confirm = if auto_confirm {
                        true
                    } else {
                        print!(
                            "  Attempt update for {}/{} anyway? (y/N): ",
                            meta.user, meta.repo
                        );
                        use std::io::Write;
                        let _ = std::io::stdout().flush();
                        let mut input = String::new();
                        if std::io::stdin().read_line(&mut input).is_ok() {
                            let trimmed = input.trim().to_lowercase();
                            trimmed == "y" || trimmed == "yes"
                        } else {
                            false
                        }
                    };
                    if confirm {
                        pkg_install_internal(&format!("{}/{}", meta.user, meta.repo), None, true);
                    }
                }
            }
        }
    }
}

fn pkg_list() {
    migrate_legacy_packages();
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
            .map(|rd| rd.filter_map(|e| e.ok()).collect())
            .unwrap_or_default();
        repos.sort_by_key(|e: &std::fs::DirEntry| e.file_name());

        for repo_entry in &repos {
            let path = repo_entry.path();
            let repo = repo_entry.file_name();

            if path.is_dir() {
                let n_hy = std::fs::read_dir(&path)
                    .map(|rd| {
                        rd.filter_map(|e| e.ok())
                            .filter(|e| {
                                let ext = e
                                    .path()
                                    .extension()
                                    .and_then(|x| x.to_str())
                                    .unwrap_or("")
                                    .to_lowercase();
                                ext == "hay" || ext == "hy"
                            })
                            .count()
                    })
                    .unwrap_or(0);

                if n_hy > 0 {
                    if !found {
                        println!("Installed packages (~/.hay/packages/):\n");
                        found = true;
                    }
                    let user_s = user.to_string_lossy().to_string();
                    let repo_s = repo.to_string_lossy().to_string();
                    let version = read_pkg_metadata(&user_s, &repo_s)
                        .map(|m| normalize_version(&m.version))
                        .unwrap_or_else(|| "unknown".into());
                    println!(
                        "  {}/{}  v{}  ({} script file{})",
                        user_s,
                        repo_s,
                        version,
                        n_hy,
                        if n_hy == 1 { "" } else { "s" }
                    );
                }
            } else if path.is_file() {
                let ext = path
                    .extension()
                    .and_then(|x| x.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if matches!(ext.as_str(), "so" | "dll" | "dylib" | "wasm") {
                    if !found {
                        println!("Installed packages (~/.hay/packages/):\n");
                        found = true;
                    }
                    let user_s = user.to_string_lossy().to_string();
                    let clean_name = repo
                        .to_string_lossy()
                        .trim_end_matches(&format!(".{ext}"))
                        .to_string();
                    let version = read_pkg_metadata(&user_s, &clean_name)
                        .map(|m| normalize_version(&m.version))
                        .unwrap_or_else(|| "unknown".into());
                    println!(
                        "  {}/{}  v{}  (native {} plugin)",
                        user_s, clean_name, version, ext
                    );
                }
            }
        }
    }
    if !found {
        println!("No packages installed.");
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct PkgMetadata {
    user: String,
    repo: String,
    version: String,
    installed_at: String,
    pkg_type: String, // "native" or "script"
}

fn pkg_metadata_path(user: &str, repo: &str) -> std::path::PathBuf {
    packages_dir()
        .join(user)
        .join(format!("{repo}.metadata.json"))
}

fn read_pkg_metadata(user: &str, repo: &str) -> Option<PkgMetadata> {
    let path = pkg_metadata_path(user, repo);
    if path.exists() {
        if let Ok(content) = std::fs::read_to_string(path) {
            return serde_json::from_str(&content).ok();
        }
    }
    None
}

/// Strip a leading 'v' from version strings so output is consistently `vX.Y.Z`.
fn normalize_version(v: &str) -> String {
    v.trim().trim_start_matches('v').to_string()
}

fn write_pkg_metadata(meta: &PkgMetadata) {
    let path = pkg_metadata_path(&meta.user, &meta.repo);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(content) = serde_json::to_string_pretty(meta) {
        let _ = std::fs::write(path, content);
    }
}

#[derive(serde::Deserialize)]
struct GhEntry {
    name: String,
    r#type: String,
    download_url: Option<String>,
}

#[derive(serde::Deserialize)]
struct GhRelease {
    tag_name: String,
    assets: Vec<GhAsset>,
}

#[derive(serde::Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
}

#[derive(serde::Deserialize)]
struct GhCommitInfo {
    sha: String,
}

fn current_target_triple() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("windows", "x86_64") => "x86_64-pc-windows-msvc",
        (os, arch) => {
            eprintln!("Unsupported target platform: {os}-{arch}");
            std::process::exit(1);
        }
    }
}

fn current_target_ext() -> &'static str {
    match std::env::consts::OS {
        "linux" => "so",
        "macos" => "dylib",
        "windows" => "dll",
        _ => "so",
    }
}

#[cfg(test)]
mod dist_update_tests {
    use super::{parse_dist_update_args, DistUpdateMode};

    #[test]
    fn parse_empty_returns_install() {
        assert_eq!(
            parse_dist_update_args(&[]).unwrap(),
            DistUpdateMode::Install
        );
    }

    #[test]
    fn parse_help_long() {
        assert_eq!(
            parse_dist_update_args(&["--help"]).unwrap(),
            DistUpdateMode::Help
        );
    }

    #[test]
    fn parse_help_short() {
        assert_eq!(
            parse_dist_update_args(&["-h"]).unwrap(),
            DistUpdateMode::Help
        );
    }

    #[test]
    fn parse_check() {
        assert_eq!(
            parse_dist_update_args(&["--check"]).unwrap(),
            DistUpdateMode::Check
        );
    }

    #[test]
    fn parse_help_wins_over_check() {
        assert_eq!(
            parse_dist_update_args(&["--check", "--help"]).unwrap(),
            DistUpdateMode::Help
        );
    }

    #[test]
    fn parse_unknown_flag_fails() {
        assert!(parse_dist_update_args(&["--foo"]).is_err());
    }

    #[test]
    fn parse_positional_fails() {
        assert!(parse_dist_update_args(&["nightly"]).is_err());
    }
}
