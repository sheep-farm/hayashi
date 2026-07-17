use hayashi_lang::io::packages;
use hayashi_lang::lang;

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
    "parallel",
    "in",
    "while",
    "return",
    "break",
    "continue",
    "match",
    "try",
    "catch",
    "import",
    "install",
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
    "rbind",
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
            if let Err(e) = lang::run_source_verbose(&src, &mut interp, verbose, None) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
            return;
        }
        Some("install") => {
            let pkg = args_clean.get(2).unwrap_or_else(|| {
                eprintln!("Usage: hay install user/repo [version] [-y]");
                eprintln!("       hay install --file repositories.txt [-y]");
                std::process::exit(1);
            });

            if *pkg == "--file" {
                let file_path = args_clean.get(3).unwrap_or_else(|| {
                    eprintln!("Usage: hay install --file repositories.txt [-y]");
                    std::process::exit(1);
                });
                if let Err(e) = packages::install_from_file(file_path, yes) {
                    eprintln!("hay install: {e}");
                    std::process::exit(1);
                }
            } else {
                let version = args_clean.get(3).copied();
                let (user, repo) = packages::parse_spec(pkg).unwrap_or_else(|e| {
                    eprintln!("hay install: {e}");
                    std::process::exit(1);
                });
                if !yes {
                    if let Some(installed_path) = packages::is_pkg_installed(user, repo) {
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
                if let Err(e) = packages::install(pkg, version, true) {
                    eprintln!("hay install: {e}");
                    std::process::exit(1);
                }
            }
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
        Some("dap") => {
            let program = args_clean.get(2).unwrap_or_else(|| {
                eprintln!("Usage: hay dap <script.hay>");
                std::process::exit(1);
            });
            hayashi_lang::lang::dap::adapter::run_dap(
                std::io::stdin(),
                std::io::stdout(),
                std::path::Path::new(program),
            );
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
                "Usage: hay [script.hay | - | dap | install | remove | list | update | check-plugin | validate | dist-update]"
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
    let path = std::path::Path::new(path);
    if let Err(e) = lang::run_source_verbose(&src, &mut interp, verbose, Some(path)) {
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
    println!("Usage: hay dist-update [--help] [--check] [--nightly]");
    println!();
    println!("Options:");
    println!("  --help, -h  Show this help message and exit");
    println!("  --check     Report whether a newer release is available without");
    println!("              downloading or replacing the current binary");
    println!("  --nightly   Install the latest nightly build from the dev branch");
    println!("              (pre-release, may be unstable)");
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
    let release: packages::GhRelease = serde_json::from_str(&release_body)
        .map_err(|e| format!("cannot parse release payload: {e}"))?;

    let remote_version = release.tag_name.trim_start_matches('v').to_string();
    if !is_newer_version(&remote_version, VERSION) {
        return Ok(None);
    }
    Ok(Some(remote_version))
}

/// Download and replace the current binary with the given release version.
fn dist_update_install(remote_version: &str) {
    let target = packages::current_target_triple().unwrap_or_else(|e| {
        eprintln!("hay dist-update: {e}");
        std::process::exit(1);
    });
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
    let release: packages::GhRelease = serde_json::from_str(&release_body).unwrap_or_else(|e| {
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

/// Download and install the latest nightly build (pre-release tagged "nightly").
fn dist_update_nightly() {
    let target = packages::current_target_triple().unwrap_or_else(|e| {
        eprintln!("hay dist-update: {e}");
        std::process::exit(1);
    });
    let (asset_ext, archive_cmd) = dist_asset_kind();
    let asset_name = format!("hay-nightly-{target}.{asset_ext}");

    // Fetch the nightly release by tag
    let release_url = "https://api.github.com/repos/sheep-farm/hayashi/releases/tags/nightly";
    let release_resp = match ureq::get(release_url)
        .set("User-Agent", "hay")
        .set("Accept", "application/vnd.github.v3+json")
        .call()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("hay dist-update: cannot fetch nightly release: {e}");
            eprintln!("hay dist-update: nightly builds may not be available yet");
            std::process::exit(1);
        }
    };

    let release_body: String = release_resp.into_string().unwrap_or_default();
    let release: packages::GhRelease = serde_json::from_str(&release_body).unwrap_or_else(|e| {
        eprintln!("hay dist-update: cannot parse nightly release payload: {e}");
        std::process::exit(1);
    });

    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .unwrap_or_else(|| {
            eprintln!("hay dist-update: no nightly asset found for {asset_name}");
            std::process::exit(1);
        });

    let exe_path = std::env::current_exe().unwrap_or_else(|e| {
        eprintln!("hay dist-update: cannot locate current executable: {e}");
        std::process::exit(1);
    });

    let tmp_dir = std::env::temp_dir().join("hay-dist-update-nightly");
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
        println!("hay dist-update: installed nightly build. Please restart hay.");
    } else {
        println!("hay dist-update: installed nightly build. Run `hay --version` to verify.");
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DistUpdateMode {
    Help,
    Check,
    Install,
    Nightly,
}

/// Parse dist-update arguments into a safe mode or fail on unknown flags.
fn parse_dist_update_args(argv: &[&str]) -> Result<DistUpdateMode, String> {
    if argv.iter().any(|a| *a == "--help" || *a == "-h") {
        return Ok(DistUpdateMode::Help);
    }

    let mut check = false;
    let mut nightly = false;
    for arg in argv {
        if *arg == "--check" {
            check = true;
        } else if *arg == "--nightly" {
            nightly = true;
        } else if arg.starts_with('-') {
            return Err(format!("unknown flag '{arg}'"));
        } else {
            return Err(format!("unexpected positional argument '{arg}'"));
        }
    }

    if nightly {
        Ok(DistUpdateMode::Nightly)
    } else if check {
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
        Ok(DistUpdateMode::Nightly) => {
            println!("hay dist-update: current version {VERSION}");
            println!("hay dist-update: fetching nightly build...");
            dist_update_nightly();
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
    println!("Hayashi version {VERSION}");
    println!("Copyright (C) 2026 Flávio de Vasconcellos Corrêa");
    println!();
    println!("Hayashi is free software licensed under GPL-3.0-only.");
    println!(
        "You may redistribute it under the terms of the GNU General Public License version 3."
    );
    println!("This program comes with ABSOLUTELY NO WARRANTY.");
    println!();
    println!("Source code:          <https://github.com/sheep-farm/hayashi>");
    println!("License text:         <https://www.gnu.org/licenses/gpl-3.0.html>");
    println!("Project website:      <https://haylang.dev>");
    println!();
    println!("In honor of Fumio Hayashi. Type 'exit' or Ctrl-D to quit.");

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
    println!("    hay                 Start interactive REPL (multi-line)");
    println!("    hay script.hay      Run a script file");
    println!("    hay dap script.hay  Start DAP debugger server");
    println!("    hay --version       Print version");
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
    println!("    hay install user/repo [version]    Install from GitHub (-y to bypass overwrite prompt)");
    println!("    hay install --file repos.txt       Install packages from file (format: user/repo [v.N.N.N])");
    println!("    hay remove  user/repo              Uninstall a package");
    println!("    hay list                           List installed packages");
    println!("    hay update [user/repo]             Update package(s) (-y to bypass prompt)");
    println!(
        "    hay check-plugin [name]            Check integrity/version with remote repository"
    );
    println!(
        "    hay validate [options]             Run the empirical validation programme (R/Python)"
    );
    println!("    hay dist-update                    Check and install the latest hay release from GitHub");
    println!();
    println!("In REPL, type help() for full command list or help(cmd) for details.");
}

fn pkg_install_internal(spec: &str, version: Option<&str>, force_overwrite: bool) {
    if let Err(e) = packages::install(spec, version, force_overwrite) {
        eprintln!("hay install: {e}");
        std::process::exit(1);
    }
}

fn pkg_remove(spec: &str) {
    if let Err(e) = packages::remove(spec) {
        eprintln!("hay remove: {e}");
        std::process::exit(1);
    }
}

fn migrate_legacy_packages() {
    packages::migrate_legacy_packages();
}

fn get_installed_packages() -> Vec<packages::PkgMetadata> {
    packages::get_installed_packages()
}

fn check_pkg_integrity(meta: &packages::PkgMetadata) -> Result<(String, bool), String> {
    packages::check_integrity(meta).map_err(|e| e.to_string())
}

fn normalize_version(v: &str) -> String {
    packages::normalize_version(v)
}

fn pkg_check_plugin(spec_opt: Option<&str>) {
    migrate_legacy_packages();

    if let Some(spec) = spec_opt {
        let (user, repo) = packages::parse_spec(spec).unwrap_or_else(|e| {
            eprintln!("hay check-plugin: {e}");
            std::process::exit(1);
        });

        match packages::read_pkg_metadata(user, repo) {
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
        let (user, repo) = packages::parse_spec(spec).unwrap_or_else(|e| {
            eprintln!("hay update: {e}");
            std::process::exit(1);
        });

        let meta = match packages::read_pkg_metadata(user, repo) {
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
    let dir = packages::packages_dir();
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
                    let version = packages::read_pkg_metadata(&user_s, &repo_s)
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
                    let version = packages::read_pkg_metadata(&user_s, &clean_name)
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
