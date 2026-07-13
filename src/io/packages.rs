//! Package management for Hayashi plugins.
//!
//! Supports installing script and native plugins from GitHub repositories
//! into `~/.hay/packages/`. This module is native-only because it performs
//! network I/O and filesystem operations.

use crate::lang::error::{HayashiError, Result};
use std::path::PathBuf;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the canonical package installation directory: `~/.hay/packages`.
pub fn packages_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .unwrap_or_else(|_| std::env::var("USERPROFILE").unwrap_or_else(|_| ".".into()));
    std::path::Path::new(&home).join(".hay").join("packages")
}

/// Checks whether a package is already installed as either a script directory
/// or a native/WASM binary file.
pub fn is_pkg_installed(user: &str, repo: &str) -> Option<PathBuf> {
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

/// Metadata persisted for each installed package.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct PkgMetadata {
    pub user: String,
    pub repo: String,
    pub version: String,
    pub installed_at: String,
    pub pkg_type: String, // "native" or "script"
}

/// Path to the JSON metadata file for a package.
pub fn pkg_metadata_path(user: &str, repo: &str) -> PathBuf {
    packages_dir()
        .join(user)
        .join(format!("{repo}.metadata.json"))
}

/// Reads the persisted metadata for a package, if any.
pub fn read_pkg_metadata(user: &str, repo: &str) -> Option<PkgMetadata> {
    let path = pkg_metadata_path(user, repo);
    if path.exists() {
        if let Ok(content) = std::fs::read_to_string(path) {
            return serde_json::from_str(&content).ok();
        }
    }
    None
}

/// Writes metadata for a package, creating parent directories as needed.
pub fn write_pkg_metadata(meta: &PkgMetadata) {
    let path = pkg_metadata_path(&meta.user, &meta.repo);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(content) = serde_json::to_string_pretty(meta) {
        let _ = std::fs::write(path, content);
    }
}

/// Installs a package from GitHub.
///
/// * `spec` must be of the form `user/repo`.
/// * `version` may be `None` (latest), a tag like `v1.2.3`, or a plain version.
/// * If `force` is `false` and the package is already installed, the function
///   returns `Ok(())` without doing anything.
pub fn install(spec: &str, version: Option<&str>, force: bool) -> Result<()> {
    let (user, repo) = parse_spec(spec)?;

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

    if !force {
        if let Some(installed_path) = is_pkg_installed(user, repo) {
            println!(
                "Package {user}/{repo} is already installed at {}",
                installed_path.display()
            );
            return Ok(());
        }
    }

    let dest = packages_dir().join(user).join(repo);
    let api_url = format!("https://api.github.com/repos/{user}/{repo}/contents/");
    println!("Fetching {user}/{repo}...");

    let resp = ureq::get(&api_url)
        .set("User-Agent", "hay")
        .set("Accept", "application/vnd.github.v3+json")
        .call()
        .map_err(|e| HayashiError::Runtime(format!("cannot reach GitHub API: {e}")))?;

    let body: String = resp.into_string().unwrap_or_default();
    let entries: Vec<GhEntry> = serde_json::from_str(&body)
        .map_err(|e| HayashiError::Runtime(format!("cannot parse GitHub response: {e}")))?;

    // Check plugin compatibility: look for hayashi.toml in repo root
    if let Some(toml_entry) = entries.iter().find(|e| e.name == "hayashi.toml") {
        if let Some(url) = &toml_entry.download_url {
            if let Ok(resp) = ureq::get(url).set("User-Agent", "hay").call() {
                let toml_body = resp.into_string().unwrap_or_default();
                // Parse min_version = "x.y.z" (simple TOML, no crate needed)
                if let Some(line) = toml_body
                    .lines()
                    .find(|l| l.trim_start().starts_with("min_version"))
                {
                    if let Some(val) = line.split('=').nth(1) {
                        let min_ver = val.trim().trim_matches('"').trim_matches('\'');
                        if !meets_min_version(VERSION, min_ver) {
                            return Err(HayashiError::Runtime(format!(
                                "{user}/{repo} requires Hayashi >= {min_ver} (you have {VERSION})"
                            )));
                        }
                    }
                }
                // Parse primitives = ["export", "plot", ...]
                if let Some(line) = toml_body
                    .lines()
                    .find(|l| l.trim_start().starts_with("primitives"))
                {
                    if let Some(val) = line.split('=').nth(1) {
                        let primitives: Vec<&str> = val
                            .trim()
                            .trim_matches(['[', ']'])
                            .split(',')
                            .map(|s| s.trim().trim_matches('"').trim_matches('\''))
                            .filter(|s| !s.is_empty())
                            .collect();
                        if !primitives.is_empty() {
                            println!(
                                "install: {user}/{repo} overrides builtin(s): {}",
                                primitives.join(", ")
                            );
                        }
                    }
                }
            }
        }
    }

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

        let release_resp = ureq::get(&release_url)
            .set("User-Agent", "hay")
            .set("Accept", "application/vnd.github.v3+json")
            .call()
            .map_err(|e| {
                if let Some(tag) = &version_tag {
                    HayashiError::Runtime(format!("release {tag} not found for {user}/{repo}: {e}"))
                } else {
                    HayashiError::Runtime(format!(
                        "no scripts or native releases found for {user}/{repo}: {e}"
                    ))
                }
            })?;

        let release_body: String = release_resp.into_string().unwrap_or_default();
        let release: GhRelease = serde_json::from_str(&release_body)
            .map_err(|e| HayashiError::Runtime(format!("cannot parse release payload: {e}")))?;

        let target = current_target_triple()?;
        let ext = current_target_ext();

        let matching_asset = release
            .assets
            .iter()
            .find(|asset| asset.name.contains(target) && asset.name.ends_with(ext));

        if let Some(asset) = matching_asset {
            println!("Found binary release for {target}: {}", asset.name);
            let parent_dir = packages_dir().join(user);
            std::fs::create_dir_all(&parent_dir).map_err(|e| {
                HayashiError::Runtime(format!("cannot create {}: {e}", parent_dir.display()))
            })?;
            let dest_file = parent_dir.join(format!("{repo}.{ext}"));

            print!("Downloading {} ... ", asset.name);
            match ureq::get(&asset.browser_download_url).call() {
                Ok(resp) => {
                    let mut reader = resp.into_reader();
                    let mut out_file = std::fs::File::create(&dest_file).map_err(|e| {
                        HayashiError::Runtime(format!("cannot create {}: {e}", dest_file.display()))
                    })?;
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
                        return Ok(());
                    } else {
                        println!("write error");
                    }
                }
                Err(e) => println!("download error: {e}"),
            }
            return Err(HayashiError::Runtime(format!(
                "failed to download native plugin {user}/{repo}"
            )));
        } else {
            return Err(HayashiError::Runtime(format!(
                "no compatible release asset found for {target}"
            )));
        }
    }

    std::fs::create_dir_all(&dest)
        .map_err(|e| HayashiError::Runtime(format!("cannot create {}: {e}", dest.display())))?;

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
    Ok(())
}

/// Installs multiple packages listed in a file.
///
/// File format: one `user/repo [version]` per line, blank lines and `#` comments ignored.
/// Returns the number of packages successfully processed.
pub fn install_from_file(file_path: &str, force: bool) -> Result<usize> {
    let content = std::fs::read_to_string(file_path)
        .map_err(|e| HayashiError::Runtime(format!("cannot read file '{file_path}': {e}")))?;

    let mut installed_count = 0;
    let mut failed_count = 0;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        let spec = parts[0];
        let version = parts.get(1).copied();

        let (user, repo) = match parse_spec(spec) {
            Ok((u, r)) => (u, r),
            Err(e) => {
                eprintln!("install: {e} in file '{file_path}'");
                failed_count += 1;
                continue;
            }
        };

        println!("Installing {user}/{repo}...");
        if let Err(e) = install(spec, version, force) {
            eprintln!("install: {e}");
            failed_count += 1;
            continue;
        }
        installed_count += 1;
    }

    println!();
    println!("Installation summary:");
    println!("  Installed: {installed_count}");
    if failed_count > 0 {
        println!("  Failed: {failed_count}");
    }
    Ok(installed_count)
}

/// Removes an installed package.
pub fn remove(spec: &str) -> Result<()> {
    let (user, repo) = parse_spec(spec)?;

    let dir = packages_dir().join(user).join(repo);
    let ext = current_target_ext();
    let file = packages_dir().join(user).join(format!("{repo}.{ext}"));
    let meta_file = pkg_metadata_path(user, repo);

    let mut removed = false;

    if dir.exists() && dir.is_dir() {
        std::fs::remove_dir_all(&dir)
            .map_err(|e| HayashiError::Runtime(format!("cannot remove {}: {e}", dir.display())))?;
        removed = true;
    }

    if file.exists() && file.is_file() {
        std::fs::remove_file(&file)
            .map_err(|e| HayashiError::Runtime(format!("cannot remove {}: {e}", file.display())))?;
        removed = true;
    }

    if meta_file.exists() {
        let _ = std::fs::remove_file(&meta_file);
    }

    if !removed {
        return Err(HayashiError::Runtime(format!(
            "package '{spec}' not installed"
        )));
    }

    let user_dir = packages_dir().join(user);
    if user_dir.exists() {
        let _ = std::fs::remove_dir(&user_dir);
    }

    println!("Removed {spec}");
    Ok(())
}

/// Lists installed packages. Returns tuples of `(user, repo, version, kind)`.
pub fn list_installed() -> Vec<(String, String, String, String)> {
    migrate_legacy_packages();
    let mut result = Vec::new();
    let dir = packages_dir();
    if !dir.is_dir() {
        return result;
    }
    let mut users: Vec<_> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .collect()
        })
        .unwrap_or_default();
    users.sort_by_key(|e: &std::fs::DirEntry| e.file_name());

    for user_entry in &users {
        let user = user_entry.file_name().to_string_lossy().to_string();
        let mut repos: Vec<_> = std::fs::read_dir(user_entry.path())
            .map(|rd| rd.filter_map(|e| e.ok()).collect())
            .unwrap_or_default();
        repos.sort_by_key(|e: &std::fs::DirEntry| e.file_name());

        for repo_entry in &repos {
            let path = repo_entry.path();
            let repo_name = repo_entry.file_name().to_string_lossy().to_string();

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
                    let version = read_pkg_metadata(&user, &repo_name)
                        .map(|m| normalize_version(&m.version))
                        .unwrap_or_else(|| "unknown".into());
                    result.push((user.clone(), repo_name, version, "script".to_string()));
                }
            } else if path.is_file() {
                let ext = path
                    .extension()
                    .and_then(|x| x.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if matches!(ext.as_str(), "so" | "dll" | "dylib" | "wasm") {
                    let clean_name = repo_name.trim_end_matches(&format!(".{ext}")).to_string();
                    let version = read_pkg_metadata(&user, &clean_name)
                        .map(|m| normalize_version(&m.version))
                        .unwrap_or_else(|| "unknown".into());
                    result.push((user.clone(), clean_name, version, ext));
                }
            }
        }
    }
    result
}

/// Creates metadata files for legacy packages that lack them.
pub fn migrate_legacy_packages() {
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

/// Reads all installed package metadata.
pub fn get_installed_packages() -> Vec<PkgMetadata> {
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

/// Checks whether a package is up-to-date with its remote GitHub repository.
/// Returns `(remote_version, is_up_to_date)`.
pub fn check_integrity(meta: &PkgMetadata) -> Result<(String, bool)> {
    if meta.pkg_type == "native" {
        let release_url = format!(
            "https://api.github.com/repos/{}/{}/releases/latest",
            meta.user, meta.repo
        );
        let resp = ureq::get(&release_url)
            .set("User-Agent", "hay")
            .set("Accept", "application/vnd.github.v3+json")
            .call()
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let body: String = resp
            .into_string()
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let release: GhRelease =
            serde_json::from_str(&body).map_err(|e| HayashiError::Runtime(e.to_string()))?;

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
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;

        let body: String = resp
            .into_string()
            .map_err(|e| HayashiError::Runtime(e.to_string()))?;
        let commits: Vec<GhCommitInfo> =
            serde_json::from_str(&body).map_err(|e| HayashiError::Runtime(e.to_string()))?;

        if let Some(first) = commits.first() {
            let up_to_date = meta.version == first.sha;
            Ok((first.sha.clone(), up_to_date))
        } else {
            Err(HayashiError::Runtime(
                "No commits found in remote repository".to_string(),
            ))
        }
    }
}

pub fn parse_spec(spec: &str) -> Result<(&str, &str)> {
    if let Some(pos) = spec.find('/') {
        Ok((&spec[..pos], &spec[pos + 1..]))
    } else {
        Err(HayashiError::Runtime(format!(
            "expected 'user/repo', got '{spec}'"
        )))
    }
}

/// Compares only the numeric part (ignores -dev, -rc, etc. pre-release suffixes),
/// so 0.2.9-dev is considered compatible with min_version "0.2.9".
fn meets_min_version(current: &str, required: &str) -> bool {
    fn parse_nums(v: &str) -> Vec<u32> {
        let v = v.trim_start_matches('v');
        v.split('.')
            .map(|s| {
                s.chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
            })
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap_or(0))
            .collect()
    }
    let cur = parse_nums(current);
    let req = parse_nums(required);
    for (c, r) in cur.iter().zip(req.iter()) {
        match c.cmp(r) {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {}
        }
    }
    cur.len() >= req.len()
}

pub fn current_target_triple() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok("x86_64-unknown-linux-gnu"),
        ("macos", "aarch64") => Ok("aarch64-apple-darwin"),
        ("macos", "x86_64") => Ok("x86_64-apple-darwin"),
        ("windows", "x86_64") => Ok("x86_64-pc-windows-msvc"),
        (os, arch) => Err(HayashiError::Runtime(format!(
            "Unsupported target platform: {os}-{arch}"
        ))),
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

pub fn normalize_version(v: &str) -> String {
    v.trim().trim_start_matches('v').to_string()
}

#[derive(serde::Deserialize)]
struct GhEntry {
    name: String,
    r#type: String,
    download_url: Option<String>,
}

#[derive(serde::Deserialize)]
pub struct GhRelease {
    pub tag_name: String,
    pub assets: Vec<GhAsset>,
}

#[derive(serde::Deserialize)]
pub struct GhAsset {
    pub name: String,
    pub browser_download_url: String,
}

#[derive(serde::Deserialize)]
struct GhCommitInfo {
    sha: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meets_min_version_works() {
        assert!(meets_min_version("0.2.8", "0.2.8"));
        assert!(meets_min_version("0.2.9-dev", "0.2.8"));
        assert!(meets_min_version("0.3.0", "0.2.8"));
        assert!(!meets_min_version("0.2.7", "0.2.8"));
        assert!(meets_min_version("0.2.10", "0.2.9"));
    }
}
