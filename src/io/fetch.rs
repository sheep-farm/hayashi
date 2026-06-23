use crate::lang::error::{HayashiError, Result};
use std::io::Write;

/// Baixa `url` para um arquivo temporário e retorna o caminho.
/// O arquivo temporário é mantido vivo enquanto o `TempPath` existir.
pub fn download_to_temp(url: &str) -> Result<tempfile::TempPath> {
    let ext = url
        .split('?')
        .next()
        .unwrap_or(url)
        .rsplit('.')
        .next()
        .unwrap_or("tmp");

    let mut tmp = tempfile::Builder::new()
        .suffix(&format!(".{ext}"))
        .tempfile()
        .map_err(|e| HayashiError::Runtime(format!("cannot create temp file: {e}")))?;

    let resp = ureq::get(url)
        .call()
        .map_err(|e| HayashiError::Runtime(format!("HTTP error for '{url}': {e}")))?;

    let mut reader = resp.into_reader();
    std::io::copy(&mut reader, &mut tmp)
        .map_err(|e| HayashiError::Runtime(format!("download error: {e}")))?;

    tmp.flush()
        .map_err(|e| HayashiError::Runtime(format!("flush error: {e}")))?;

    Ok(tmp.into_temp_path())
}

pub fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}
