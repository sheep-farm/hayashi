use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};

pub fn read_message<R: Read>(reader: &mut BufReader<R>) -> std::io::Result<Option<String>> {
    let mut header = String::new();
    loop {
        header.clear();
        let n = reader.read_line(&mut header)?;
        if n == 0 {
            return Ok(None);
        }
        if header == "\r\n" || header == "\n" {
            continue;
        }
        break;
    }

    let content_len = header
        .trim()
        .strip_prefix("Content-Length: ")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid DAP header: {header:?}"),
            )
        })?;

    // Consume the empty line (\r\n) that follows the header.
    let mut blank = String::new();
    reader.read_line(&mut blank)?;

    let mut body = vec![0u8; content_len];
    reader.read_exact(&mut body)?;

    Ok(Some(String::from_utf8_lossy(&body).to_string()))
}

pub fn write_message<W: Write>(writer: &mut W, msg: &str) -> std::io::Result<()> {
    let header = format!("Content-Length: {}\r\n\r\n", msg.len());
    writer.write_all(header.as_bytes())?;
    writer.write_all(msg.as_bytes())?;
    writer.flush()?;
    Ok(())
}

pub fn send<W: Write, T: Serialize>(writer: &mut W, value: &T) -> std::io::Result<()> {
    let msg = serde_json::to_string(value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    write_message(writer, &msg)
}

pub fn recv<R: Read, T: for<'de> Deserialize<'de>>(
    reader: &mut BufReader<R>,
) -> std::io::Result<Option<T>> {
    match read_message(reader)? {
        Some(json) => {
            let value = serde_json::from_str(&json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            Ok(Some(value))
        }
        None => Ok(None),
    }
}
