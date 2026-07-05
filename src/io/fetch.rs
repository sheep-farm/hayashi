use crate::lang::error::{HayashiError, Result};
use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::time::Duration;
use url::{Host, Url};

const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_DOWNLOAD_BYTES: u64 = 100 * 1024 * 1024;

struct ValidatedDownloadUrl {
    url: Url,
    resolver_key: String,
    socket_addrs: Vec<SocketAddr>,
}

/// Downloads `url` to a temporary file and returns the path.
/// The temporary file is kept alive while the `TempPath` exists.
pub fn download_to_temp(url: &str) -> Result<tempfile::TempPath> {
    let validated = validate_download_url(url)?;
    download_validated_to_temp(url, validated, MAX_DOWNLOAD_BYTES)
}

fn download_validated_to_temp(
    url: &str,
    validated: ValidatedDownloadUrl,
    max_download_bytes: u64,
) -> Result<tempfile::TempPath> {
    let ext = extension_from_url(&validated.url);

    let mut tmp = tempfile::Builder::new()
        .suffix(&format!(".{ext}"))
        .tempfile()
        .map_err(|e| HayashiError::Runtime(format!("cannot create temp file: {e}")))?;

    let resolver_key = validated.resolver_key.clone();
    let socket_addrs = validated.socket_addrs.clone();
    let agent = ureq::AgentBuilder::new()
        .timeout(DOWNLOAD_TIMEOUT)
        .redirects(0)
        .resolver(move |addr: &str| {
            if addr == resolver_key {
                Ok(socket_addrs.clone())
            } else {
                resolve_safe_socket_addrs(addr)
            }
        })
        .build();

    let resp = agent
        .get(validated.url.as_str())
        .call()
        .map_err(|e| HayashiError::Runtime(format!("HTTP error for '{url}': {e}")))?;
    let status = resp.status();
    if !(200..=299).contains(&status) {
        return Err(HayashiError::Runtime(format!(
            "HTTP error for '{url}': status {status} {}",
            resp.status_text()
        )));
    }

    if let Some(content_length) = resp.header("Content-Length") {
        let content_length = content_length.parse::<u64>().map_err(|e| {
            HayashiError::Runtime(format!("invalid Content-Length for '{url}': {e}"))
        })?;
        if content_length > max_download_bytes {
            return Err(HayashiError::Runtime(format!(
                "download too large for '{url}': {content_length} bytes exceeds {max_download_bytes} byte limit"
            )));
        }
    }

    let mut reader = resp.into_reader();
    copy_limited(&mut reader, &mut tmp, max_download_bytes)
        .map_err(|e| HayashiError::Runtime(format!("download error: {e}")))?;

    tmp.flush()
        .map_err(|e| HayashiError::Runtime(format!("flush error: {e}")))?;

    Ok(tmp.into_temp_path())
}

pub fn is_url(s: &str) -> bool {
    parse_http_url(s).is_ok()
}

fn validate_download_url(raw: &str) -> Result<ValidatedDownloadUrl> {
    let url = parse_http_url(raw)?;
    match url
        .host()
        .ok_or_else(|| HayashiError::Runtime(format!("URL has no host: '{raw}'")))?
    {
        Host::Domain(host) => validate_host_label(host, raw)?,
        Host::Ipv4(ip) => validate_public_ip(IpAddr::V4(ip), raw)?,
        Host::Ipv6(ip) => validate_public_ip(IpAddr::V6(ip), raw)?,
    }

    let host = url
        .host_str()
        .ok_or_else(|| HayashiError::Runtime(format!("URL has no host: '{raw}'")))?;
    let resolver_key = resolver_key_for_url(&url, raw)?;
    let socket_addrs = resolve_safe_socket_addrs(&resolver_key)
        .map_err(|e| HayashiError::Runtime(format!("cannot resolve '{host}': {e}")))?;

    Ok(ValidatedDownloadUrl {
        url,
        resolver_key,
        socket_addrs,
    })
}

fn parse_http_url(raw: &str) -> Result<Url> {
    let url =
        Url::parse(raw).map_err(|e| HayashiError::Runtime(format!("invalid URL '{raw}': {e}")))?;

    if !matches!(url.scheme(), "http" | "https") {
        return Err(HayashiError::Runtime(format!(
            "unsupported URL scheme for '{raw}': expected http or https"
        )));
    }

    url.host()
        .ok_or_else(|| HayashiError::Runtime(format!("URL has no host: '{raw}'")))?;

    Ok(url)
}

fn resolver_key_for_url(url: &Url, raw: &str) -> Result<String> {
    let port = url.port_or_known_default().ok_or_else(|| {
        HayashiError::Runtime(format!("URL has no port or known default port: '{raw}'"))
    })?;

    match url
        .host()
        .ok_or_else(|| HayashiError::Runtime(format!("URL has no host: '{raw}'")))?
    {
        Host::Domain(host) => Ok(format!("{host}:{port}")),
        Host::Ipv4(ip) => Ok(format!("{ip}:{port}")),
        Host::Ipv6(ip) => Ok(format!("[{ip}]:{port}")),
    }
}

fn validate_host_label(host: &str, raw: &str) -> Result<()> {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty()
        || host == "localhost"
        || host.ends_with(".localhost")
        || host == "ip6-localhost"
        || host == "ip6-loopback"
    {
        return Err(HayashiError::Runtime(format!(
            "unsafe local URL target rejected: '{raw}'"
        )));
    }

    Ok(())
}

fn resolve_safe_socket_addrs(addr: &str) -> std::io::Result<Vec<SocketAddr>> {
    let socket_addrs = addr.to_socket_addrs()?.collect::<Vec<_>>();
    if socket_addrs.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AddrNotAvailable,
            "host resolved to no addresses",
        ));
    }

    for socket_addr in &socket_addrs {
        if let Err(err) = validate_public_ip(socket_addr.ip(), addr) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                err,
            ));
        }
    }

    Ok(socket_addrs)
}

fn validate_public_ip(ip: IpAddr, raw: &str) -> Result<()> {
    let is_safe = match ip {
        IpAddr::V4(ip) => is_public_ipv4(ip),
        IpAddr::V6(ip) => is_public_ipv6(ip),
    };

    if is_safe {
        Ok(())
    } else {
        Err(HayashiError::Runtime(format!(
            "unsafe local or private URL target rejected: '{raw}'"
        )))
    }
}

fn is_public_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, _, _] = ip.octets();

    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_documentation()
        || a == 0
        || a == 127
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 192 && b == 0)
        || (a == 198 && (18..=19).contains(&b))
        || a >= 224)
}

fn is_public_ipv6(ip: Ipv6Addr) -> bool {
    if let Some(ipv4) = embedded_ipv4_addr(ip) {
        return is_public_ipv4(ipv4);
    }

    let segments = ip.segments();

    !(ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || ip.is_unique_local()
        || ip.is_unicast_link_local()
        || (segments[0] == 0x2001 && segments[1] == 0x0db8))
}

fn embedded_ipv4_addr(ip: Ipv6Addr) -> Option<Ipv4Addr> {
    let segments = ip.segments();
    let first_five_zero = segments[..5].iter().all(|segment| *segment == 0);
    if first_five_zero && (segments[5] == 0 || segments[5] == 0xffff) {
        let [a, b] = segments[6].to_be_bytes();
        let [c, d] = segments[7].to_be_bytes();
        return Some(Ipv4Addr::new(a, b, c, d));
    }

    None
}

fn extension_from_url(url: &Url) -> &str {
    let file_name = url.path().rsplit('/').next().unwrap_or_default();
    let Some((_, ext)) = file_name.rsplit_once('.') else {
        return "tmp";
    };

    if ext.is_empty()
        || ext.len() > 16
        || !ext
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        "tmp"
    } else {
        ext
    }
}

fn copy_limited<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    max_bytes: u64,
) -> std::io::Result<u64> {
    let mut total = 0_u64;
    let mut buffer = [0_u8; 8192];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(total);
        }

        let read_u64 = u64::try_from(read).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "read size does not fit into u64",
            )
        })?;

        if total.saturating_add(read_u64) > max_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("download exceeds {max_bytes} byte limit"),
            ));
        }

        writer.write_all(&buffer[..read])?;
        total += read_u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Cursor;
    use std::net::TcpListener;
    use std::thread;
    use std::thread::JoinHandle;

    #[test]
    fn is_url_rejects_malformed_and_non_http_urls() {
        assert!(!is_url("https://"));
        assert!(!is_url("ftp://example.com/data.csv"));
        assert!(!is_url("/tmp/data.csv"));
    }

    #[test]
    fn is_url_recognizes_http_urls_even_when_download_policy_rejects_them() {
        assert!(is_url("http://localhost/data.csv"));
        assert!(is_url("http://127.0.0.1/data.csv"));
        assert!(is_url("http://169.254.169.254/latest/meta-data"));
    }

    #[test]
    fn validate_download_url_rejects_local_and_private_targets() {
        for url in [
            "http://localhost/data.csv",
            "http://example.localhost/data.csv",
            "http://0.0.0.0/data.csv",
            "http://127.0.0.1/data.csv",
            "http://10.0.0.1/data.csv",
            "http://172.16.0.1/data.csv",
            "http://192.168.0.1/data.csv",
            "http://192.0.2.1/data.csv",
            "http://198.18.0.1/data.csv",
            "http://224.0.0.1/data.csv",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/data.csv",
            "http://[fd00::1]/data.csv",
            "http://[2001:db8::1]/data.csv",
            "http://[::ffff:127.0.0.1]/data.csv",
            "http://[::ffff:10.0.0.1]/data.csv",
            "http://[::ffff:192.168.0.1]/data.csv",
            "http://[::ffff:169.254.169.254]/data.csv",
            "http://[::127.0.0.1]/data.csv",
        ] {
            assert!(
                validate_download_url(url).is_err(),
                "{url} should not pass the download safety policy"
            );
        }
    }

    #[test]
    fn is_url_accepts_public_http_urls() {
        assert!(is_url("https://example.com/data.csv"));
        assert!(is_url("http://example.com/data.csv?download=true"));
        assert!(is_url("http://[2606:4700:4700::1111]/data.csv"));
    }

    #[test]
    fn download_validated_to_temp_saves_successful_response() {
        let body = b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc";

        let bytes = run_validated_download(body, 10).unwrap();

        assert_eq!(bytes, b"abc");
    }

    #[test]
    fn download_validated_to_temp_rejects_redirect_response() {
        let body = b"HTTP/1.1 302 Found\r\nLocation: http://example.org/data.csv\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

        let err = run_validated_download(body, 10).unwrap_err();

        assert!(format!("{err}").contains("302"));
    }

    #[test]
    fn download_validated_to_temp_rejects_non_success_response() {
        let body =
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found";

        let err = run_validated_download(body, 20).unwrap_err();

        assert!(format!("{err}").contains("404"));
    }

    #[test]
    fn download_validated_to_temp_rejects_oversized_content_length() {
        let body = b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabcdef";

        let err = run_validated_download(body, 5).unwrap_err();

        assert!(format!("{err}").contains("exceeds 5 byte limit"));
    }

    #[test]
    fn download_validated_to_temp_rejects_streamed_body_over_limit() {
        let body = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nabcdef";

        let err = run_validated_download(body, 5).unwrap_err();

        assert!(format!("{err}").contains("exceeds 5 byte limit"));
    }

    #[test]
    fn copy_limited_allows_payloads_at_the_limit() {
        let mut reader = Cursor::new(b"abcde");
        let mut writer = Vec::new();

        let copied = copy_limited(&mut reader, &mut writer, 5).unwrap();

        assert_eq!(copied, 5);
        assert_eq!(writer, b"abcde");
    }

    #[test]
    fn copy_limited_rejects_payloads_over_the_limit() {
        let mut reader = Cursor::new(b"abcdef");
        let mut writer = Vec::new();

        let err = copy_limited(&mut reader, &mut writer, 5).unwrap_err();

        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("exceeds 5 byte limit"));
    }

    fn run_validated_download(response: &'static [u8], limit: u64) -> Result<Vec<u8>> {
        let (url, validated, handle) = validated_url_for_response(response);
        let result = download_validated_to_temp(&url, validated, limit)
            .and_then(|path| fs::read(&path).map_err(|e| HayashiError::Runtime(e.to_string())));

        handle.join().unwrap();
        result
    }

    fn validated_url_for_response(
        response: &'static [u8],
    ) -> (String, ValidatedDownloadUrl, JoinHandle<()>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let socket_addr = listener.local_addr().unwrap();
        let port = socket_addr.port();
        let url = format!("http://example.com:{port}/data.csv");
        let parsed_url = Url::parse(&url).unwrap();

        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request);
            stream.write_all(response).unwrap();
            stream.flush().unwrap();
        });

        let validated = ValidatedDownloadUrl {
            url: parsed_url,
            resolver_key: format!("example.com:{port}"),
            socket_addrs: vec![socket_addr],
        };

        (url, validated, handle)
    }
}
