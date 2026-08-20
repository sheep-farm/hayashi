use super::protocol::*;
use super::transport;
use std::io::{BufReader, Cursor};

#[test]
fn read_write_message_roundtrip() {
    let msg = r#"{"seq":1,"type":"request","command":"initialize"}"#;
    let mut encoded: Vec<u8> = Vec::new();
    transport::write_message(&mut encoded, msg).unwrap();

    let mut reader = BufReader::new(Cursor::new(encoded));
    let decoded = transport::read_message(&mut reader).unwrap();
    assert_eq!(decoded, Some(msg.to_string()));
}

#[test]
fn read_message_with_optional_leading_blank_lines() {
    let msg = r#"{"seq":2,"type":"request","command":"setBreakpoints"}"#;
    let header = format!("\r\n\r\nContent-Length: {}\r\n\r\n{}", msg.len(), msg);
    let mut reader = BufReader::new(Cursor::new(header.into_bytes()));
    let decoded = transport::read_message(&mut reader).unwrap();
    assert_eq!(decoded, Some(msg.to_string()));
}

#[test]
fn read_message_returns_none_on_eof() {
    let mut reader = BufReader::new(Cursor::new(Vec::new()));
    let decoded = transport::read_message(&mut reader).unwrap();
    assert_eq!(decoded, None);
}

#[test]
fn response_serializes_success_body() {
    let resp = Response::new(1, 1, "initialize", true).with_body(Capabilities::default());
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("\"success\":true"));
    assert!(json.contains("\"command\":\"initialize\""));
    assert!(json.contains("\"supportsConfigurationDoneRequest\":true"));
}

#[test]
fn response_omits_body_when_none() {
    let resp = Response::ok(2, 2, "launch");
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("\"success\":true"));
    assert!(!json.contains("\"body\""));
}

#[test]
fn event_serializes_stopped() {
    use serde_json::json;
    let event = Event {
        seq: 5,
        type_field: "event",
        event: "stopped".into(),
        body: Some(json!({"reason":"breakpoint","threadId":1})),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"event\":\"stopped\""));
    assert!(json.contains("\"reason\":\"breakpoint\""));
}
