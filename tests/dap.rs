use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct DapSession {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    child: Child,
    seq: i64,
}

impl DapSession {
    fn spawn(script: &str) -> (Self, std::path::PathBuf) {
        let tmp = std::env::temp_dir().join("hay_dap_test.hay");
        std::fs::write(&tmp, script).unwrap();

        let bin = env!("CARGO_BIN_EXE_hay");
        let mut child = Command::new(bin)
            .arg("dap")
            .arg(&tmp)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        (
            Self {
                stdin,
                stdout,
                child,
                seq: 0,
            },
            tmp,
        )
    }

    fn next_seq(&mut self) -> i64 {
        self.seq += 1;
        self.seq
    }

    fn send(&mut self, command: &str, arguments: &str) {
        let seq = self.next_seq();
        let body = if arguments.is_empty() {
            format!("{{\"seq\":{seq},\"type\":\"request\",\"command\":\"{command}\"}}")
        } else {
            format!(
                "{{\"seq\":{seq},\"type\":\"request\",\"command\":\"{command}\",\"arguments\":{arguments}}}"
            )
        };
        let header = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        self.stdin.write_all(header.as_bytes()).unwrap();
        self.stdin.flush().unwrap();
    }

    fn read(&mut self) -> String {
        let mut header = String::new();
        loop {
            header.clear();
            if self.stdout.read_line(&mut header).unwrap() == 0 {
                panic!("EOF while reading DAP header");
            }
            if header == "\r\n" || header == "\n" {
                continue;
            }
            break;
        }
        let len = header
            .trim()
            .strip_prefix("Content-Length: ")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| panic!("invalid DAP header: {header:?}"));
        // Consume the empty line after the header.
        let mut blank = String::new();
        self.stdout.read_line(&mut blank).unwrap();
        let mut body = vec![0u8; len];
        self.stdout.read_exact(&mut body).unwrap();
        String::from_utf8(body).unwrap()
    }

    fn wait_for(&mut self, needle: &str, attempts: usize) -> String {
        for _ in 0..attempts {
            let msg = self.read();
            if msg.contains(needle) {
                return msg;
            }
        }
        panic!("did not find {needle} in DAP output");
    }

    fn close(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn dap_breakpoint_and_variables() {
    let (mut session, _tmp) = DapSession::spawn("let x = 1\nlet y = 2\nlet z = x + y\n");

    session.send("initialize", "");
    session.send(
        "setBreakpoints",
        &format!(
            "{{\"source\":{{\"path\":\"{}\"}},\"breakpoints\":[{{\"line\":3}}]}}",
            _tmp.display()
        ),
    );
    session.send(
        "launch",
        &format!(
            "{{\"request\":\"launch\",\"name\":\"test\",\"program\":\"{}\"}}",
            _tmp.display()
        ),
    );
    session.send("configurationDone", "");

    let stopped = session.wait_for("\"event\":\"stopped\"", 100);
    assert!(stopped.contains("\"reason\":\"breakpoint\""));

    session.send("variables", "{\"variablesReference\":1}");
    let vars = session.wait_for("\"command\":\"variables\"", 100);
    assert!(vars.contains("\"name\":\"x\""));
    assert!(vars.contains("\"value\":\"1\""));
    assert!(vars.contains("\"name\":\"y\""));
    assert!(vars.contains("\"value\":\"2\""));

    session.send("continue", "");
    session.wait_for("\"event\":\"terminated\"", 100);
    session.send("disconnect", "");
    session.close();
}

#[test]
fn dap_runs_to_completion_without_breakpoints() {
    let (mut session, _tmp) = DapSession::spawn("let a = 10\n");

    session.send("initialize", "");
    session.send(
        "launch",
        &format!(
            "{{\"request\":\"launch\",\"name\":\"test\",\"program\":\"{}\"}}",
            _tmp.display()
        ),
    );
    session.send("configurationDone", "");

    session.wait_for("\"event\":\"terminated\"", 100);
    session.send("disconnect", "");
    session.close();
}
