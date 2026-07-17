use crate::lang;
use crate::lang::dap::protocol::*;
use crate::lang::dap::transport;
use crate::lang::interpreter::{ControlMessage, DebugCommand, DebugEvent, Interpreter};
use serde_json::json;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};

/// Runs a Hayashi script under a DAP server.
///
/// `input` is typically `std::io::stdin()`, `output` is typically
/// `std::io::stdout()`. `program` is the path to the `.hay` file to debug.
pub fn run_dap<R: Read + Send + 'static, W: Write + Send + 'static>(
    input: R,
    output: W,
    program: &Path,
) {
    let output = Arc::new(Mutex::new(output));

    let (tx_control, rx_control) = mpsc::channel::<ControlMessage>();
    let tx_control_reader = tx_control.clone();
    drop(tx_control);
    let (tx_event, rx_event) = mpsc::channel::<DebugEvent>();
    let (tx_response, rx_response) = mpsc::channel::<Response>();
    let tx_response_reader = tx_response.clone();

    let seq = Arc::new(std::sync::Mutex::new(0i64));

    // Writer thread: serializes debug events to stdout.
    let out_events = output.clone();
    let seq_events = seq.clone();
    let writer_handle = std::thread::spawn(move || {
        while let Ok(msg) = rx_event.recv() {
            let mut event = debug_event_to_protocol(&msg);
            {
                let mut guard = out_events.lock().unwrap();
                let mut s = seq_events.lock().unwrap();
                *s += 1;
                event.seq = *s;
                if transport::send(&mut *guard, &event).is_err() {
                    break;
                }
            }
            if matches!(msg, DebugEvent::Terminated | DebugEvent::Exited(_)) {
                break;
            }
        }
    });

    // Response writer thread: serializes responses to stdout.
    let out_responses = output.clone();
    let seq_responses = seq.clone();
    let tx_response_event = tx_event.clone();
    let response_handle = std::thread::spawn(move || {
        while let Ok(mut resp) = rx_response.recv() {
            let command = resp.command.clone();
            let mut guard = out_responses.lock().unwrap();
            let mut s = seq_responses.lock().unwrap();
            *s += 1;
            resp.seq = *s;
            if transport::send(&mut *guard, &resp).is_err() {
                break;
            }
            drop(guard);
            // After the initialize response, send the initialized event.
            if command == "initialize" {
                let _ = tx_response_event.send(DebugEvent::Initialized);
            }
        }
    });

    // Reader thread: reads DAP requests from stdin and forwards them.
    let mut reader = BufReader::new(input);
    let reader_handle = std::thread::spawn(move || loop {
        match transport::read_message(&mut reader) {
            Ok(Some(json)) => {
                if let Ok(req) = serde_json::from_str::<Request>(&json) {
                    handle_request(&req, &tx_control_reader, &tx_response_reader);
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    });

    // Main thread: run the interpreter.
    let mut interp = Interpreter::new();
    interp.load_plugins();
    interp.set_current_source(program);

    // Give the interpreter its own event sender; keep a clone for the main thread.
    let tx_event_for_interp = tx_event.clone();
    interp.enable_debug(program.to_path_buf(), tx_event_for_interp, rx_control);

    // Wait for the client to finish configuration before running the script.
    let _ = interp.debug_wait_for_start();

    let result = match std::fs::read_to_string(program) {
        Ok(src) => lang::run_source_verbose(&src, &mut interp, false, Some(program)),
        Err(e) => {
            eprintln!("hay dap: cannot read '{}': {e}", program.display());
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        let _ = tx_event.send(DebugEvent::Output {
            category: "stderr".into(),
            output: format!("{e}"),
        });
    }

    // Send final lifecycle events. Then drop the interpreter (and its event
    // sender) so the writer thread can terminate before we join it.
    let _ = tx_event.send(DebugEvent::Terminated);
    let _ = tx_event.send(DebugEvent::Exited(0));
    drop(interp);
    drop(tx_response);

    let _ = writer_handle.join();
    let _ = response_handle.join();
    let _ = reader_handle.join();
}

fn handle_request(
    req: &Request,
    tx_control: &Sender<ControlMessage>,
    tx_response: &Sender<Response>,
) {
    match req.command.as_str() {
        "continue" => {
            let _ = tx_control.send(ControlMessage::Command(DebugCommand::Continue));
        }
        "next" => {
            let _ = tx_control.send(ControlMessage::Command(DebugCommand::StepOver));
        }
        "stepIn" => {
            let _ = tx_control.send(ControlMessage::Command(DebugCommand::StepIn));
        }
        "stepOut" => {
            let _ = tx_control.send(ControlMessage::Command(DebugCommand::StepOut));
        }
        "pause" => {
            let _ = tx_control.send(ControlMessage::Command(DebugCommand::Pause));
        }
        "disconnect" | "terminate" => {
            let _ = tx_control.send(ControlMessage::Command(DebugCommand::Disconnect));
        }
        _ => {
            // All other requests (initialize, launch, setBreakpoints, etc.)
            // need access to interpreter state, so forward them to the main thread.
            let _ = tx_control.send(ControlMessage::Request(req.clone(), tx_response.clone()));
        }
    }
}

fn debug_event_to_protocol(event: &DebugEvent) -> Event {
    let (name, body) = match event {
        DebugEvent::Initialized => ("initialized", json!({})),
        DebugEvent::Stopped {
            reason,
            description,
            thread_id,
            preserve_focus_hint,
        } => {
            let mut map = serde_json::Map::new();
            map.insert("reason".into(), json!(reason));
            if let Some(d) = description {
                map.insert("description".into(), json!(d));
            }
            map.insert("threadId".into(), json!(thread_id));
            map.insert("preserveFocusHint".into(), json!(preserve_focus_hint));
            ("stopped", json!(map))
        }
        DebugEvent::Output { category, output } => (
            "output",
            json!({
                "category": category,
                "output": output
            }),
        ),
        DebugEvent::Terminated => ("terminated", json!({})),
        DebugEvent::Exited(code) => ("exited", json!({ "exitCode": code })),
    };

    Event {
        seq: 0,
        type_field: "event",
        event: name.into(),
        body: Some(body),
    }
}
