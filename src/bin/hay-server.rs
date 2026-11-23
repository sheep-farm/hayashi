//! `hay-server` — JSON-lines data-plane server for embedding Hayashi.
//!
//! One request/response per line. The server keeps persistent sessions so a
//! host (QNesto, tests, other languages) can bind typed inputs, execute
//! snippets, and read typed outputs without spawning a process per node.
//!
//! Model results are passed by reference inside a session (`set_ref`); full
//! model JSON is exported for inspection. DataFrame/scalar/list/dict values
//! can be sent and received as JSON.

use hayashi_lang::lang::host_api::{self, Session};
use serde_json::{json, Value as Json};
use std::collections::HashMap;
use std::io::{BufRead, Write};

fn main() {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut sessions: HashMap<String, Session> = HashMap::new();
    let mut next_session = 0u64;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let request: Json = match serde_json::from_str(line) {
            Ok(request) => request,
            Err(error) => {
                let response = json!({
                    "id": Json::Null,
                    "ok": false,
                    "error": format!("invalid JSON: {error}"),
                });
                let _ = writeln!(stdout, "{response}");
                let _ = stdout.flush();
                continue;
            }
        };
        let id = request.get("id").cloned().unwrap_or(Json::Null);
        let command = request
            .get("cmd")
            .and_then(Json::as_str)
            .unwrap_or_default();
        let session_id = request
            .get("session")
            .and_then(Json::as_str)
            .unwrap_or_default();

        let result: Result<Json, String> = (|| match command {
            "ping" => Ok(json!({"pong": true, "version": env!("CARGO_PKG_VERSION")})),
            "create" => {
                next_session += 1;
                let id = format!("s{next_session}");
                sessions.insert(id.clone(), Session::new());
                Ok(json!({"session": id}))
            }
            "dispose" => {
                sessions.remove(session_id);
                Ok(json!({"disposed": session_id}))
            }
            "reset" => with_session(&mut sessions, session_id, |session| {
                session.reset();
                Ok(json!({"reset": true}))
            }),
            "clear" => with_session(&mut sessions, session_id, |session| {
                session.clear();
                Ok(json!({"cleared": true}))
            }),
            "remove" => {
                let name = request
                    .get("name")
                    .and_then(Json::as_str)
                    .ok_or_else(|| "name is required".to_string())?;
                with_session(&mut sessions, session_id, |session| {
                    session.remove(name);
                    Ok(json!({"removed": name}))
                })
            }
            "set" => {
                let name = request
                    .get("name")
                    .and_then(Json::as_str)
                    .ok_or_else(|| "name is required".to_string())?;
                let value = request
                    .get("value")
                    .ok_or_else(|| "value is required".to_string())?;
                let parsed = host_api::value_from_json(value)?;
                with_session(&mut sessions, session_id, |session| {
                    session.set_value(name, parsed)?;
                    Ok(json!({"set": name}))
                })
            }
            "set_ref" => {
                let name = request
                    .get("name")
                    .and_then(Json::as_str)
                    .ok_or_else(|| "name is required".to_string())?;
                let from = request
                    .get("from")
                    .and_then(Json::as_str)
                    .ok_or_else(|| "from is required".to_string())?;
                with_session(&mut sessions, session_id, |session| {
                    session.set_ref(name, from)?;
                    Ok(json!({"set": name, "from": from}))
                })
            }
            "get" => {
                let name = request
                    .get("name")
                    .and_then(Json::as_str)
                    .ok_or_else(|| "name is required".to_string())?;
                with_session(&mut sessions, session_id, |session| {
                    session
                        .get_value(name)
                        .map(|value| host_api::value_to_json(&value))
                        .ok_or_else(|| format!("unknown variable '{name}'"))
                })
            }
            "get_summary" => {
                let name = request
                    .get("name")
                    .and_then(Json::as_str)
                    .ok_or_else(|| "name is required".to_string())?;
                with_session(&mut sessions, session_id, |session| {
                    session
                        .get_value(name)
                        .map(|value| {
                            json!({
                                "kind": host_api::value_kind(&value),
                                "summary": host_api::value_summary(&value),
                            })
                        })
                        .ok_or_else(|| format!("unknown variable '{name}'"))
                })
            }
            "list" => with_session(&mut sessions, session_id, |session| {
                let variables: Vec<Json> = session
                    .list_variables()
                    .into_iter()
                    .map(|(name, kind, summary)| {
                        json!({"name": name, "kind": kind, "summary": summary})
                    })
                    .collect();
                Ok(json!({"variables": variables}))
            }),
            "execute" => {
                let source = request
                    .get("source")
                    .and_then(Json::as_str)
                    .ok_or_else(|| "source is required".to_string())?;
                let inputs = request
                    .get("inputs")
                    .and_then(Json::as_object)
                    .cloned()
                    .unwrap_or_default();
                let outputs: Vec<String> = request
                    .get("outputs")
                    .and_then(Json::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Json::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                with_session(&mut sessions, session_id, |session| {
                    for (name, spec) in &inputs {
                        if let Some(reference) = spec.get("ref").and_then(Json::as_str) {
                            session.set_ref(name, reference)?;
                        } else if let Some(value) = spec.get("value") {
                            session.set_value(name, host_api::value_from_json(value)?)?;
                        } else {
                            return Err(format!("input '{name}' needs 'value' or 'ref'"));
                        }
                    }
                    // Missing outputs must not keep a value from a previous run.
                    for name in &outputs {
                        session.remove(name);
                    }
                    let result = session.run(source);
                    let mut exported = serde_json::Map::new();
                    for name in &outputs {
                        if let Some(value) = session.get_value(name) {
                            exported.insert(name.clone(), host_api::value_to_json(&value));
                        }
                    }
                    Ok(json!({
                        "output": result.output,
                        "stderr": result.stderr,
                        "error": result.error,
                        "outputs": exported,
                    }))
                })
            }
            "run" | "eval" => {
                let source = request
                    .get("source")
                    .and_then(Json::as_str)
                    .ok_or_else(|| "source is required".to_string())?;
                with_session(&mut sessions, session_id, |session| {
                    let result = session.run(source);
                    Ok(json!({
                        "output": result.output,
                        "stderr": result.stderr,
                        "error": result.error,
                    }))
                })
            }
            "" => Err("cmd is required".into()),
            other => Err(format!("unknown command '{other}'")),
        })();

        let response = match result {
            Ok(result) => json!({"id": id, "ok": true, "result": result}),
            Err(error) => json!({"id": id, "ok": false, "error": error}),
        };
        let _ = writeln!(stdout, "{response}");
        let _ = stdout.flush();
    }
}

fn with_session(
    sessions: &mut HashMap<String, Session>,
    session_id: &str,
    action: impl FnOnce(&mut Session) -> Result<Json, String>,
) -> Result<Json, String> {
    let session = sessions
        .get_mut(session_id)
        .ok_or_else(|| format!("unknown session '{session_id}'"))?;
    action(session)
}
