use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Request {
    pub seq: i64,
    #[serde(rename = "type")]
    pub type_field: String,
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Response {
    pub seq: i64,
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub request_seq: i64,
    pub success: bool,
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<Value>,
}

impl Response {
    pub fn new(seq: i64, request_seq: i64, command: impl Into<String>, success: bool) -> Self {
        Self {
            seq,
            type_field: "response",
            request_seq,
            success,
            command: command.into(),
            message: None,
            body: None,
        }
    }

    pub fn ok(seq: i64, request_seq: i64, command: impl Into<String>) -> Self {
        Self::new(seq, request_seq, command, true)
    }

    pub fn err(
        seq: i64,
        request_seq: i64,
        command: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let mut r = Self::new(seq, request_seq, command, false);
        r.message = Some(message.into());
        r
    }

    pub fn with_body(mut self, body: impl Serialize) -> Self {
        self.body = Some(serde_json::to_value(body).unwrap_or(Value::Null));
        self
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Event {
    pub seq: i64,
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub event: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<Value>,
}

impl Event {
    pub fn new(event: impl Into<String>, body: impl Serialize) -> Self {
        Self {
            seq: 0,
            type_field: "event",
            event: event.into(),
            body: Some(serde_json::to_value(body).unwrap_or(Value::Null)),
        }
    }

    pub fn initialized() -> Self {
        Self {
            seq: 0,
            type_field: "event",
            event: "initialized".into(),
            body: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Capabilities {
    pub supports_configuration_done_request: bool,
    pub supports_exception_info_request: bool,
    pub supports_set_variable: bool,
    pub supports_step_back: bool,
    pub supports_restart_request: bool,
    pub supports_evaluate_for_hovers: bool,
    pub supports_value_formatting_options: bool,
    pub supports_function_breakpoints: bool,
    pub supports_conditional_breakpoints: bool,
    pub supports_hit_conditional_breakpoints: bool,
    pub supports_log_points: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            supports_configuration_done_request: true,
            supports_exception_info_request: true,
            supports_set_variable: false,
            supports_step_back: false,
            supports_restart_request: false,
            supports_evaluate_for_hovers: false,
            supports_value_formatting_options: false,
            supports_function_breakpoints: false,
            supports_conditional_breakpoints: false,
            supports_hit_conditional_breakpoints: false,
            supports_log_points: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StackFrame {
    pub id: i64,
    pub name: String,
    pub source: Option<Source>,
    pub line: i64,
    pub column: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Scope {
    pub name: String,
    pub presentation_hint: Option<String>,
    pub variables_reference: i64,
    pub expensive: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Variable {
    pub name: String,
    pub value: String,
    #[serde(rename = "type")]
    pub type_field: Option<String>,
    pub variables_reference: i64,
    pub named_variables: Option<i64>,
    pub indexed_variables: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Thread {
    pub id: i64,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Breakpoint {
    pub line: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetBreakpointsArguments {
    pub source: Source,
    pub breakpoints: Vec<SourceBreakpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceBreakpoint {
    pub line: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaunchArguments {
    #[serde(rename = "type")]
    pub type_field: String,
    pub name: String,
    pub request: String,
    pub program: String,
    #[serde(flatten)]
    pub extra: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StackTraceArguments {
    pub thread_id: i64,
    pub start_frame: Option<i64>,
    pub levels: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopesArguments {
    pub frame_id: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VariablesArguments {
    pub variables_reference: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluateArguments {
    pub expression: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_id: Option<i64>,
}
