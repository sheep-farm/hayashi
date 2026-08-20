use anyhow::{Context, Result};
use gag::BufferRedirect;
use jupyter_protocol::messaging::{JupyterMessage, JupyterMessageContent};
use jupyter_protocol::{
    connection_info::ConnectionInfo, ErrorOutput, ExecuteInput, ExecuteReply, ExecuteRequest,
    ExecuteResult, ExecutionCount, KernelInfoReply, LanguageInfo, Media, ReplyStatus,
    ShutdownReply, Status, StreamContent,
};
use jupyter_zmq_client::{
    create_kernel_control_connection, create_kernel_heartbeat_connection,
    create_kernel_iopub_connection, create_kernel_shell_connection, create_kernel_stdin_connection,
    KernelControlConnection, KernelIoPubConnection, KernelShellConnection, KernelStdinConnection,
};
use tokio::task;
use uuid::Uuid;
use zeromq::{SocketRecv as _, SocketSend as _};

use hayashi_lang::{lang, Interpreter};

use crate::mime::value_to_media;

pub struct HayashiKernel {
    execution_count: ExecutionCount,
    iopub: KernelIoPubConnection,
    shell: KernelShellConnection,
    _stdin: KernelStdinConnection,
    worker_tx: std::sync::mpsc::Sender<ExecuteJob>,
}

struct ExecuteJob {
    code: String,
    reply_tx: tokio::sync::oneshot::Sender<ExecuteJobResult>,
}

struct ExecuteJobResult {
    status: ReplyStatus,
    error: Option<String>,
    stdout: Option<String>,
    stderr: Option<String>,
    result: Option<Media>,
}

impl HayashiKernel {
    pub async fn start(connection_file: &str) -> Result<()> {
        let conn_info: ConnectionInfo = serde_json::from_str(
            &std::fs::read_to_string(connection_file).context("failed to read connection file")?,
        )
        .context("failed to parse connection file")?;

        let session_id = Uuid::new_v4().to_string();

        let mut heartbeat = create_kernel_heartbeat_connection(&conn_info)
            .await
            .context("failed to create heartbeat connection")?;
        let shell = create_kernel_shell_connection(&conn_info, &session_id)
            .await
            .context("failed to create shell connection")?;
        let stdin = create_kernel_stdin_connection(&conn_info, &session_id)
            .await
            .context("failed to create stdin connection")?;
        let mut control = create_kernel_control_connection(&conn_info, &session_id)
            .await
            .context("failed to create control connection")?;
        let iopub = create_kernel_iopub_connection(&conn_info, &session_id)
            .await
            .context("failed to create iopub connection")?;

        let (worker_tx, worker_rx) = std::sync::mpsc::channel();
        task::spawn_blocking(move || interpreter_worker(worker_rx));

        let mut kernel = Self {
            execution_count: ExecutionCount::new(0),
            iopub,
            shell,
            _stdin: stdin,
            worker_tx,
        };

        kernel
            .iopub
            .send(JupyterMessage::new(Status::starting(), None))
            .await?;
        kernel
            .iopub
            .send(JupyterMessage::new(Status::idle(), None))
            .await?;

        let hb_handle = tokio::spawn(async move {
            while let Ok(msg) = heartbeat.socket.recv().await {
                let _ = heartbeat.socket.send(msg).await;
            }
        });

        let control_handle = tokio::spawn(async move {
            if let Err(err) = handle_control(&mut control).await {
                eprintln!("control channel error: {err}");
            }
        });

        let shell_handle = tokio::spawn(async move {
            if let Err(err) = kernel.handle_shell().await {
                eprintln!("shell channel error: {err}");
            }
        });

        let _ = tokio::try_join!(hb_handle, control_handle, shell_handle);
        Ok(())
    }

    async fn handle_shell(&mut self) -> jupyter_zmq_client::Result<()> {
        loop {
            let msg = self.shell.read().await?;
            match &msg.content {
                JupyterMessageContent::KernelInfoRequest(_) => {
                    self.iopub.send(Status::busy().as_child_of(&msg)).await?;
                    let reply = kernel_info_reply().as_child_of(&msg);
                    self.shell.send(reply).await?;
                    self.iopub.send(Status::idle().as_child_of(&msg)).await?;
                }
                JupyterMessageContent::ExecuteRequest(req) => {
                    let req = req.clone();
                    self.handle_execute(msg, &req).await?;
                }
                JupyterMessageContent::ShutdownRequest(req) => {
                    let reply = ShutdownReply {
                        restart: req.restart,
                        status: ReplyStatus::Ok,
                        error: None,
                    }
                    .as_child_of(&msg);
                    self.shell.send(reply).await?;
                    std::process::exit(0);
                }
                _ => {}
            }
        }
    }

    async fn handle_execute(
        &mut self,
        msg: JupyterMessage,
        req: &ExecuteRequest,
    ) -> jupyter_zmq_client::Result<()> {
        self.iopub.send(Status::busy().as_child_of(&msg)).await?;

        self.execution_count.increment();
        let count = self.execution_count;

        self.iopub
            .send(
                ExecuteInput {
                    code: req.code.clone(),
                    execution_count: count,
                }
                .as_child_of(&msg),
            )
            .await?;

        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        let job = ExecuteJob {
            code: req.code.clone(),
            reply_tx,
        };
        if self.worker_tx.send(job).is_err() {
            let reply = ExecuteReply {
                status: ReplyStatus::Error,
                execution_count: count,
                payload: Vec::new(),
                user_expressions: None,
                error: None,
            }
            .as_child_of(&msg);
            self.shell.send(reply).await?;
            return Ok(());
        }

        let output = match reply_rx.await {
            Ok(out) => out,
            Err(_) => ExecuteJobResult {
                status: ReplyStatus::Error,
                error: Some("worker crashed".to_string()),
                stdout: None,
                stderr: None,
                result: None,
            },
        };

        if let Some(text) = output.stdout {
            self.iopub
                .send(StreamContent::stdout(&text).as_child_of(&msg))
                .await?;
        }
        if let Some(text) = output.stderr {
            self.iopub
                .send(StreamContent::stderr(&text).as_child_of(&msg))
                .await?;
        }

        if let Some(media) = output.result {
            self.iopub
                .send(ExecuteResult::new(count, media).as_child_of(&msg))
                .await?;
        }

        if let Some(err) = output.error {
            self.iopub
                .send(
                    ErrorOutput {
                        ename: "RuntimeError".to_string(),
                        evalue: err.clone(),
                        traceback: vec![err],
                    }
                    .as_child_of(&msg),
                )
                .await?;
        }

        self.iopub.send(Status::idle().as_child_of(&msg)).await?;

        let reply = ExecuteReply {
            status: output.status,
            execution_count: count,
            payload: Vec::new(),
            user_expressions: None,
            error: None,
        }
        .as_child_of(&msg);
        self.shell.send(reply).await?;

        Ok(())
    }
}

fn interpreter_worker(rx: std::sync::mpsc::Receiver<ExecuteJob>) {
    let mut interp = Interpreter::new();
    interp.load_plugins();

    while let Ok(job) = rx.recv() {
        let result = execute_code(&mut interp, &job.code);
        let _ = job.reply_tx.send(result);
    }
}

fn execute_code(interp: &mut Interpreter, code: &str) -> ExecuteJobResult {
    let mut output = ExecuteJobResult {
        status: ReplyStatus::Ok,
        error: None,
        stdout: None,
        stderr: None,
        result: None,
    };

    let stdout_redirect = match BufferRedirect::stdout() {
        Ok(r) => Some(r),
        Err(err) => {
            eprintln!("failed to redirect stdout: {err}");
            None
        }
    };
    let stderr_redirect = match BufferRedirect::stderr() {
        Ok(r) => Some(r),
        Err(err) => {
            eprintln!("failed to redirect stderr: {err}");
            None
        }
    };

    interp.set_auto_display(false);

    let result = lang::run_source(code, interp);

    if let Some(mut r) = stdout_redirect {
        let mut buf = String::new();
        let _ = std::io::Read::read_to_string(&mut r, &mut buf);
        if !buf.is_empty() {
            output.stdout = Some(buf);
        }
    }
    if let Some(mut r) = stderr_redirect {
        let mut buf = String::new();
        let _ = std::io::Read::read_to_string(&mut r, &mut buf);
        if !buf.is_empty() {
            output.stderr = Some(buf);
        }
    }

    match result {
        Ok(()) => {
            if let Some(val) = interp.take_last_expr_value() {
                if !matches!(val, hayashi_lang::lang::interpreter::value::Value::Nil) {
                    output.result = Some(value_to_media(&val));
                }
            }
        }
        Err(err) => {
            let err_string = format!("{err}");
            output.status = ReplyStatus::Error;
            output.error = Some(err_string);
        }
    }

    output
}

fn kernel_info_reply() -> KernelInfoReply {
    KernelInfoReply {
        status: ReplyStatus::Ok,
        protocol_version: "5.3".to_string(),
        implementation: "hayashi".to_string(),
        implementation_version: env!("CARGO_PKG_VERSION").to_string(),
        language_info: LanguageInfo {
            name: "hayashi".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            mimetype: Some("text/x-hayashi".to_string()),
            file_extension: Some(".hay".to_string()),
            pygments_lexer: None,
            codemirror_mode: Some(jupyter_protocol::CodeMirrorMode::Simple(
                "hayashi".to_string(),
            )),
            nbconvert_exporter: None,
        },
        banner: "Hayashi kernel".to_string(),
        help_links: Vec::new(),
        debugger: false,
        error: None,
    }
}

async fn handle_control(control: &mut KernelControlConnection) -> jupyter_zmq_client::Result<()> {
    loop {
        let msg = control.read().await?;
        match &msg.content {
            JupyterMessageContent::KernelInfoRequest(_) => {
                let reply = kernel_info_reply().as_child_of(&msg);
                control.send(reply).await?;
            }
            JupyterMessageContent::ShutdownRequest(req) => {
                let reply = ShutdownReply {
                    restart: req.restart,
                    status: ReplyStatus::Ok,
                    error: None,
                }
                .as_child_of(&msg);
                control.send(reply).await?;
                std::process::exit(0);
            }
            _ => {}
        }
    }
}
