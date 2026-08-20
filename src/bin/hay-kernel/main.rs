use anyhow::{Context, Result};
use jupyter_zmq_client::user_data_dir;

mod kernel;
mod mime;

use kernel::HayashiKernel;

const USAGE: &str = "Usage: hay-kernel [--connection-file <path> | --install]";

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--install") {
        install_kernel().context("failed to install hayashi kernel")?;
        return Ok(());
    }

    let connection_file = match args.iter().position(|a| a == "--connection-file") {
        Some(i) => args.get(i + 1).cloned(),
        None => args.get(1).cloned(),
    };

    let connection_file = connection_file.context(USAGE)?;
    HayashiKernel::start(&connection_file)
        .await
        .context("hayashi kernel failed")?;
    Ok(())
}

fn install_kernel() -> Result<()> {
    let exe = std::env::current_exe().context("failed to get current executable path")?;

    let data_dir = user_data_dir().context("failed to determine user data directory")?;
    let kernel_dir = data_dir.join("kernels").join("hayashi");
    std::fs::create_dir_all(&kernel_dir).context("failed to create kernel directory")?;

    let kernel_json = serde_json::json!({
        "argv": [exe.to_string_lossy(), "--connection-file", "{connection_file}"],
        "display_name": "Hayashi",
        "language": "hayashi",
        "interrupt_mode": "message",
        "metadata": {
            "debugger": false
        }
    });

    let spec_path = kernel_dir.join("kernel.json");
    std::fs::write(
        &spec_path,
        serde_json::to_string_pretty(&kernel_json).context("failed to serialize kernel spec")?,
    )
    .context("failed to write kernel spec")?;

    println!("Installed Hayashi kernel spec to {}", spec_path.display());
    Ok(())
}
