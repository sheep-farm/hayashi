use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn assert_rejected_workbook(fixture: &str, expected_error: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/calamine")
        .join(fixture);
    let mut command = if cfg!(target_os = "linux") {
        // Bound the old parser's allocation while reproducing the regression.
        let mut limited = Command::new("sh");
        limited.args([
            "-c",
            "ulimit -v 524288 || exit 125; exec \"$@\"",
            "hayashi-excel-test",
        ]);
        limited.arg(env!("CARGO_BIN_EXE_hay"));
        limited
    } else {
        Command::new(env!("CARGO_BIN_EXE_hay"))
    };
    let mut child = command
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start the Excel loader subprocess");
    writeln!(
        child.stdin.take().expect("piped stdin"),
        "load \"{}\" as df",
        path.to_string_lossy().replace('\\', "/")
    )
    .expect("write the workbook load command");

    let deadline = Instant::now() + Duration::from_secs(5);
    while child.try_wait().expect("poll Excel loader").is_none() {
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child
                .wait_with_output()
                .expect("reap timed-out Excel loader");
            panic!("{fixture}: parser exceeded five seconds: {output:?}");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let output = child
        .wait_with_output()
        .expect("collect Excel loader output");
    let error = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{fixture}: {error}");
    assert!(error.contains(expected_error), "{fixture}: {error}");
}

#[test]
fn excel_rejects_short_cfb_directory() {
    assert_rejected_workbook("cfb_dir_short_chunk_panic.xls", "sheet is empty");
}

#[test]
fn excel_rejects_cyclic_cfb_difat() {
    assert_rejected_workbook("cfb_difat_oom.xls", "cannot open");
}

#[test]
fn excel_loads_valid_biff5_workbook() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/calamine/biff5_write.xls");
    let (frame, rows) = hayashi_lang::io::excel::load_excel(
        path.to_str().expect("fixture path"),
        Some("SheetJS"),
        None,
        None,
    )
    .expect("load the valid compound workbook");
    assert_eq!(rows, 3);
    assert_eq!(frame.n_rows(), 3);
    assert_eq!(frame.column_names(), ["1", "2", "3", "unnamed"]);
    assert_eq!(frame.get_string("unnamed").unwrap(), ["sheetjs", "0.3", ""]);
}
