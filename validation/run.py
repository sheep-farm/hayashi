#!/usr/bin/env python3
"""Hayashi empirical validation programme orchestrator.

Reads validation/matrix.yml, runs reference and Hayashi scripts for each case,
compares declared quantities against tolerances, and updates MATRIX.md.
"""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

VALIDATION_DIR = Path(__file__).resolve().parent
ROOT_DIR = VALIDATION_DIR.parent
MATRIX_YML = VALIDATION_DIR / "matrix.yml"
MATRIX_MD = VALIDATION_DIR / "MATRIX.md"


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    log(f"  $ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd or ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )


def check_executable(name: str) -> bool:
    return shutil.which(name) is not None


def python_executable() -> str:
    """Return the Python interpreter to use, preferring a local venv."""
    venv_python = VALIDATION_DIR / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python" if check_executable("python") else "python3"


def parse_hayashi_csv(path: Path) -> dict[str, dict[str, float]]:
    """Parse the CSV produced by Hayashi OLS export from a file."""
    with open(path) as f:
        return parse_hayashi_csv_from_string(f.read())


def parse_hayashi_csv_from_string(text: str) -> dict[str, dict[str, float]]:
    """Parse the CSV produced by Hayashi OLS export from a string.

    Hayashi may print an "Exported OLS → ..." line before the CSV data, so we
    locate the header row and parse from there.
    """
    import csv
    import io

    header = "Variable,Coef,Std_Err"
    start = text.find(header)
    if start == -1:
        raise ValueError(f"CSV header not found in Hayashi output: {text[:200]!r}")

    result = {"coefficients": {}, "standard_errors": {}}
    reader = csv.DictReader(io.StringIO(text[start:]))
    for row in reader:
        var = row.get("Variable")
        coef = row.get("Coef")
        se = row.get("Std_Err")
        if not var or coef is None or se is None or coef == "" or se == "":
            continue
        # Normalise intercept label across implementations.
        if var == "const":
            var = "Intercept"
        result["coefficients"][var] = float(coef)
        result["standard_errors"][var] = float(se)
    return result


def parse_reference_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def approx_equal(a: float, b: float, tol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
        return True
    return abs(a - b) <= tol


def compare_quantities(
    hayashi: dict[str, Any],
    reference: dict[str, Any],
    tolerances: dict[str, float],
) -> tuple[str, list[str]]:
    failures: list[str] = []
    for quantity in tolerances:
        if quantity not in reference:
            failures.append(f"{quantity}: missing in reference")
            continue
        if quantity not in hayashi:
            failures.append(f"{quantity}: missing in Hayashi output")
            continue

        ref_val = reference[quantity]
        hay_val = hayashi[quantity]
        tol = float(tolerances[quantity])

        if isinstance(ref_val, dict):
            # Compare per-coefficient quantities (e.g., coefficients).
            for key in ref_val:
                if key not in hay_val:
                    failures.append(f"{quantity}.{key}: missing in Hayashi")
                    continue
                if not approx_equal(float(hay_val[key]), float(ref_val[key]), tol):
                    failures.append(
                        f"{quantity}.{key}: {hay_val[key]} vs {ref_val[key]} (tol={tol})"
                    )
        elif isinstance(ref_val, (int, float)):
            if not approx_equal(float(hay_val), float(ref_val), tol):
                failures.append(
                    f"{quantity}: {hay_val} vs {ref_val} (tol={tol})"
                )
        else:
            if hay_val != ref_val:
                failures.append(f"{quantity}: {hay_val} != {ref_val}")

    if failures:
        return "fail", failures
    return "pass", []


def run_case(case: dict[str, Any]) -> tuple[str, list[str]]:
    case_id = case["id"]
    case_dir = VALIDATION_DIR / "cases" / case_id
    hayashi_dir = case_dir / "hayashi"
    reference_dir = case_dir / "reference"
    data_dir = case_dir / "data"

    log(f"\n[case] {case_id}: {case['title']}")

    # Ensure data directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Resolve script paths relative to the validation directory.
    reference_scripts = case.get("reference_scripts", {})
    references = case.get("references", [])
    available_refs: list[str] = []
    reference_errors: list[str] = []
    r_res: subprocess.CompletedProcess[str] | None = None
    py_res: subprocess.CompletedProcess[str] | None = None

    if "R" in references and "R" in reference_scripts:
        if not check_executable("Rscript"):
            log("  Rscript not found; skipping R reference")
        else:
            r_script = str(VALIDATION_DIR / reference_scripts["R"])
            r_res = run_command(["Rscript", r_script])
            if r_res.returncode == 0:
                available_refs.append("R")
            else:
                reference_errors.append(f"R script failed:\n{r_res.stderr}")

    if "Python" in references and "Python" in reference_scripts:
        py_exe = python_executable()
        if not Path(py_exe).exists() and not check_executable(py_exe):
            log("  python/python3 not found; skipping Python reference")
        else:
            py_script = str(VALIDATION_DIR / reference_scripts["Python"])
            py_res = run_command([py_exe, py_script])
            if py_res.returncode == 0:
                available_refs.append("Python")
            else:
                log(f"  Python script failed:\n{py_res.stderr}")
                reference_errors.append(f"Python script failed:\n{py_res.stderr}")

    if "Stata" in references and "Stata" in reference_scripts:
        if not check_executable("stata"):
            log("  Stata not found; skipping Stata reference")
        else:
            st_script = str(VALIDATION_DIR / reference_scripts["Stata"])
            st_res = run_command(["stata", "-b", "do", st_script])
            if st_res.returncode == 0:
                available_refs.append("Stata")
            else:
                log(f"  Stata script failed:\n{st_res.stderr}")

    if not available_refs:
        msg = ["No reference implementation could run."] + reference_errors
        return "blocked", msg

    # Run Hayashi script.
    if not check_executable("hay"):
        # Fall back to the local release binary.
        hay_exe = str(ROOT_DIR / "target" / "release" / "hay")
    else:
        hay_exe = "hay"

    hay_script = str(VALIDATION_DIR / case.get("hayashi_script", f"cases/{case_id}/hayashi/run.hay"))
    hay_res = run_command([hay_exe, hay_script])
    if hay_res.returncode != 0:
        return "blocked", [f"Hayashi script failed:\n{hay_res.stderr}"]

    # Parse outputs.
    # Prefer the stdout emitted by the reference/Python scripts; fall back to
    # the written JSON file if stdout is empty (e.g., when run directly).
    reference: dict[str, Any] | None = None
    if py_res and py_res.stdout.strip():
        try:
            reference = json.loads(py_res.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError as e:
            return "blocked", [f"Could not parse reference stdout as JSON: {e}"]
    if reference is None and r_res and r_res.stdout.strip():
        try:
            reference = json.loads(r_res.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError as e:
            return "blocked", [f"Could not parse reference stdout as JSON: {e}"]
    if reference is None:
        expected_json = reference_dir / "expected.json"
        if not expected_json.exists():
            return "blocked", [f"Reference output not found: {expected_json}"]
        reference = parse_reference_json(expected_json)

    # Prefer the stdout emitted by Hayashi; fall back to the written CSV.
    hayashi: dict[str, dict[str, float]] | None = None
    if hay_res.stdout.strip():
        try:
            hayashi = parse_hayashi_csv_from_string(hay_res.stdout)
        except Exception as e:
            return "blocked", [f"Could not parse Hayashi stdout as CSV: {e}"]
    if hayashi is None:
        hayashi_csv = hayashi_dir / "output.csv"
        if not hayashi_csv.exists():
            return "blocked", [f"Hayashi output not found: {hayashi_csv}"]
        hayashi = parse_hayashi_csv(hayashi_csv)

    # Compare declared quantities.
    tolerances = case.get("comparison", {}).get("tolerances", {})
    status, failures = compare_quantities(hayashi, reference, tolerances)

    if failures:
        for f in failures:
            log(f"  FAIL: {f}")
    else:
        log("  PASS")

    return status, failures


def update_matrix_md(cases: list[dict[str, Any]]) -> None:
    lines = [
        "# Hayashi Validation Matrix",
        "",
        "| Family | Dataset | Reference | Status | Blocking Issue | Notes |",
        "|---|---|---:|---|---|---|",
    ]
    for case in cases:
        family = case.get("estimator_family", "")
        dataset = case.get("dataset", {}).get("name", "")
        refs = ", ".join(case.get("references", []))
        status = case.get("status", "not-started")
        issue = case.get("result", {}).get("issues_opened", [])
        issue_str = ", ".join(str(i) for i in issue) if issue else "—"
        notes = case.get("notes", "")
        lines.append(f"| {family} | {dataset} | {refs} | {status} | {issue_str} | {notes} |")

    lines.extend([
        "",
        "## Status legend",
        "",
        "- `pass` — Hayashi matches reference within declared tolerances.",
        "- `fail` — Hayashi differs from reference beyond tolerances.",
        "- `blocked` — cannot run because of a missing feature or bug.",
        "- `not-supported` — estimator/workflow not supported yet.",
        "- `not-started` — registered but not implemented.",
        "",
        "This matrix is generated from `validation/matrix.yml` by `validation/run.py`.",
        "",
    ])
    MATRIX_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    log("Hayashi empirical validation programme")
    log(f"Root: {ROOT_DIR}")

    if not MATRIX_YML.exists():
        log(f"ERROR: {MATRIX_YML} not found")
        return 1

    with open(MATRIX_YML) as f:
        matrix = yaml.safe_load(f) or {}

    registry = matrix.get("cases", [])
    registry_ids = {entry.get("id") for entry in registry if entry.get("id")}

    # Auto-discover every case directory containing a case.yml.
    discovered: list[dict[str, Any]] = []
    for case_yml in sorted(VALIDATION_DIR.glob("cases/*/case.yml")):
        case_id = case_yml.parent.name
        with open(case_yml) as f:
            case = yaml.safe_load(f) or {}
        case["id"] = case_id
        discovered.append(case)

    # Merge registry entries with discovered cases. Registry entries provide
    # optional metadata (notes, dimension) but are not required.
    cases_by_id: dict[str, dict[str, Any]] = {}
    for case in discovered:
        cases_by_id[case["id"]] = case
    for entry in registry:
        case_id = entry.get("id")
        if not case_id or case_id not in cases_by_id:
            continue
        # Registry notes are optional; if present they override the case.yml notes.
        if entry.get("notes"):
            cases_by_id[case_id]["notes"] = entry["notes"]
        cases_by_id[case_id]["dimension"] = entry.get("dimension", cases_by_id[case_id].get("dimension", "numerical"))

    cases = list(cases_by_id.values())
    log(f"Discovered {len(cases)} validation case(s)")
    if not cases:
        log("No validation cases found in validation/cases/*/case.yml")
        return 0

    # Warn about registry entries that no longer exist on disk.
    discovered_ids = {case["id"] for case in cases}
    for case_id in registry_ids - discovered_ids:
        log(f"WARNING: registry entry '{case_id}' has no case.yml on disk; skipping")

    overall_status = "pass"
    for case in cases:
        status, failures = run_case(case)
        case["status"] = status
        if status != "pass":
            overall_status = status
        summary = "; ".join(failures) if failures else "matches reference"
        case.setdefault("result", {})["summary"] = summary

    # Write updated matrix.yml (id + notes + dimension for each discovered case).
    matrix["cases"] = [
        {
            "id": case["id"],
            "notes": case.get("notes", ""),
            "dimension": case.get("dimension", "numerical"),
        }
        for case in cases
    ]
    with open(MATRIX_YML, "w") as f:
        yaml.dump(matrix, f, sort_keys=False, allow_unicode=True)

    # Regenerate MATRIX.md.
    update_matrix_md(cases)

    log(f"\nOverall status: {overall_status}")
    return 0 if overall_status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
