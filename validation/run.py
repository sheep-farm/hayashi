:
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


def parse_hayashi_csv(path: Path) -> dict[str, dict[str, float]]:
    """Parse the CSV produced by Hayashi OLS export."""
    import csv
    result = {"coefficients": {}, "standard_errors": {}}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = row["Variable"]
            result["coefficients"][var] = float(row["Coef"])
            result["standard_errors"][var] = float(row["Std_Err"])
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
        tol = tolerances[quantity]

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

    # Run reference scripts first (they may produce the dataset).
    reference_scripts = case.get("reference_scripts", {})
    references = case.get("references", [])

    if "R" in references and "R" in reference_scripts:
        if not check_executable("Rscript"):
            return "blocked", ["Rscript not found in PATH"]
        r_script = reference_scripts["R"]
        r_res = run_command(["Rscript", r_script])
        if r_res.returncode != 0:
            return "blocked", [f"R script failed:\n{r_res.stderr}"]

    if "Python" in references and "Python" in reference_scripts:
        if not check_executable("python") and not check_executable("python3"):
            return "blocked", ["python/python3 not found in PATH"]
        py_exe = "python" if check_executable("python") else "python3"
        py_script = reference_scripts["Python"]
        py_res = run_command([py_exe, py_script])
        if py_res.returncode != 0:
            return "blocked", [f"Python script failed:\n{py_res.stderr}"]

    if "Stata" in references and "Stata" in reference_scripts:
        if not check_executable("stata"):
            log("  Stata not found; skipping Stata reference")
        else:
            st_script = reference_scripts["Stata"]
            st_res = run_command(["stata", "-b", "do", st_script])
            if st_res.returncode != 0:
                log(f"  Stata script failed:\n{st_res.stderr}")

    # Run Hayashi script.
    if not check_executable("hay"):
        # Fall back to the local debug binary.
        hay_exe = str(ROOT_DIR / "target" / "debug" / "hay")
    else:
        hay_exe = "hay"

    hay_script = case.get("hayashi_script", f"cases/{case_id}/hayashi/run.hay")
    hay_res = run_command([hay_exe, hay_script])
    if hay_res.returncode != 0:
        return "blocked", [f"Hayashi script failed:\n{hay_res.stderr}"]

    # Parse outputs.
    expected_json = reference_dir / "expected.json"
    hayashi_csv = hayashi_dir / "output.csv"

    if not expected_json.exists():
        return "blocked", [f"Reference output not found: {expected_json}"]
    if not hayashi_csv.exists():
        return "blocked", [f"Hayashi output not found: {hayashi_csv}"]

    reference = parse_reference_json(expected_json)
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

    cases = matrix.get("cases", [])
    if not cases:
        log("No cases registered in matrix.yml")
        return 0

    overall_status = "pass"
    for case in cases:
        status, failures = run_case(case)
        case["status"] = status
        if status != "pass":
            overall_status = status
        case.setdefault("result", {})["summary"] = "; ".join(failures) if failures else "matches reference"

    # Write updated matrix.yml.
    with open(MATRIX_YML, "w") as f:
        yaml.dump(matrix, f, sort_keys=False, allow_unicode=True)

    # Regenerate MATRIX.md.
    update_matrix_md(cases)

    log(f"\nOverall status: {overall_status}")
    return 0 if overall_status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
