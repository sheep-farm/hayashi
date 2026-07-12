#!/usr/bin/env python3
"""Hayashi empirical validation programme orchestrator.

Reads validation/matrix.yml, runs reference and Hayashi scripts for each case,
compares declared quantities against tolerances, and updates MATRIX.md.
"""

import argparse
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


def _is_number(token: str) -> bool:
    """Return True if token is a valid signed decimal number."""
    try:
        float(token)
        return True
    except ValueError:
        return False


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
    """Parse the CSV produced by Hayashi OLS/WLS export from a string.

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


def parse_hayashi_txt_table(text: str) -> dict[str, dict[str, float]]:
    """Parse a plain-text coefficient table from Hayashi (IV, logit, probit, poisson, etc.).

    The table may use pipe separators (IV/logit/probit) or aligned whitespace
    (Poisson-style statsmodels output). We detect the header by looking for
    "coef" and "std err" and then parse the data rows accordingly.
    """
    import re

    # Locate the header line by looking for the column titles (case-insensitive).
    # Accept both "std err" / "Std Err" and "SE" / "stderr" spellings.
    lines = text.splitlines()
    start_idx = -1
    for i, line in enumerate(lines):
        line_lower = line.lower()
        has_coef = "coef" in line_lower
        has_se = (
            "std err" in line_lower
            or "stderr" in line_lower
            or "fm-se" in line_lower
            or line_lower.strip().endswith(" se")
            or (" se" in line_lower and " sse" not in line_lower)
        )
        if has_coef and has_se:
            start_idx = i
            break
    if start_idx == -1:
        raise ValueError(f"Text table header not found in Hayashi output: {text[:200]!r}")

    header = lines[start_idx]
    pipe_delimited = " | " in header
    result = {"coefficients": {}, "standard_errors": {}}

    if pipe_delimited:
        # Match lines like: "educ       |    0.1320 |    0.0540 |    2.440 |    0.015"
        pattern = re.compile(r"^\s*(\S.*?)\s*\|\s*([-+]?\d+\.?\d*)\s*\|\s*([-+]?\d+\.?\d*)\s*\|")
        for line in lines[start_idx + 1:]:
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("="):
                continue
            m = pattern.match(line)
            if not m:
                continue
            var = m.group(1).strip()
            if var.lower() == "variable":
                continue
            if "(omitted)" in line:
                continue
            # Normalise intercept label across implementations.
            if var == "const":
                var = "Intercept"
            result["coefficients"][var] = float(m.group(2))
            result["standard_errors"][var] = float(m.group(3))
    else:
        # Whitespace-aligned table (Poisson style). Header row is followed by
        # a divider and then rows where the first token is the variable name.
        for line in lines[start_idx + 1:]:
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("="):
                continue
            # Stop at the start of a secondary section (e.g., Heckman selection equation).
            lower = line.lower()
            if any(marker in lower for marker in ("equação de seleção", "selection equation", "seleção", "γ̂", "gamma")):
                break
            # Split on whitespace and reconstruct the variable name, allowing
            # names like "lambda (IMR)" where the first token is not a number.
            parts = line.split()
            if len(parts) < 3:
                continue
            i = 0
            while i < len(parts) and not _is_number(parts[i]):
                i += 1
            if i == 0 or i + 2 > len(parts):
                continue
            var = " ".join(parts[:i])
            try:
                coef = float(parts[i])
                se = float(parts[i + 1])
            except ValueError:
                continue
            # Normalise intercept label across implementations.
            if var == "const":
                var = "Intercept"
            result["coefficients"][var] = coef
            result["standard_errors"][var] = se

    return result


def parse_hayashi_local_level(text: str) -> dict[str, float]:
    """Parse the plain-text output of a local-level Kalman filter result.

    Expects a block like:

        =================== Local-Level Kalman Filter ===================
        Observations:               690
        sigma_obs:             2.113422
        sigma_state:           0.000004
        Log-likelihood:      -1499.7967
        ===========================================================
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("sigma_obs:"):
            result["sigma_obs"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("sigma_state:"):
            result["sigma_state"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Log-likelihood:"):
            result["log_likelihood"] = float(line.split(":", 1)[1].strip())
    if "sigma_obs" not in result or "sigma_state" not in result:
        raise ValueError(f"Local-level Kalman output not found in Hayashi stdout: {text[:200]!r}")
    return result


def parse_hayashi_pca(text: str) -> dict[str, dict[str, float]]:
    """Parse PCA eigenvalues, variance ratios, and loadings from Hayashi text."""
    import re

    eigenvalues: dict[str, float] = {}
    variance_ratios: dict[str, float] = {}
    loadings: dict[str, float] = {}
    in_components = False
    in_loadings = False

    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if "component" in lower and "eigenvalue" in lower:
            in_components = True
            in_loadings = False
            continue
        if lower == "loadings":
            in_components = False
            in_loadings = False
            continue
        if in_components:
            match = re.match(
                r"^(PC\d+)\s+([-+]?\d*\.?\d+)\s+[-+]?\d*\.?\d+\s+"
                r"([-+]?\d*\.?\d+)$",
                stripped,
            )
            if match:
                component = match.group(1)
                variance_ratios[component] = float(match.group(2))
                eigenvalues[component] = float(match.group(3))
            continue
        if not in_loadings and stripped.startswith("Variable"):
            in_loadings = True
            continue
        if in_loadings:
            parts = stripped.split()
            if len(parts) < 2 or parts[0].startswith("="):
                continue
            variable = parts[0]
            try:
                values = [float(value) for value in parts[1:]]
            except ValueError:
                continue
            for index, value in enumerate(values, start=1):
                loadings[f"{variable}:PC{index}"] = abs(value)

    if not eigenvalues or not variance_ratios or not loadings:
        raise ValueError(f"PCA output not found in Hayashi stdout: {text[:300]!r}")
    return {
        "explained_variance": eigenvalues,
        "explained_variance_ratio": variance_ratios,
        "absolute_loadings": loadings,
    }


def parse_reference_json(stdout: str) -> dict[str, Any] | None:
    """Extract JSON from reference stdout, tolerating pretty-printed output."""
    text = stdout.strip()
    # Fast path: single-line JSON emitted by toJSON(..., pretty = FALSE).
    try:
        return json.loads(text.splitlines()[-1])
    except json.JSONDecodeError:
        pass
    # Fallback: find the largest JSON object/array in the output.
    for start_char, end_char in ("{", "}"), ("[", "]"):
        start = text.find(start_char)
        if start == -1:
            continue
        # Search for the matching outer object by bracket counting.
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def normalise_intercept(data: dict[str, Any]) -> dict[str, Any]:
    """Rename intercept labels ('const', '_cons' or '(Intercept)') to 'Intercept' and clean up Heckman lambda label."""
    for key in ("coefficients", "standard_errors"):
        if key not in data:
            continue
        d = data[key]
        for src in ("const", "_cons", "(Intercept)"):
            if src in d:
                d["Intercept"] = d.pop(src)
        # Hayashi prints the inverse Mills ratio as "lambda (IMR)".
        if "lambda (IMR)" in d:
            d["lambda_IMR"] = d.pop("lambda (IMR)")
    return data


def parse_hayashi_psm(text: str) -> dict[str, dict[str, float]]:
    """Parse ATT and SE from the PSM summary block.

    Used when the generic coefficient table is not present (e.g. before the
    parsable Parameters table was added to the PSM display).
    """
    import re

    m = re.search(r"ATT\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", text)
    se_m = re.search(r"SE\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", text)
    if not m or not se_m:
        raise ValueError(f"Could not parse PSM ATT/SE from Hayashi output: {text[:200]!r}")
    return {
        "coefficients": {"ATT": float(m.group(1))},
        "standard_errors": {"ATT": float(se_m.group(1))},
    }


def parse_hayashi_rd(text: str) -> dict[str, dict[str, float]]:
    """Parse τ̂ and SE from the RDD summary block."""
    import re

    # Find the treatment-effect line. It may appear just below a header such as
    # "Efeito de Tratamento (τ̂):".
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "τ̂" in line or "Efeito de Tratamento" in line:
            # The next non-empty line contains the numbers.
            for j in range(i + 1, len(lines)):
                candidate = lines[j].strip()
                if not candidate:
                    continue
                numbers = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", candidate)
                if len(numbers) >= 2:
                    return {
                        "coefficients": {"tau": float(numbers[0])},
                        "standard_errors": {"tau": float(numbers[1])},
                    }
                break
            break
    raise ValueError(f"Could not parse RDD tau/SE from Hayashi output: {text[:200]!r}")


def parse_hayashi_km(text: str) -> dict[str, dict[str, float]]:
    """Parse a Kaplan-Meier survival table at selected time points."""
    import re

    lines = text.splitlines()
    start_idx = -1
    for i, line in enumerate(lines):
        if re.match(r"^\s*Time\s+S\(t\)", line):
            start_idx = i
            break
    if start_idx == -1:
        raise ValueError(f"KM table header not found in Hayashi output: {text[:200]!r}")

    curve: dict[float, float] = {}
    for line in lines[start_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("="):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            s = float(parts[1])
        except ValueError:
            continue
        curve[t] = s

    times = [10, 20, 30, 40, 50, 60, 70]
    result: dict[str, float] = {}
    for t in times:
        available = [tt for tt in curve if tt <= t]
        if not available:
            raise ValueError(f"No KM estimate available at or before t={t}")
        result[f"t{t}"] = curve[max(available)]

    if not result:
        raise ValueError(f"Could not parse KM survival probabilities: {text[:200]!r}")
    return {"survival_probabilities": result}


def parse_hayashi_margins(text: str) -> dict[str, dict[str, float]]:
    """Parse a Hayashi average-marginal-effects table."""
    import re

    lines = text.splitlines()
    start_idx = -1
    for i, line in enumerate(lines):
        lower = line.lower()
        if "dy/dx" in lower and ("std.err" in lower or "std err" in lower):
            start_idx = i
            break
    if start_idx == -1:
        raise ValueError(f"Margins table header not found in Hayashi output: {text[:200]!r}")

    result: dict[str, dict[str, float]] = {
        "marginal_effects": {},
        "standard_errors": {},
    }
    pattern = re.compile(
        r"^\s*(\S.*?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
    )

    for line in lines[start_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("="):
            continue
        lower = stripped.lower()
        if lower.startswith("n ="):
            break
        match = pattern.match(line)
        if not match:
            continue
        var = match.group(1).strip()
        if var.lower() == "variable":
            continue
        result["marginal_effects"][var] = float(match.group(2))
        result["standard_errors"][var] = float(match.group(3))

    if not result["marginal_effects"]:
        raise ValueError(f"Could not parse margins rows from Hayashi output: {text[:200]!r}")
    return result


def parse_hayashi_synth(text: str) -> dict[str, dict[str, float]]:
    """Parse the post-treatment effect table and return the mean ATT."""
    import re

    effects: list[float] = []
    lines = text.splitlines()
    for line in lines:
        # Post-treatment rows are marked with an asterisk.
        if "*" not in line:
            continue
        # Extract all numeric tokens.
        numbers = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", line)
        if len(numbers) >= 3:
            # The right-most column is the effect.
            effects.append(float(numbers[-1]))
    if not effects:
        raise ValueError(f"Could not parse synthetic-control post-treatment effects: {text[:200]!r}")
    att = sum(effects) / len(effects)
    return {"coefficients": {"ATT": att}}


def parse_hayashi_svar(text: str) -> dict[str, dict[str, float]]:
    """Parse SVAR A and B matrices from Hayashi text output."""
    import re

    lines = text.splitlines()
    section: str | None = None
    rows: list[list[float]] = []
    a_matrix: list[list[float]] = []
    b_matrix: list[list[float]] = []

    for line in lines:
        stripped = line.strip()
        if "A Matrix" in stripped:
            section = "A"
            rows = []
            continue
        if "B Matrix" in stripped:
            if section == "A":
                a_matrix = rows
            section = "B"
            rows = []
            continue
        if stripped.startswith("[") and stripped.endswith("]") and section in ("A", "B"):
            numbers = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", stripped)
            rows.append([float(n) for n in numbers])

    if section == "B":
        b_matrix = rows

    if not a_matrix or not b_matrix:
        raise ValueError(f"Could not parse SVAR A/B matrices: {text[:200]!r}")

    return {
        "a_matrix": {"a" + str(i): v for i, v in enumerate(sum(a_matrix, []))},
        "b_matrix": {"b" + str(i): v for i, v in enumerate(sum(b_matrix, []))},
    }


def parse_hayashi_pcse(text: str) -> dict[str, dict[str, float]]:
    """Parse a PCSE coefficient table where SE column is labelled 'PCSE'."""
    import re

    lines = text.splitlines()
    start_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"Variável\s+coef\s+PCSE", line):
            start_idx = i
            break
    if start_idx == -1:
        raise ValueError(f"PCSE table header not found in Hayashi output: {text[:200]!r}")

    result: dict[str, dict[str, float]] = {"coefficients": {}, "standard_errors": {}}
    pattern = re.compile(
        r"^\s*(\S.*?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]\d+)?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]\d+)?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]\d+)?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]\d+)?)"
    )

    for line in lines[start_idx + 1:]:
        stripped = line.strip()
        if not stripped or set(stripped) <= {"─", "═", " "}:
            continue
        match = pattern.match(line)
        if not match:
            continue
        var = match.group(1).strip()
        if var.lower() == "variável":
            continue
        result["coefficients"][var] = float(match.group(2))
        result["standard_errors"][var] = float(match.group(3))

    if not result["coefficients"]:
        raise ValueError(f"Could not parse PCSE rows from Hayashi output: {text[:200]!r}")
    return result


def parse_hayashi_zip(text: str) -> dict[str, dict[str, float]]:
    """Parse a zero-inflated count model (ZIP/ZINB) coefficient table.

    Hayashi prints a 'Count Model' block (with real variable names) and an
    'Inflate Model (Logit)' block (with z0..zN placeholders).  z0 maps to the
    intercept and zN (N>=1) maps to the N-th non-intercept variable from the
    count block.
    """
    import re

    lines = text.splitlines()
    count_names: list[str] = []
    result: dict[str, dict[str, float]] = {"coefficients": {}, "standard_errors": {}}
    section: str | None = None
    pattern = re.compile(
        r"^\s*(\S.*?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
    )

    # First pass: collect count-model variable order.
    for line in lines:
        stripped = line.strip()
        if "Count Model" in stripped:
            section = "count"
            continue
        if "Inflate Model" in stripped:
            break
        if not stripped or stripped.startswith("-") or stripped.startswith("="):
            continue
        lower = stripped.lower()
        if "coef" in lower and "std err" in lower:
            continue
        match = pattern.match(line)
        if not match:
            continue
        var = match.group(1).strip()
        if var.lower() == "variable" or var == "":
            continue
        count_names.append(var)

    # Second pass: store coefficients with mapped names.
    section = None
    for line in lines:
        stripped = line.strip()
        if "Count Model" in stripped:
            section = "count"
            continue
        if "Inflate Model" in stripped:
            section = "inflate"
            continue
        if not stripped or stripped.startswith("-") or stripped.startswith("="):
            continue
        lower = stripped.lower()
        if "coef" in lower and "std err" in lower:
            continue
        match = pattern.match(line)
        if not match:
            continue
        var = match.group(1).strip()
        if var.lower() == "variable" or var == "":
            continue
        coef = float(match.group(2))
        se = float(match.group(3))

        if section == "count":
            key = f"count_{var}"
        elif section == "inflate":
            m = re.match(r"^z(\d+)$", var)
            if m and count_names:
                idx = int(m.group(1))
                var = count_names[idx] if idx < len(count_names) else var
            key = f"inflate_{var}"
        else:
            continue

        result["coefficients"][key] = coef
        result["standard_errors"][key] = se

    if not result["coefficients"]:
        raise ValueError(f"Could not parse ZIP/ZINB coefficient table: {text[:200]!r}")
    return result


def parse_hayashi_mlogit(text: str) -> dict[str, dict[str, float]]:
    """Parse a multinomial-logit coefficient table with per-category sections.

    Hayashi prints one coefficient block per non-base category, e.g.
    "y=1.0 vs base y=4.0".  We flatten the coefficients by prefixing the
    category label so that every {category}:{variable} pair is unique.
    """
    import re

    section_re = re.compile(r"y=([0-9.]+)\s+vs base y=[0-9.]+")
    result: dict[str, dict[str, float]] = {"coefficients": {}, "standard_errors": {}}
    current_cat: str | None = None

    for line in text.splitlines():
        m = section_re.search(line)
        if m:
            current_cat = m.group(1)
            continue
        if current_cat is None:
            continue
        line = line.strip()
        if not line or line.startswith("-") or line.startswith("="):
            continue
        lower = line.lower()
        if "variable" in lower or "coef" in lower or "std err" in lower:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        var = parts[0]
        if var.lower() == "variable":
            continue
        try:
            coef = float(parts[1])
            se = float(parts[2])
        except ValueError:
            continue
        if var == "const":
            var = "Intercept"
        key = f"{current_cat}:{var}"
        result["coefficients"][key] = coef
        result["standard_errors"][key] = se

    if not result["coefficients"]:
        raise ValueError(f"Could not parse multinomial-logit coefficient table: {text[:200]!r}")
    return result


def parse_hayashi_sur(text: str) -> dict[str, dict[str, float]]:
    """Parse a SUR coefficient table with per-equation sections.

    Hayashi prints a block for each equation, e.g. "Equation: value".
    We flatten the coefficients by prefixing the equation name so that
    every {equation}:{variable} pair is unique.
    """
    import re

    section_re = re.compile(r"Equation:\s+(\S+)")
    result: dict[str, dict[str, float]] = {"coefficients": {}, "standard_errors": {}}
    current_eq: str | None = None

    for line in text.splitlines():
        m = section_re.search(line)
        if m:
            current_eq = m.group(1)
            continue
        if current_eq is None:
            continue
        line = line.strip()
        if not line or line.startswith("-") or line.startswith("="):
            continue
        lower = line.lower()
        if "variable" in lower or "coef" in lower or "std err" in lower or "r²" in lower:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        var = parts[0]
        if var.lower() == "variable":
            continue
        try:
            coef = float(parts[1])
            se = float(parts[2])
        except ValueError:
            continue
        if var == "const":
            var = "Intercept"
        key = f"{current_eq}:{var}"
        result["coefficients"][key] = coef
        result["standard_errors"][key] = se

    if not result["coefficients"]:
        raise ValueError(f"Could not parse SUR coefficient table: {text[:200]!r}")
    return result


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


def compare_against_references(
    hayashi: dict[str, Any],
    references: dict[str, dict[str, Any]],
    tolerances: dict[str, float],
) -> tuple[list[str], dict[str, list[str]]]:
    """Compare Hayashi output independently with every reference output."""
    failures: list[str] = []
    failures_by_reference: dict[str, list[str]] = {}
    for reference_name, reference in references.items():
        _, reference_failures = compare_quantities(hayashi, reference, tolerances)
        failures_by_reference[reference_name] = reference_failures
        failures.extend(
            f"{reference_name}: {failure}" for failure in reference_failures
        )
    return failures, failures_by_reference


def run_case(case: dict[str, Any]) -> tuple[str, list[str], dict[str, dict]]:
    """Run a single validation case.

    Returns (status, failures, ref_report) where ref_report maps each declared
    reference name to ``{"status": ..., "detail": ..., "used": bool}``.
    """
    case_id = case["id"]
    case_dir = VALIDATION_DIR / "cases" / case_id
    hayashi_dir = case_dir / "hayashi"
    data_dir = case_dir / "data"

    log(f"\n[case] {case_id}: {case['title']}")

    # Ensure data directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Resolve script paths relative to the validation directory.
    reference_scripts = case.get("reference_scripts", {})
    references = case.get("references", [])
    ref_report: dict[str, dict] = {}
    ref_results: dict[str, subprocess.CompletedProcess[str]] = {}

    # ── Run each declared reference ──────────────────────────────────
    for ref_name in references:
        if ref_name not in reference_scripts:
            ref_report[ref_name] = {"status": "missing", "detail": "no script declared", "used": False}
            continue

        if ref_name == "R":
            if not check_executable("Rscript"):
                ref_report[ref_name] = {"status": "missing", "detail": "Rscript not found", "used": False}
                continue
            r_script = str(VALIDATION_DIR / reference_scripts["R"])
            res = run_command(["Rscript", r_script])
            ref_results["R"] = res
            if res.returncode == 0:
                ref_report[ref_name] = {"status": "passed", "detail": "", "used": False}
            else:
                detail = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "non-zero exit"
                ref_report[ref_name] = {"status": "failed", "detail": detail, "used": False}

        elif ref_name == "Python":
            py_exe = python_executable()
            if not Path(py_exe).exists() and not check_executable(py_exe):
                ref_report[ref_name] = {"status": "missing", "detail": "python not found", "used": False}
                continue
            py_script = str(VALIDATION_DIR / reference_scripts["Python"])
            res = run_command([py_exe, py_script])
            ref_results["Python"] = res
            if res.returncode == 0:
                ref_report[ref_name] = {"status": "passed", "detail": "", "used": False}
            else:
                detail = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "non-zero exit"
                ref_report[ref_name] = {"status": "failed", "detail": detail, "used": False}

        elif ref_name == "Stata":
            if not check_executable("stata"):
                ref_report[ref_name] = {"status": "missing", "detail": "stata not found", "used": False}
                continue
            st_script = str(VALIDATION_DIR / reference_scripts["Stata"])
            res = run_command(["stata", "-b", "do", st_script])
            ref_results["Stata"] = res
            if res.returncode == 0:
                ref_report[ref_name] = {"status": "passed", "detail": "", "used": False}
            else:
                detail = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "non-zero exit"
                ref_report[ref_name] = {"status": "failed", "detail": detail, "used": False}
        else:
            ref_report[ref_name] = {"status": "missing", "detail": f"unknown reference '{ref_name}'", "used": False}

    # ── Print per-reference status ───────────────────────────────────
    log("  References:")
    for name in references:
        info = ref_report.get(name, {"status": "unknown", "detail": ""})
        detail = f" ({info['detail']})" if info.get("detail") else ""
        log(f"    {name}: {info['status']}{detail}")

    # ── Strict policy: every declared reference must pass ────────────
    available_refs = [name for name, info in ref_report.items() if info["status"] == "passed"]
    failed_refs = [name for name, info in ref_report.items() if info["status"] in ("failed", "missing")]

    if failed_refs:
        msgs = []
        for name in failed_refs:
            info = ref_report[name]
            msgs.append(f"{name}: {info['status']} ({info['detail']})")
        return "blocked", msgs, ref_report

    if not available_refs:
        return "blocked", ["No reference implementation could run."], ref_report

    # Run Hayashi script.
    if not check_executable("hay"):
        hay_exe = str(ROOT_DIR / "target" / "release" / "hay")
    else:
        hay_exe = "hay"

    hay_script = str(VALIDATION_DIR / case.get("hayashi_script", f"cases/{case_id}/hayashi/run.hay"))
    hay_res = run_command([hay_exe, hay_script])
    if hay_res.returncode != 0:
        return "blocked", [f"Hayashi script failed:\n{hay_res.stderr}"], ref_report

    # ── Parse every declared reference output ────────────────────────
    reference_outputs: dict[str, dict[str, Any]] = {}
    for reference_name in available_refs:
        stdout = ref_results[reference_name].stdout.strip()
        if not stdout:
            return (
                "blocked",
                [f"{reference_name} reference produced no JSON output"],
                ref_report,
            )
        parsed = parse_reference_json(stdout)
        if parsed is None:
            return (
                "blocked",
                [f"Could not parse {reference_name} reference stdout as JSON"],
                ref_report,
            )
        reference_outputs[reference_name] = normalise_intercept(parsed)

    # Prefer the stdout emitted by Hayashi; fall back to the written file.
    hayashi: dict[str, dict[str, float]] | None = None
    output_format = case.get("output_format", "csv")
    family = case.get("estimator_family", "")
    if hay_res.stdout.strip():
        try:
            if family == "mlogit":
                hayashi = normalise_intercept(parse_hayashi_mlogit(hay_res.stdout))
            elif family == "pca":
                hayashi = parse_hayashi_pca(hay_res.stdout)
            elif family == "sur":
                hayashi = normalise_intercept(parse_hayashi_sur(hay_res.stdout))
            elif family in ("zip", "zinb"):
                hayashi = normalise_intercept(parse_hayashi_zip(hay_res.stdout))
            elif family == "km":
                hayashi = normalise_intercept(parse_hayashi_km(hay_res.stdout))
            elif family == "svar":
                hayashi = normalise_intercept(parse_hayashi_svar(hay_res.stdout))
            elif family == "pcse":
                hayashi = normalise_intercept(parse_hayashi_pcse(hay_res.stdout))
            elif output_format == "margins":
                hayashi = normalise_intercept(parse_hayashi_margins(hay_res.stdout))
            elif output_format == "txt":
                if family == "kalman":
                    hayashi = parse_hayashi_local_level(hay_res.stdout)
                else:
                    hayashi = normalise_intercept(parse_hayashi_txt_table(hay_res.stdout))
            else:
                hayashi = normalise_intercept(parse_hayashi_csv_from_string(hay_res.stdout))
        except Exception as e:
            # Fall back to family-specific parsers for causal estimators.
            try:
                if family == "psm":
                    hayashi = normalise_intercept(parse_hayashi_psm(hay_res.stdout))
                elif family == "rd" or family == "rdd":
                    hayashi = normalise_intercept(parse_hayashi_rd(hay_res.stdout))
                elif family == "synth":
                    hayashi = normalise_intercept(parse_hayashi_synth(hay_res.stdout))
                else:
                    raise
            except Exception:
                return "blocked", [f"Could not parse Hayashi stdout ({output_format}): {e}"], ref_report
    if hayashi is None:
        if output_format == "margins":
            hayashi_txt = hayashi_dir / "output.txt"
            if not hayashi_txt.exists():
                return "blocked", [f"Hayashi output not found: {hayashi_txt}"], ref_report
            hayashi = normalise_intercept(parse_hayashi_margins(hayashi_txt.read_text()))
        elif output_format == "txt":
            hayashi_txt = hayashi_dir / "output.txt"
            if not hayashi_txt.exists():
                return "blocked", [f"Hayashi output not found: {hayashi_txt}"], ref_report
            if family == "kalman":
                hayashi = parse_hayashi_local_level(hayashi_txt.read_text())
            else:
                hayashi = normalise_intercept(parse_hayashi_txt_table(hayashi_txt.read_text()))
        else:
            hayashi_csv = hayashi_dir / "output.csv"
            if not hayashi_csv.exists():
                return "blocked", [f"Hayashi output not found: {hayashi_csv}"], ref_report
            hayashi = normalise_intercept(parse_hayashi_csv(hayashi_csv))

    # Compare declared quantities independently against every reference.
    tolerances = case.get("comparison", {}).get("tolerances", {})
    failures, failures_by_reference = compare_against_references(
        hayashi,
        reference_outputs,
        tolerances,
    )
    for reference_name, reference_failures in failures_by_reference.items():
        ref_report[reference_name]["used"] = True
        if reference_failures:
            ref_report[reference_name]["detail"] = "; ".join(reference_failures)
    status = "fail" if failures else "pass"

    if failures:
        for f in failures:
            log(f"  FAIL: {f}")
    else:
        log(f"  PASS (compared against {', '.join(reference_outputs)})")

    return status, failures, ref_report


def render_matrix_md(cases: list[dict[str, Any]]) -> str:
    lines = [
        "# Hayashi Validation Matrix",
        "",
        "| Family | Dataset | Reference | Status | Blocking Issue | Notes |",
        "|---|---|---:|---|---|---|",
    ]
    for case in cases:
        family = case.get("estimator_family", "")
        dataset = case.get("dataset", {}).get("name", "")
        declared_refs = case.get("references", [])
        ref_info = case.get("result", {}).get("references", {})
        if ref_info:
            ref_parts = []
            for name in declared_refs:
                info = ref_info.get(name, {})
                st = info.get("status", "?")
                used = " *" if info.get("used") else ""
                ref_parts.append(f"{name}:{st}{used}")
            refs = ", ".join(ref_parts)
        else:
            refs = ", ".join(declared_refs)
        status = case.get("status", "not-started")
        issue = case.get("result", {}).get("issues_opened", [])
        issue_str = ", ".join(str(i) for i in issue) if issue else "—"
        notes = case.get("notes", "").replace("\n", " ")
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
        "The Reference column shows per-reference status as `name:status`,",
        "where `*` marks the reference used for comparison. A declared",
        "reference that fails or is missing blocks the case.",
        "",
        "This matrix is generated from `validation/matrix.yml` by `validation/run.py`.",
        "",
        "This matrix covers the core empirical estimators. Some commands are",
        "intentionally excluded for the reasons described in the \"Estimators not",
        "covered by validation\" section of the README.",
        "",
        "Esta matriz abrange os estimadores empíricos centrais. Alguns comandos são",
        "deixados de fora intencionalmente pelos motivos descritos na seção",
        "\"Estimators not covered by validation\" do README.",
        "",
    ])
    return "\n".join(lines) + "\n"


def update_matrix_md(cases: list[dict[str, Any]]) -> None:
    MATRIX_MD.write_text(render_matrix_md(cases))


def _case_matrix_metadata(case: dict[str, Any]) -> tuple[str, str, str, str, str]:
    family = case.get("estimator_family", "")
    dataset = case.get("dataset", {}).get("name", "")
    status = case.get("status", "not-started")
    issue = case.get("result", {}).get("issues_opened", [])
    issue_str = ", ".join(str(i) for i in issue) if issue else "—"
    notes = case.get("notes", "").replace("\n", " ")
    return family, dataset, status, issue_str, notes


def matrix_md_metadata_matches(cases: list[dict[str, Any]], text: str) -> bool:
    """Return True when MATRIX.md reflects stable case metadata.

    The Reference column may include dynamic per-reference run status, so this
    check intentionally compares only the stable metadata columns.
    """
    actual_rows: list[tuple[str, str, str, str, str]] = []
    for line in text.splitlines():
        if (
            not line.startswith("| ")
            or line.startswith("| Family ")
            or line.startswith("|---")
        ):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            return False
        family, dataset, _reference, status, issue, notes = cells
        actual_rows.append((family, dataset, status, issue, notes))

    expected_rows = [_case_matrix_metadata(case) for case in cases]
    return actual_rows == expected_rows


def _validation_relative_path(path: str) -> Path:
    return VALIDATION_DIR / path


def _check_declared_script(
    findings: list[str],
    case_id: str,
    label: str,
    script_path: str | None,
) -> None:
    if not script_path:
        findings.append(f"{case_id}: missing {label} script path")
        return
    if not _validation_relative_path(script_path).exists():
        findings.append(f"{case_id}: declared {label} script not found: {script_path}")


def check_metadata(
    _matrix: dict[str, Any],
    cases: list[dict[str, Any]],
    registry_ids: set[str],
    discovered_ids: set[str],
) -> list[str]:
    """Validate validation metadata without running estimator scripts."""
    findings: list[str] = []

    for case_id in sorted(registry_ids - discovered_ids):
        findings.append(f"{case_id}: matrix.yml registry entry has no case.yml on disk")
    for case_id in sorted(discovered_ids - registry_ids):
        findings.append(f"{case_id}: case.yml exists but matrix.yml has no registry entry")

    for case in sorted(cases, key=lambda c: c["id"]):
        case_id = case["id"]
        case_dir = VALIDATION_DIR / "cases" / case_id
        if not (case_dir / "README.md").exists():
            findings.append(f"{case_id}: missing README.md")

        status = case.get("status", "not-started")
        manifest_status = case.get("_manifest_status", status)
        registry_entry = next(
            (entry for entry in _matrix.get("cases", []) if entry.get("id") == case_id),
            {},
        )
        if manifest_status == "not-started" and registry_entry.get("status") == "pass":
            findings.append(
                f"{case_id}: not-started case cannot have a recorded pass result"
            )
        references = case.get("references", [])
        tolerances = case.get("comparison", {}).get("tolerances", {})

        if status == "pass":
            if not references:
                findings.append(
                    f"{case_id}: status pass requires at least one declared reference"
                )
            if not tolerances:
                findings.append(f"{case_id}: status pass requires comparison tolerances")

        hayashi_script = case.get("hayashi_script", f"cases/{case_id}/hayashi/run.hay")
        _check_declared_script(findings, case_id, "Hayashi", hayashi_script)

        reference_scripts = case.get("reference_scripts", {})
        for ref_name in references:
            _check_declared_script(
                findings,
                case_id,
                f"{ref_name} reference",
                reference_scripts.get(ref_name),
            )

    if not MATRIX_MD.exists():
        findings.append("validation/MATRIX.md is missing")
    elif not matrix_md_metadata_matches(cases, MATRIX_MD.read_text()):
        findings.append("validation/MATRIX.md is stale; regenerate it with validation/run.py")

    return findings


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hayashi empirical validation cases.",
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=[],
        metavar="CASE_ID",
        help="Run only the named validation case. May be repeated.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered validation cases and exit.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not rewrite validation/matrix.yml or validation/MATRIX.md.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Exit with status 0 when validation cases are blocked. By default blocked counts as failure.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check validation metadata consistency without running validation cases.",
    )
    return parser.parse_args(argv)


def load_cases() -> tuple[dict[str, Any], list[dict[str, Any]], set[str], set[str]]:
    if not MATRIX_YML.exists():
        raise FileNotFoundError(f"{MATRIX_YML} not found")

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
        case["_manifest_status"] = case.get("status", "not-started")
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
        cases_by_id[case_id]["dimension"] = entry.get(
            "dimension",
            cases_by_id[case_id].get("dimension", "numerical"),
        )
        cases_by_id[case_id]["status"] = entry.get(
            "status",
            cases_by_id[case_id].get("status", "not-started"),
        )

    cases = list(cases_by_id.values())
    discovered_ids = {case["id"] for case in cases}
    return matrix, cases, registry_ids, discovered_ids


def list_cases(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        print(f"{case['id']}\t{case.get('title', '')}")


def select_cases(cases: list[dict[str, Any]], case_ids: list[str]) -> list[dict[str, Any]]:
    if not case_ids:
        return cases

    cases_by_id = {case["id"]: case for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in cases_by_id]
    if missing:
        known = ", ".join(sorted(cases_by_id))
        missing_str = ", ".join(missing)
        raise ValueError(f"Unknown validation case(s): {missing_str}\nKnown cases: {known}")

    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for case_id in case_ids:
        if case_id in seen:
            continue
        seen.add(case_id)
        selected.append(cases_by_id[case_id])
    return selected


def run_cases(cases: list[dict[str, Any]]) -> str:
    overall_status = "pass"
    for case in cases:
        declared_status = case.get("status", "not-started")
        if declared_status in ("blocked", "not-supported"):
            # Keep the declared status and skip execution; the case files
            # should document why it is blocked/not-supported.
            status = declared_status
            failures = []
            summary = case.get("result", {}).get("summary", "")
            log(f"\n[case] {case['id']}: {case.get('title', '')}")
            log(f"  {declared_status.upper()}: {summary}")
        else:
            status, failures, ref_report = run_case(case)
            if status == "blocked":
                for f in failures:
                    log(f"  BLOCKED: {f}")
                if not failures:
                    log(f"  BLOCKED")
            # Store per-reference report in the case result for audit trail.
            if ref_report:
                case.setdefault("result", {})["references"] = ref_report
        case["status"] = status
        if status == "fail":
            overall_status = "fail"
        elif status == "blocked" and overall_status != "fail":
            overall_status = "blocked"
        summary = "; ".join(failures) if failures else case.get("result", {}).get("summary", "matches reference")
        case.setdefault("result", {})["summary"] = summary
    return overall_status


def write_matrix(matrix: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    # Write updated matrix.yml (id + notes + dimension + status for each discovered case).
    matrix["cases"] = [
        {
            "id": case["id"],
            "notes": case.get("notes", ""),
            "dimension": case.get("dimension", "numerical"),
            "status": case.get("status", "not-started"),
        }
        for case in cases
    ]
    with open(MATRIX_YML, "w") as f:
        yaml.dump(matrix, f, sort_keys=False, allow_unicode=True)

    # Regenerate MATRIX.md.
    update_matrix_md(cases)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    log("Hayashi empirical validation programme")
    log(f"Root: {ROOT_DIR}")

    try:
        matrix, cases, registry_ids, discovered_ids = load_cases()
    except FileNotFoundError as e:
        log(f"ERROR: {e}")
        return 1

    log(f"Discovered {len(cases)} validation case(s)")
    if not cases:
        log("No validation cases found in validation/cases/*/case.yml")
        return 0

    # Warn about registry entries that no longer exist on disk.
    for case_id in registry_ids - discovered_ids:
        log(f"WARNING: registry entry '{case_id}' has no case.yml on disk; skipping")

    if args.list:
        list_cases(cases)
        return 0

    if args.check:
        findings = check_metadata(matrix, cases, registry_ids, discovered_ids)
        if findings:
            log("Validation metadata check failed:")
            for finding in findings:
                log(f"  - {finding}")
            return 1
        log("Validation metadata check passed")
        return 0

    try:
        selected_cases = select_cases(cases, args.case_ids)
    except ValueError as e:
        log(f"ERROR: {e}")
        return 1

    if args.case_ids:
        log(f"Selected {len(selected_cases)} validation case(s)")

    overall_status = run_cases(selected_cases)

    if args.no_write:
        log("Skipping matrix update (--no-write)")
    else:
        write_matrix(matrix, cases)

    log(f"\nOverall status: {overall_status}")
    if overall_status == "fail":
        return 1
    if overall_status == "blocked" and not args.allow_blocked:
        log("ERROR: validation blocked (use --allow-blocked to tolerate)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
