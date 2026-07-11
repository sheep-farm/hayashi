#!/usr/bin/env python3
"""Focused tests for validation runner metadata checks."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


RUN_PY = Path(__file__).resolve().parent / "run.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("validation_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MetadataCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.validation_dir = self.root / "validation"
        self.validation_dir.mkdir()
        self.module = load_runner_module()
        self.module.ROOT_DIR = self.root
        self.module.VALIDATION_DIR = self.validation_dir
        self.module.MATRIX_YML = self.validation_dir / "matrix.yml"
        self.module.MATRIX_MD = self.validation_dir / "MATRIX.md"

    def tearDown(self):
        self.tmp.cleanup()

    def write_case(
        self,
        case_id: str = "ols_example",
        *,
        status: str = "active",
        references: list[str] | None = None,
        tolerances: dict[str, float] | None = None,
        include_readme: bool = True,
        include_python_script: bool = True,
    ) -> None:
        references = ["Python"] if references is None else references
        tolerances = {"coefficients": 1e-6} if tolerances is None else tolerances
        case_dir = self.validation_dir / "cases" / case_id
        (case_dir / "hayashi").mkdir(parents=True)
        (case_dir / "reference").mkdir()
        (case_dir / "hayashi" / "run.hay").write_text("# hayashi script\n")
        if include_python_script:
            (case_dir / "reference" / "run.py").write_text("print('{}')\n")
        if include_readme:
            (case_dir / "README.md").write_text("# OLS example\n")
        case = {
            "title": "OLS example",
            "estimator_family": "ols",
            "status": status,
            "dataset": {
                "name": "example",
                "source": "simulated",
                "licence": "public-domain",
            },
            "references": references,
            "comparison": {
                "quantities": list(tolerances),
                "tolerances": tolerances,
            },
            "hayashi_script": f"cases/{case_id}/hayashi/run.hay",
            "reference_scripts": {
                "Python": f"cases/{case_id}/reference/run.py",
            },
            "result": {"summary": "matches reference"},
        }
        with open(case_dir / "case.yml", "w") as f:
            yaml.safe_dump(case, f, sort_keys=False)

    def write_matrix(self, entries: list[dict]) -> None:
        with open(self.module.MATRIX_YML, "w") as f:
            yaml.safe_dump({"cases": entries}, f, sort_keys=False)

    def test_metadata_check_accepts_consistent_case(self):
        self.write_case()
        self.write_matrix([
            {
                "id": "ols_example",
                "notes": "Example case.",
                "dimension": "numerical",
                "status": "pass",
            }
        ])
        matrix, cases, registry_ids, discovered_ids = self.module.load_cases()
        matrix_md = self.module.render_matrix_md(cases).replace(
            "| ols | example | Python | pass | — | Example case. |",
            "| ols | example | Python:passed * | pass | — | Example case. |",
        )
        self.module.MATRIX_MD.write_text(matrix_md)

        findings = self.module.check_metadata(matrix, cases, registry_ids, discovered_ids)

        self.assertEqual(findings, [])

    def test_metadata_check_rejects_not_started_case_with_pass_result(self):
        self.write_case(status="not-started")
        self.write_matrix([
            {
                "id": "ols_example",
                "notes": "Example case.",
                "dimension": "numerical",
                "status": "pass",
            }
        ])
        matrix, cases, registry_ids, discovered_ids = self.module.load_cases()
        self.module.MATRIX_MD.write_text(self.module.render_matrix_md(cases))

        findings = self.module.check_metadata(matrix, cases, registry_ids, discovered_ids)

        self.assertIn(
            "ols_example: not-started case cannot have a recorded pass result",
            findings,
        )

    def test_metadata_check_rejects_pass_case_without_reference(self):
        self.write_case(references=[])
        self.write_matrix([
            {
                "id": "ols_example",
                "notes": "Example case.",
                "dimension": "numerical",
                "status": "pass",
            }
        ])
        matrix, cases, registry_ids, discovered_ids = self.module.load_cases()
        self.module.MATRIX_MD.write_text(self.module.render_matrix_md(cases))

        findings = self.module.check_metadata(matrix, cases, registry_ids, discovered_ids)

        self.assertIn("ols_example: status pass requires at least one declared reference", findings)

    def test_metadata_check_rejects_stale_matrix_md(self):
        self.write_case()
        self.write_matrix([
            {
                "id": "ols_example",
                "notes": "Example case.",
                "dimension": "numerical",
                "status": "pass",
            }
        ])
        self.module.MATRIX_MD.write_text("# stale\n")
        matrix, cases, registry_ids, discovered_ids = self.module.load_cases()

        findings = self.module.check_metadata(matrix, cases, registry_ids, discovered_ids)

        self.assertIn("validation/MATRIX.md is stale; regenerate it with validation/run.py", findings)

    def test_metadata_check_rejects_missing_readme(self):
        self.write_case(include_readme=False)
        self.write_matrix([
            {
                "id": "ols_example",
                "notes": "Example case.",
                "dimension": "numerical",
                "status": "pass",
            }
        ])
        matrix, cases, registry_ids, discovered_ids = self.module.load_cases()
        self.module.MATRIX_MD.write_text(self.module.render_matrix_md(cases))

        findings = self.module.check_metadata(matrix, cases, registry_ids, discovered_ids)

        self.assertIn("ols_example: missing README.md", findings)

    def test_metadata_check_rejects_missing_reference_script(self):
        self.write_case(include_python_script=False)
        self.write_matrix([
            {
                "id": "ols_example",
                "notes": "Example case.",
                "dimension": "numerical",
                "status": "pass",
            }
        ])
        matrix, cases, registry_ids, discovered_ids = self.module.load_cases()
        self.module.MATRIX_MD.write_text(self.module.render_matrix_md(cases))

        findings = self.module.check_metadata(matrix, cases, registry_ids, discovered_ids)

        self.assertIn(
            "ols_example: declared Python reference script not found: "
            "cases/ols_example/reference/run.py",
            findings,
        )

    def test_metadata_check_rejects_directory_missing_from_registry(self):
        self.write_case()
        self.write_matrix([])
        matrix, cases, registry_ids, discovered_ids = self.module.load_cases()
        self.module.MATRIX_MD.write_text(self.module.render_matrix_md(cases))

        findings = self.module.check_metadata(matrix, cases, registry_ids, discovered_ids)

        self.assertIn("ols_example: case.yml exists but matrix.yml has no registry entry", findings)

    def test_parse_hayashi_margins_table(self):
        text = """
==============================================
 Average Marginal Effects — LOGIT
==============================================
Variable                dy/dx   Std.Err.        z    P>|z|
----------------------------------------------
nwifeinc            -0.003811   0.001538   -2.478   0.0132 **
educ                 0.039496   0.008468    4.664   0.0000 ***
----------------------------------------------
n = 753
==============================================
"""

        parsed = self.module.parse_hayashi_margins(text)

        self.assertAlmostEqual(parsed["marginal_effects"]["nwifeinc"], -0.003811)
        self.assertAlmostEqual(parsed["standard_errors"]["educ"], 0.008468)

    def test_compare_against_references_reports_each_reference(self):
        hayashi = {"coefficients": {"x": 1.0}}
        references = {
            "R": {"coefficients": {"x": 1.0}},
            "Python": {"coefficients": {"x": 1.5}},
        }

        failures, failures_by_reference = self.module.compare_against_references(
            hayashi,
            references,
            {"coefficients": 1e-6},
        )

        self.assertEqual(failures_by_reference["R"], [])
        self.assertEqual(len(failures_by_reference["Python"]), 1)
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("Python: coefficients.x:"))

    def test_run_case_compares_hayashi_with_all_references(self):
        case_id = "ols_example"
        case_dir = self.validation_dir / "cases" / case_id
        self.write_case(case_id=case_id, references=["R", "Python"])
        (case_dir / "reference" / "run.R").write_text("# R reference\n")
        case = yaml.safe_load((case_dir / "case.yml").read_text())
        case["id"] = case_id
        case["reference_scripts"]["R"] = f"cases/{case_id}/reference/run.R"

        def fake_run_command(cmd, cwd=None):
            if cmd[0] == "Rscript":
                stdout = '{"coefficients": {"x": 1.0}}\n'
            elif cmd[0] in {"python", "python3"}:
                stdout = '{"coefficients": {"x": 2.0}}\n'
            else:
                stdout = "Variable,Coef,Std_Err\nx,1.0,0.1\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        with patch.object(self.module, "check_executable", return_value=True), patch.object(
            self.module, "run_command", side_effect=fake_run_command
        ):
            status, failures, ref_report = self.module.run_case(case)

        self.assertEqual(status, "fail")
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("Python: coefficients.x:"))
        self.assertTrue(ref_report["R"]["used"])
        self.assertTrue(ref_report["Python"]["used"])


if __name__ == "__main__":
    unittest.main()
