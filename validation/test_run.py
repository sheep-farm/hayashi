#!/usr/bin/env python3
"""Focused tests for validation runner metadata checks."""

import importlib.util
from copy import deepcopy
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
        (case_dir / "hayashi" / "run.hay").write_text("# hayashi script\n", encoding="utf-8")
        if include_python_script:
            (case_dir / "reference" / "run.py").write_text("print('{}')\n", encoding="utf-8")
        if include_readme:
            (case_dir / "README.md").write_text("# OLS example\n", encoding="utf-8")
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
        with open(case_dir / "case.yml", "w", encoding="utf-8") as f:
            yaml.safe_dump(case, f, sort_keys=False)

    def write_matrix(self, entries: list[dict]) -> None:
        with open(self.module.MATRIX_YML, "w", encoding="utf-8") as f:
            yaml.safe_dump({"cases": entries}, f, sort_keys=False)

    def evidence_case(self):
        self.write_case(references=["R", "Python"])
        case_dir = self.validation_dir / "cases" / "ols_example"
        (case_dir / "reference" / "run.R").write_text("# R reference\n", encoding="utf-8")
        self.write_matrix([{"id": "ols_example", "status": "pass"}])
        loaded = self.module.load_cases()
        case = loaded[1][0]
        case["reference_scripts"]["R"] = "cases/ols_example/reference/run.R"
        case["reference_evidence"] = {
            ref: {"coefficients": {"class": "exact", "rationale": "Same contract."}}
            for ref in case["references"]
        }
        return loaded

    def test_evidence_accepts_all_classes_and_absence(self):
        loaded = self.evidence_case()
        case = loaded[1][0]
        for evidence_class in ("exact", "convention-matched", "behavioural-proxy", None):
            with self.subTest(evidence_class=evidence_class):
                if evidence_class is None:
                    del case["reference_evidence"]
                else:
                    for entries in case["reference_evidence"].values():
                        entries["coefficients"]["class"] = evidence_class
                self.module.MATRIX_MD.write_text(self.module.render_matrix_md([case]), encoding="utf-8")
                self.assertEqual(self.module.check_metadata(*loaded), [])

    def test_evidence_rejects_malformed_metadata_without_traceback(self):
        loaded = self.evidence_case()
        case = loaded[1][0]
        valid = deepcopy(case["reference_evidence"])
        invalid = []
        for value in (None, [], "exact", 42, True):
            invalid.append((value, "must be a mapping"))
            changed = deepcopy(valid)
            changed["Python"] = value
            invalid.append((changed, "must be a mapping"))
            changed = deepcopy(valid)
            changed["Python"]["coefficients"] = value
            invalid.append((changed, "must be a mapping"))
        for value in (None, [], {}, 1, True, "EXACT", "proxy", ""):
            changed = deepcopy(valid)
            changed["Python"]["coefficients"]["class"] = value
            invalid.append((changed, ".class must be"))
        for value in (None, [], {}, 1, True, "", " \n\t"):
            changed = deepcopy(valid)
            changed["Python"]["coefficients"]["rationale"] = value
            invalid.append((changed, ".rationale must be"))
        for field in ("class", "rationale"):
            changed = deepcopy(valid)
            del changed["Python"]["coefficients"][field]
            invalid.append((changed, "must contain only class and rationale"))
        changed = deepcopy(valid)
        changed["Python"]["coefficients"]["override"] = "exact"
        invalid.append((changed, "must contain only class and rationale"))
        for evidence, message in invalid:
            with self.subTest(evidence=evidence):
                case["reference_evidence"] = evidence
                self.module.MATRIX_MD.write_text(self.module.render_matrix_md([case]), encoding="utf-8")
                findings = self.module.check_metadata(*loaded)
                self.assertTrue(any(message in finding for finding in findings), findings)
                with patch.object(self.module, "load_cases", return_value=loaded), patch.object(
                    self.module, "run_cases"
                ) as run_cases, patch.object(self.module, "log"):
                    self.assertEqual(self.module.main(["--check"]), 1)
                run_cases.assert_not_called()

    def test_evidence_requires_exact_reference_and_tolerance_key_coverage(self):
        loaded = self.evidence_case()
        case = loaded[1][0]
        valid = deepcopy(case["reference_evidence"])
        invalid = [{}, {"Python": valid["Python"]}, {**valid, "Stata": valid["R"]},
                   {**valid, 1: valid["R"]}]
        for keys in ([], ["coefficients.x"], ["coefficient"],
                     ["coefficients", "standard_errors"], ["coefficients", 1]):
            changed = deepcopy(valid)
            changed["Python"] = {key: valid["Python"]["coefficients"] for key in keys}
            invalid.append(changed)
        for evidence in invalid:
            with self.subTest(evidence=evidence):
                case["reference_evidence"] = evidence
                self.module.MATRIX_MD.write_text(self.module.render_matrix_md([case]), encoding="utf-8")
                findings = self.module.check_metadata(*loaded)
                self.assertTrue(any("must cover exactly" in finding for finding in findings))

        # Literal dotted keys, not comparison.quantities or prefix expansion, govern coverage.
        case["comparison"]["tolerances"] = {"coefficients.x": 1e-6}
        case["reference_evidence"] = {
            ref: {"coefficients.x": entries["coefficients"]}
            for ref, entries in valid.items()
        }
        self.assertEqual(self.module.check_reference_evidence(case), [])

    def test_evidence_notes_are_stable_and_freshness_detects_class_changes(self):
        loaded = self.evidence_case()
        case = loaded[1][0]
        case["notes"] = "Existing\nnotes."
        case["comparison"]["tolerances"]["standard_errors"] = 0.5
        for entries in case["reference_evidence"].values():
            entries["standard_errors"] = {
                "class": "behavioural-proxy", "rationale": "Proxy | not inference.\nMore."
            }
        expected = (
            "Existing notes. Evidence: Python (behavioural-proxy: standard_errors; "
            "exact: coefficients); R (behavioural-proxy: standard_errors; exact: coefficients)."
        )
        rendered = self.module.render_matrix_md([case])
        self.assertIn(expected, rendered)
        self.assertIn("README.md#reference-evidence-classes", rendered)
        self.assertIn("Unannotated cases are unclassified, not exact.", rendered)
        rows = [line for line in rendered.splitlines() if line.startswith("| ")]
        self.assertTrue(all(len(row.strip("|").split("|")) == 6 for row in rows))
        reordered = deepcopy(case)
        reordered["reference_evidence"] = {
            ref: dict(reversed(list(entries.items())))
            for ref, entries in reversed(list(case["reference_evidence"].items()))
        }
        self.assertEqual(rendered, self.module.render_matrix_md([reordered]))
        self.assertEqual(rendered, self.module.render_matrix_md([case]))
        with_execution_details = rendered.replace("| R, Python |", "| R:passed *, Python:passed * |")
        self.assertTrue(self.module.matrix_md_metadata_matches([case], with_execution_details))
        case["reference_evidence"]["Python"]["coefficients"]["class"] = "convention-matched"
        self.assertFalse(self.module.matrix_md_metadata_matches([case], rendered))
        self.module.MATRIX_MD.write_text(rendered, encoding="utf-8")
        self.assertIn("validation/MATRIX.md is stale; regenerate it with validation/run.py",
                      self.module.check_metadata(*loaded))
        del case["reference_evidence"]
        self.assertNotIn("Evidence:", self.module.render_matrix_md([case]))
        self.assertFalse(self.module.matrix_md_metadata_matches([case], rendered))

    def test_evidence_does_not_change_comparison_or_reference_failure_results(self):
        case = self.evidence_case()[1][0]
        evidence = case.pop("reference_evidence")
        scenarios = {
            "match": "pass", "mismatch": "fail", "unavailable": "partial",
            "failed": "partial", "all_unavailable": "blocked", "malformed": "blocked",
            "empty": "blocked",
        }
        for scenario, expected_status in scenarios.items():
            def fake_run_command(cmd, cwd=None, quiet=False):
                if cmd[0] == "Rscript":
                    stdout = '{"coefficients": {"x": 1.0}}'
                elif cmd[0] == "python":
                    stdout = '{"coefficients": {"x": 2.0}}' if scenario == "mismatch" else '{"coefficients": {"x": 1.0}}'
                    if scenario == "malformed":
                        stdout = "not JSON"
                    elif scenario == "empty":
                        stdout = ""
                    elif scenario == "failed":
                        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="failed")
                else:
                    stdout = "Variable,Coef,Std_Err\nx,1.0,0.1\n"
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

            def executable_available(name):
                if name == "Rscript" and scenario in ("unavailable", "all_unavailable"):
                    return False
                return not (name == "python" and scenario == "all_unavailable")

            with patch.object(self.module, "python_executable", return_value="python"), patch.object(
                self.module, "check_executable", side_effect=executable_available
            ), patch.object(self.module, "run_command", side_effect=fake_run_command):
                baseline = self.module.run_case(case, quiet=True)
                self.assertEqual(baseline[0], expected_status)
                for evidence_class in ("exact", "convention-matched", "behavioural-proxy"):
                    with self.subTest(scenario=scenario, evidence_class=evidence_class):
                        annotated = deepcopy(case)
                        annotated["reference_evidence"] = deepcopy(evidence)
                        for entries in annotated["reference_evidence"].values():
                            entries["coefficients"]["class"] = evidence_class
                        self.assertEqual(self.module.run_case(annotated, quiet=True), baseline)

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
        self.module.MATRIX_MD.write_text(matrix_md, encoding="utf-8")

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
        self.module.MATRIX_MD.write_text(self.module.render_matrix_md(cases), encoding="utf-8")

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
        self.module.MATRIX_MD.write_text(self.module.render_matrix_md(cases), encoding="utf-8")

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
        self.module.MATRIX_MD.write_text("# stale\n", encoding="utf-8")
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
        self.module.MATRIX_MD.write_text(self.module.render_matrix_md(cases), encoding="utf-8")

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
        self.module.MATRIX_MD.write_text(self.module.render_matrix_md(cases), encoding="utf-8")

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
        self.module.MATRIX_MD.write_text(self.module.render_matrix_md(cases), encoding="utf-8")

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

    def test_parse_hayashi_pca_output_is_sign_invariant(self):
        text = """
 Principal Component Analysis
 Component      Var Expl.       % Cum. Eigenvalue
 PC1               0.4167       0.4167     1.6669
 PC2               0.3636       0.7803     1.4545
 Loadings
 Variable                PC1      PC2
 educ                -0.1133  -0.8893
 exper                0.7911   0.3763
 """

        parsed = self.module.parse_hayashi_pca(text)

        self.assertAlmostEqual(parsed["explained_variance"]["PC1"], 1.6669)
        self.assertAlmostEqual(parsed["explained_variance_ratio"]["PC2"], 0.3636)
        self.assertAlmostEqual(parsed["absolute_loadings"]["educ:PC1"], 0.1133)
        self.assertAlmostEqual(parsed["absolute_loadings"]["educ:PC2"], 0.8893)

    def test_parse_hayashi_pca_rejects_missing_sections(self):
        with self.assertRaises(ValueError):
            self.module.parse_hayashi_pca("Principal Component Analysis\n")

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
        (case_dir / "reference" / "run.R").write_text("# R reference\n", encoding="utf-8")
        case = yaml.safe_load((case_dir / "case.yml").read_text(encoding="utf-8"))
        case["id"] = case_id
        case["reference_scripts"]["R"] = f"cases/{case_id}/reference/run.R"

        def fake_run_command(cmd, cwd=None, quiet=False):
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

    def test_run_cases_skips_not_started_case(self):
        case = {
            "id": "planned_example",
            "title": "Planned example",
            "status": "not-started",
            "_manifest_status": "not-started",
            "result": {"summary": "Case has not been implemented."},
        }

        with patch.object(self.module, "run_case") as run_case, patch.object(self.module, "log"):
            status = self.module.run_cases([case])

        self.assertEqual(status, "not-started")
        self.assertEqual(case["status"], "not-started")
        run_case.assert_not_called()


class MainExitStatusTests(unittest.TestCase):
    def setUp(self):
        self.module = load_runner_module()
        self.matrix = {"cases": []}
        self.cases = [
            {"id": "example_1", "title": "Example 1"},
            {"id": "example_2", "title": "Example 2"},
        ]
        for case in self.cases:
            case["references"] = ["Python"]
            case["comparison"] = {"tolerances": {"coefficients": 1e-6}}
            case["reference_evidence"] = {
                "Python": {"coefficients": {
                    "class": "behavioural-proxy", "rationale": "Related diagnostic."
                }}
            }

    def run_main_with_status(
        self,
        status: str,
        extra_args: list[str] | None = None,
        case_statuses: list[str] | None = None,
    ) -> tuple[int, list[str]]:
        argv = ["--no-write"] + list(extra_args or [])
        case_statuses = case_statuses or [status]

        def fake_run_cases(selected_cases, only_blocked=False):
            for case, case_status in zip(selected_cases, case_statuses):
                case["status"] = case_status
            return status

        with patch.object(
            self.module,
            "load_cases",
            return_value=(
                self.matrix,
                self.cases,
                {"example_1", "example_2"},
                {"example_1", "example_2"},
            ),
        ), patch.object(
            self.module,
            "select_cases",
            return_value=self.cases,
        ), patch.object(
            self.module,
            "run_cases",
            side_effect=fake_run_cases,
        ), patch.object(
            self.module,
            "write_matrix",
        ) as write_matrix, patch.object(
            self.module,
            "log",
        ) as log:
            exit_code = self.module.main(argv)

        write_matrix.assert_not_called()
        return exit_code, [call.args[0] for call in log.call_args_list]

    def test_main_accepts_pass(self):
        exit_code, _ = self.run_main_with_status("pass")

        self.assertEqual(exit_code, 0)

    def test_main_rejects_fail(self):
        exit_code, _ = self.run_main_with_status("fail", ["--allow-blocked"])

        self.assertEqual(exit_code, 1)

    def test_main_rejects_blocked_by_default(self):
        exit_code, messages = self.run_main_with_status("blocked")

        self.assertEqual(exit_code, 1)
        self.assertIn("ERROR: validation blocked (use --allow-blocked to tolerate)", messages)

    def test_main_allows_blocked_with_flag(self):
        exit_code, _ = self.run_main_with_status("blocked", ["--allow-blocked"])

        self.assertEqual(exit_code, 0)

    def test_main_rejects_partial_by_default(self):
        exit_code, messages = self.run_main_with_status("partial")

        self.assertEqual(exit_code, 1)
        self.assertIn("ERROR: validation partial (use --allow-partial to tolerate)", messages)

    def test_main_allows_partial_with_flag(self):
        exit_code, _ = self.run_main_with_status("partial", ["--allow-partial"])

        self.assertEqual(exit_code, 0)

    def test_main_rejects_partial_when_blocked_is_allowed(self):
        exit_code, messages = self.run_main_with_status(
            "blocked",
            ["--allow-blocked"],
            case_statuses=["blocked", "partial"],
        )

        self.assertEqual(exit_code, 1)
        self.assertIn("ERROR: validation partial (use --allow-partial to tolerate)", messages)

    def test_main_rejects_not_started(self):
        exit_code, messages = self.run_main_with_status("not-started")

        self.assertEqual(exit_code, 1)
        self.assertIn("ERROR: validation not started", messages)

    def test_main_rejects_blocked_when_partial_is_allowed(self):
        exit_code, messages = self.run_main_with_status(
            "blocked",
            ["--allow-partial"],
            case_statuses=["blocked", "partial"],
        )

        self.assertEqual(exit_code, 1)
        self.assertIn("ERROR: validation blocked (use --allow-blocked to tolerate)", messages)


if __name__ == "__main__":
    unittest.main()
