#!/usr/bin/env python3
"""Focused tests for validation runner metadata checks."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

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
            "status": "pass",
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


if __name__ == "__main__":
    unittest.main()
