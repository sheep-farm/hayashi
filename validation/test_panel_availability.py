"""Adversarial tests for the wagepan availability contract, using existing tooling."""

import ast
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CASE = ROOT / "validation/cases/panel_fe_twoway_cluster_wagepan"
FORMULA = "lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87"
INPUT = """input df
lwage union nr year
1 2 0 0
-2 3 0 1
-3 3 0 2
4 4 1 0
4 -4 1 1
0 -4 1 2
3 1 2 0
-4 -2 2 1
4 -3 2 2
-3 -3 3 0
2 1 3 1
-3 -1 3 2
end
"""


def runner_module():
    spec = importlib.util.spec_from_file_location("panel_validation_runner", ROOT / "validation/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_result():
    return {
        "covariance": {"union": {"union": 1.0, "d81": 0.0}, "d81": {"union": 0.0, "d81": -0.25}},
        "inference_available": False,
        "inference_reason": "negative diagonal variance for d81; covariance has materially negative eigenvalues",
        "covariance_diagnostics": {
            "finite": True, "eigenvalues": [-0.25, 1.0], "symmetry_error": 0.0,
            "symmetry_bound": 1e-13, "eigenvalue_bound": 1e-13,
        },
    }


class ReferenceAvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ast.parse((CASE / "reference/run.py").read_text(encoding="utf-8"))
        helper = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_checked_covariance_rejection")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "reference guard", "exec"), namespace)
        cls.guard = staticmethod(namespace[helper.name])

    def r_guard(self, result):
        binary = shutil.which("Rscript")
        if binary is None:
            raise RuntimeError("Rscript is required for R reference guard checks")
        code = r"""
args <- commandArgs(trailingOnly = TRUE)
for (node in parse(args[1])) {
  if (is.call(node) && identical(node[[1]], as.name("<-")) &&
      identical(node[[2]], as.name("checked_covariance_rejection"))) eval(node)
}
payload <- paste(readLines(args[2], warn = FALSE, encoding = "UTF-8"), collapse = "\n")
value <- checked_covariance_rejection(jsonlite::fromJSON(payload, simplifyVector = FALSE))
cat(value)
"""
        # File arguments avoid Windows Rscript's inline-expression/JSON quoting path.
        with tempfile.TemporaryDirectory(prefix="panel-r-availability-") as folder:
            script = Path(folder) / "guard.R"
            payload = Path(folder) / "input.json"
            script.write_text(code, encoding="utf-8")
            payload.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
            return subprocess.run([binary, str(script), str(CASE / "reference/run.R"), str(payload)],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)

    def test_expected_material_covariance_rejection_is_numeric_one(self):
        self.assertEqual(self.guard(reference_result()), 1)
        self.assertIs(type(self.guard(reference_result())), int)
        result = self.r_guard(reference_result())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "1")

    def test_admissible_nonfinite_and_wrong_reason_references_fail(self):
        admissible = reference_result()
        admissible["inference_available"] = True
        admissible["inference_reason"] = None
        admissible["covariance"]["d81"]["d81"] = 0.25
        admissible["covariance_diagnostics"]["eigenvalues"] = [0.25, 1.0]
        nonfinite = reference_result()
        nonfinite["covariance"]["d81"]["d81"] = "NaN"
        nonfinite["covariance_diagnostics"]["finite"] = False
        nonfinite["covariance_diagnostics"]["eigenvalues"] = None
        wrong_reason = reference_result()
        wrong_reason["inference_reason"] = "within design is rank deficient"
        invalid_results = [
            (admissible, "Expected finite materially indefinite d81 covariance"),
            (nonfinite, "Availability requires finite raw covariance and its spectrum"),
            (wrong_reason, "Expected finite materially indefinite d81 covariance"),
        ]
        for result, expected_reason in invalid_results:
            with self.subTest(reason=result["inference_reason"]):
                with self.assertRaises(ValueError):
                    self.guard(copy.deepcopy(result))
                completed = self.r_guard(result)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_reason, completed.stderr)
                self.assertNotEqual(completed.stdout, "1")


class NativeAvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = runner_module()
        # Match the existing runner's repo-first debug/release .exe discovery.
        executable = "hay.exe" if sys.platform == "win32" else "hay"
        candidates = [ROOT / "target" / profile / executable for profile in ["debug", "release"]]
        cls.binary = next((str(path) for path in candidates if path.exists()), None)
        if cls.binary is None:
            cls.binary = shutil.which("hay")
        if cls.binary is None:
            raise RuntimeError("Build the native hay binary in target/debug or target/release, or add it to PATH")

    def native(self, script, force_file=False):
        with tempfile.TemporaryDirectory(prefix="panel-availability-") as folder:
            source = Path(folder) / "contract.hay"
            output = Path(folder) / "output.json"
            source.write_text(script, encoding="utf-8")
            if sys.platform == "win32" or force_file:
                source = self.runner._prepare_windows_hayashi_script(source, output)
            completed = subprocess.run([self.binary, str(source)], cwd=ROOT, capture_output=True, text=True, timeout=30)
            if completed.returncode == 0 and output.exists():
                completed.stdout += "\n" + output.read_text(encoding="utf-8")
            return completed

    def small_script(self):
        source = (CASE / "hayashi/run.hay").read_text(encoding="utf-8")
        source = source.replace('load "validation/cases/panel_fe_twoway_cluster_wagepan/data/wagepan.csv" as df', INPUT)
        return source.replace(FORMULA, "lwage ~ union")

    def test_expected_rejection_passes_the_existing_exact_numeric_comparator(self):
        # The existing file transport is exercised on POSIX as well as Windows.
        for force_file in [False, True]:
            with self.subTest(force_file=force_file):
                completed = self.native(self.small_script(), force_file=force_file)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = self.runner.parse_reference_json(completed.stdout)
                self.assertEqual(result["availability"]["covariance_rejected"], 1)
                status, failures = self.runner.compare_quantities(result, {"availability": {"covariance_rejected": 1}},
                                                                {"availability.covariance_rejected": 0})
                self.assertEqual((status, failures), ("pass", []))

    def test_unexpected_success_and_wrong_errors_cannot_pass(self):
        source = self.small_script()
        successful = source.replace("let clustered = fe(lwage ~ union, df, cluster=nr, cluster2=year)",
                                    "let clustered = fe(lwage ~ union, df)")
        wrong_error = source.replace("let clustered = fe(lwage ~ union, df, cluster=nr, cluster2=year)",
                                     "let clustered = fe(missing ~ union, df, cluster=nr, cluster2=year)")
        control_error = source.replace("let control = fe(lwage ~ union, df)", "let control = fe(missing ~ union, df)")
        for script in [successful, wrong_error, control_error]:
            self.assertNotEqual(script, source)
            completed = self.native(script)
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertNotIn('"covariance_rejected": 1', completed.stdout)
        status, _ = self.runner.compare_quantities({"availability": {"covariance_rejected": 0}},
                                                  {"availability": {"covariance_rejected": 1}},
                                                  {"availability.covariance_rejected": 0})
        self.assertEqual(status, "fail")


class NativeDiscoveryTests(unittest.TestCase):
    def test_windows_repo_binary_discovery_and_path_fallback(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory(prefix="panel-native-discovery-") as folder:
            root = Path(folder)
            release = root / "target/release/hay.exe"
            debug = root / "target/debug/hay.exe"
            release.parent.mkdir(parents=True)
            debug.parent.mkdir(parents=True)
            release.touch()
            module = sys.modules[__name__]
            with patch.object(module, "ROOT", root), patch.object(module, "runner_module", return_value=runner), \
                 patch.object(sys, "platform", "win32"), patch.object(shutil, "which", return_value=None):
                class Probe:
                    pass

                NativeAvailabilityTests.setUpClass.__func__(Probe)
                self.assertEqual(Probe.binary, str(release))
                debug.touch()
                NativeAvailabilityTests.setUpClass.__func__(Probe)
                self.assertEqual(Probe.binary, str(debug))
                debug.unlink()
                release.unlink()
                with patch.object(shutil, "which", return_value="path/hay.exe"):
                    NativeAvailabilityTests.setUpClass.__func__(Probe)
                    self.assertEqual(Probe.binary, "path/hay.exe")
                with self.assertRaises(RuntimeError):
                    NativeAvailabilityTests.setUpClass.__func__(Probe)


if __name__ == "__main__":
    unittest.main()
