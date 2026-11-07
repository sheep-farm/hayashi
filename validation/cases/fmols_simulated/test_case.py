"""FMOLS fixture, inference-selection and actual consumer/reference contracts."""

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from scipy.signal import lfilter

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[2]
FIXTURES = ("iid", "endogeneity", "serial_correlation", "holdout")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module("fmols_validation_runner", ROOT / "validation/run.py")


def r_reference_available():
    if not shutil.which("Rscript"):
        return False
    result = subprocess.run(["Rscript", "--vanilla", "-e",
                             'quit(status=if(requireNamespace("cointReg",quietly=TRUE)) 0 else 1)'],
                            capture_output=True, timeout=30)
    return result.returncode == 0


def expected_tolerances():
    expected = {}
    for name in FIXTURES:
        prefix = "" if name == "iid" else name + "_"
        for field in ("Intercept", "x"):
            expected[f"coefficients.{prefix}{field}"] = 1e-10
            expected[f"standard_errors.{prefix}{field}"] = 1e-10
        for field in ("n_obs", "bandwidth"):
            expected[f"coefficients.{prefix}{field}"] = 0
    return expected


class FixtureTests(unittest.TestCase):
    def datasets(self):
        generator = module("fmols_fixture_generator", CASE / "data/gen.py")
        self.assertTrue(callable(getattr(generator, "datasets", None)),
                        "The generator must expose the four declared fixture designs")
        return generator.datasets()

    def test_declared_designs_preserve_the_original_iid_values(self):
        frames = self.datasets()
        self.assertEqual(tuple(frames), FIXTURES)
        for name, frame in frames.items():
            self.assertEqual(list(frame.columns), ["y", "x"], name)
            self.assertEqual(frame.shape, (300, 2), name)
            self.assertTrue(np.isfinite(frame.to_numpy()).all(), name)
        # Replay the original generator in this runtime, without a cross-platform byte promise.
        state = np.random.get_state()
        try:
            np.random.seed(42)
            x = np.cumsum(np.random.normal(0.0, 0.1, 300))
            e = np.random.normal(0.0, 0.5, 300)
            np.testing.assert_array_equal(frames["iid"], np.column_stack([0.5*x+e, x]))
        finally:
            np.random.set_state(state)
        repeated = self.datasets()
        for name in FIXTURES:
            np.testing.assert_array_equal(frames[name], repeated[name])

    def test_new_designs_obey_their_seeded_innovation_equations(self):
        frames = self.datasets()
        for name, seed, n, rho, gamma, burn in (
                ("endogeneity", 314159, 300, 0.0, 0.75, 0),
                ("serial_correlation", 271828, 300, 0.7, 0.0, 0),
                ("holdout", 2026103101, 600, 0.5, 0.5, 300)):
            rng = np.random.Generator(np.random.PCG64(seed))
            dx, e = rng.normal(size=n), rng.normal(size=n)
            innovations = gamma*dx + e
            if name == "serial_correlation":
                innovations[0] = e[0]/np.sqrt(1-rho**2)
            # Existing SciPy filtering independently checks the explicit generator loops.
            u = lfilter([1.0], [1.0, -rho], innovations)[burn:]
            frame = frames[name]
            np.testing.assert_allclose(frame.x, dx.cumsum()[burn:], rtol=0, atol=5e-14)
            np.testing.assert_allclose(frame.y-0.25-0.5*frame.x, u, rtol=0, atol=5e-14)

    def test_every_fixture_coefficient_and_se_is_selected(self):
        case = yaml.safe_load((CASE / "case.yml").read_text(encoding="utf-8"))
        tolerances = case["comparison"]["tolerances"]
        self.assertEqual(tolerances, expected_tolerances())
        for language in ("R", "Python"):
            self.assertEqual(set(case["reference_evidence"][language]), set(tolerances))
        reference = {"coefficients": {}, "standard_errors": {}}
        for path in expected_tolerances():
            group, field = path.split(".")
            reference[group][field] = 1.0
        self.assertEqual(runner.compare_quantities(reference, reference, tolerances), ("pass", []))
        for path in expected_tolerances():
            group, field = path.split(".")
            for value in (None, float("nan"), float("inf"), 0.0, -1.0, 2.0):
                with self.subTest(path=path, value=value):
                    changed = copy.deepcopy(reference)
                    changed[group][field] = value
                    self.assertEqual(runner.compare_quantities(changed, reference, tolerances)[0], "fail")
            changed = copy.deepcopy(reference)
            del changed[group][field]
            self.assertEqual(runner.compare_quantities(changed, reference, tolerances)[0], "fail")

    def invalid_inputs(self):
        frame = pd.concat([value.assign(fixture=name, row=np.arange(300))[["fixture", "row", "y", "x"]]
                           for name, value in self.datasets().items()], ignore_index=True)
        wrong_name = frame.copy()
        wrong_name.loc[0, "fixture"] = "unknown"
        nonfinite = frame.copy()
        nonfinite.loc[0, "y"] = np.nan
        repeated_row = frame.copy()
        repeated_row.loc[0] = repeated_row.loc[1]
        reordered = frame.iloc[np.r_[1, 0, np.arange(2, len(frame))]]
        return (frame.iloc[1:], pd.concat([frame, frame.iloc[[0]]]), wrong_name,
                nonfinite, frame.assign(extra=1), repeated_row, reordered)

    def test_python_reference_rejects_incomplete_or_invalid_fixture_data(self):
        reference = module("fmols_python_reference", CASE / "reference/run.py")
        self.assertTrue(callable(getattr(reference, "reference_results", None)),
                        "References must validate the complete fixture input before fitting")
        for frame in self.invalid_inputs():
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                reference.reference_results(frame)

    def test_iid_counts_distinguish_residual_rows_from_the_fm_sample(self):
        frames = pd.concat([value.assign(fixture=name, row=np.arange(300))[["fixture", "row", "y", "x"]]
                            for name, value in self.datasets().items()], ignore_index=True)
        reference = module("fmols_python_counts", CASE / "reference/run.py")
        result = reference.reference_results(frames, fixtures=("iid",))
        self.assertEqual(result["coefficients"]["n_obs"], 299)
        self.assertEqual(result["diagnostics"]["fixtures"]["iid"]["returned_residual_rows"], 300)

    def test_python_reference_rejects_invalid_native_inference(self):
        frames = pd.concat([value.assign(fixture=name, row=np.arange(300))[["fixture", "row", "y", "x"]]
                            for name, value in self.datasets().items()], ignore_index=True)
        reference = module("fmols_python_invalid_native", CASE / "reference/run.py")
        # Fault injection at the external fit boundary exercises otherwise inadmissible returns.
        for field, bad in (("se", 0.0), ("se", -1.0), ("se", np.nan), ("se", np.inf),
                           ("coef", np.nan), ("coef", np.inf), ("band", 6), ("rows", 299)):
            fit = SimpleNamespace(resid=pd.Series(np.zeros(300)), bandwidth=5,
                                  params=pd.Series([0.0, 0.5], index=["const", "x"]),
                                  std_errors=pd.Series([0.1, 0.1], index=["const", "x"]))
            if field == "se":
                fit.std_errors.iloc[0] = bad
            elif field == "coef":
                fit.params.iloc[0] = bad
            elif field == "band":
                fit.bandwidth = bad
            else:
                fit.resid = fit.resid.iloc[:bad]
            with self.subTest(field=field, bad=bad), patch.object(reference, "FullyModifiedOLS") as native:
                native.return_value.fit.return_value = fit
                with self.assertRaises(ValueError):
                    reference.reference_results(frames, fixtures=("iid",))

    @unittest.skipUnless(r_reference_available(), "The restored cointReg package is required for native R guards")
    def test_r_reference_rejects_invalid_native_inference(self):
        subprocess.run([sys.executable, str(CASE / "data/gen.py")], cwd=ROOT, check=True)
        code = '''args <- commandArgs(trailingOnly=TRUE)
source(args[1]); d <- read.csv(args[2]); frame <- d[d$fixture=="iid",]
native <- cointReg::cointRegFM
valid <- native(as.matrix(frame["x"]), frame$y, deter=matrix(1,300,1),
                kernel="ba", bandwidth=6L, demeaning=FALSE)
for (field in c("sd.theta","theta")) {
  values <- if(field=="sd.theta") c(0,-1,NaN,Inf) else c(NaN,Inf)
  for (value in values) {
    bad <- valid; bad[[field]][1] <- value
    assignInNamespace("cointRegFM", function(...) bad, ns="cointReg")
    failure <- tryCatch({reference_results(d,fixtures="iid"); "ACCEPTED"},
                        error=function(e) conditionMessage(e))
    stopifnot(failure!="ACCEPTED", grepl(field,failure,fixed=TRUE))
  }
}
'''
        with tempfile.TemporaryDirectory(prefix="fmols-r-native-guard-") as folder:
            script = Path(folder) / "guard.R"
            script.write_text(code, encoding="utf-8")
            result = subprocess.run(["Rscript", "--vanilla", str(script),
                                     str(CASE / "reference/run.R"), str(CASE / "data/data.csv")],
                                    cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    @unittest.skipUnless(shutil.which("Rscript"), "Rscript is required for the R input guard")
    def test_r_reference_rejects_incomplete_or_invalid_fixture_data(self):
        with tempfile.TemporaryDirectory(prefix="fmols-r-guard-") as folder:
            script = Path(folder) / "guard.R"
            script.write_text('args <- commandArgs(trailingOnly=TRUE)\n'
                              'source(args[1])\nreference_results(read.csv(args[2]))\n', encoding="utf-8")
            for frame in self.invalid_inputs():
                payload = Path(folder) / "input.csv"
                frame.to_csv(payload, index=False)
                result = subprocess.run(["Rscript", "--vanilla", str(script),
                                         str(CASE / "reference/run.R"), str(payload)],
                                        cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Invalid FMOLS fixture data", result.stderr)


class ConsumerTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("hay") and shutil.which("Rscript"),
                         "hay and Rscript are required for actual consumer/reference checks")
    def test_actual_hayashi_matches_both_references_on_all_four_fixtures(self):
        subprocess.run([sys.executable, str(CASE / "data/gen.py")], cwd=ROOT, check=True)
        refs = []
        for command in (["Rscript", str(CASE / "reference/run.R")],
                        [sys.executable, str(CASE / "reference/run.py")]):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
            refs.append(json.loads(result.stdout))
        case = yaml.safe_load((CASE / "case.yml").read_text(encoding="utf-8"))
        self.assertEqual(runner.compare_quantities(refs[0], refs[1], expected_tolerances()), ("pass", []))
        status, failures, reports = runner.run_case(case, quiet=True)
        self.assertEqual((status, failures), ("pass", []))
        self.assertEqual({name: report["status"] for name, report in reports.items()},
                         {"R": "passed", "Python": "passed"})


if __name__ == "__main__":
    unittest.main()
