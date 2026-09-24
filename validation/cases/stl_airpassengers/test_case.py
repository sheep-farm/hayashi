"""Harness tests, not an assertion that the current estimator must fail."""

import copy
import csv
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ref = module("stl_reference", CASE / "reference/run.py")
runner = module("validation_runner", ROOT / "validation/run.py")


class CaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["Rscript", str(CASE / "data/gen.R")], cwd=ROOT, check=True,
                       capture_output=True, text=True)
        cls.text = ref.DATA.read_text(encoding="ascii")
        cls.observed = np.log(ref.load_counts())

    def test_input_contract_rejects_mutations(self):
        lines = self.text.splitlines()
        mutations = [lines[:-1], lines + [lines[-1]],
                     [lines[0], lines[2], lines[1], *lines[3:]],
                     [lines[0], lines[1], lines[1], *lines[3:]]]
        for value in ("113", "nan", "inf", "0", "-1", "112.5", "112.0", "", "112,9"):
            mutations.append([lines[0], f"1,{value}", *lines[2:]])
        mutations.append(["wrong,passengers", *lines[1:]])
        for rows in mutations:
            with self.subTest(row=rows[:3]):
                with patch.object(Path, "open", return_value=io.StringIO("\n".join(rows))):
                    with self.assertRaises(ValueError):
                        ref.load_counts()

    def test_vector_shape_and_finite(self):
        for bad in (np.ones(143), np.ones(145), np.ones((144, 1)),
                    [np.nan] + [1] * 143, [np.inf] + [1] * 143,
                    [-np.inf] + [1] * 143):
            with self.subTest(shape=np.shape(bad)), self.assertRaises(ValueError):
                ref.vector(bad)

    def test_reference_shape_identity_and_reconstruction(self):
        fit, components = ref.fit_stl(self.observed)
        self.assertLessEqual(ref.check_components(fit.observed, components, self.observed), 1e-12)
        output = ref.reference_output()["coefficients"]
        self.assertEqual(len(output), 578)
        self.assertEqual(output["nobs"], 144)
        self.assertEqual(output["reconstruction_max"], 0)
        json.dumps(output, allow_nan=False)
        for key in ("observed", *ref.COMPONENTS):
            self.assertTrue(all(f"{key}_{i}" in output for i in range(1, 145)))

    def test_component_gates_reject_invalid_output(self):
        fit, components = ref.fit_stl(self.observed)
        bad = {key: value.copy() for key, value in components.items()}
        bad["trend"][0] += 1e-6
        with self.assertRaises(ValueError):
            ref.check_components(fit.observed, bad, self.observed)
        with self.assertRaises(ValueError):
            ref.check_components(fit.observed + 1e-6, components, self.observed)
        with self.assertRaises(ValueError):
            ref.check_components(fit.observed, {"trend": fit.trend}, self.observed)
        bad["trend"][0] = np.nan
        with self.assertRaises(ValueError):
            ref.check_components(fit.observed, bad, self.observed)

    def test_manifest_and_comparison_cover_every_component(self):
        case = yaml.safe_load((CASE / "case.yml").read_text())
        self.assertEqual(case["references"], ["Python"])
        self.assertEqual(set(case["reference_scripts"]), {"Python"})
        self.assertEqual(case["result"]["issues_opened"], [160])
        tolerances = case["comparison"]["tolerances"]
        self.assertEqual(tolerances, {"coefficients": 1e-8, "coefficients.nobs": 0,
                                      "coefficients.reconstruction_max": 1e-12})
        expected = ref.reference_output()
        self.assertEqual(runner.compare_quantities(expected, expected, tolerances), ("pass", []))
        for key in expected["coefficients"]:
            changed = copy.deepcopy(expected)
            changed["coefficients"][key] += 1e-7
            self.assertEqual(runner.compare_quantities(changed, expected, tolerances)[0], "fail", key)
            del changed["coefficients"][key]
            self.assertEqual(runner.compare_quantities(changed, expected, tolerances)[0], "fail", key)
        changed = copy.deepcopy(expected)
        changed["coefficients"]["trend_1"] += 0.01
        changed["coefficients"]["remainder_1"] -= 0.01
        vectors = {key: [changed["coefficients"][f"{key}_{i}"] for i in range(1, 145)]
                   for key in ref.COMPONENTS}
        self.assertLessEqual(ref.check_components(self.observed, vectors, self.observed), 1e-12)
        self.assertEqual(runner.compare_quantities(changed, expected, tolerances)[0], "fail")
        changed["coefficients"]["trend_1"] = float("nan")
        self.assertEqual(runner.compare_quantities(changed, expected, tolerances)[0], "fail")

    @unittest.skipUnless(shutil.which("hay"), "hay must be on PATH for the actual CSV test")
    def test_actual_hayashi_csv_transport(self):
        for file_output in ([True] if sys.platform == "win32" else [False, True]):
            with self.subTest(file_output=file_output), tempfile.TemporaryDirectory() as tmp:
                source = CASE / "hayashi/run.hay"
                output = Path(tmp) / "output.csv"
                script = runner._prepare_windows_hayashi_script(source, output) if file_output else source
                try:
                    result = subprocess.run(["hay", str(script)], cwd=ROOT,
                                            capture_output=True, text=True, check=True)
                    text = output.read_text(encoding="utf-8") if file_output else result.stdout
                    self.check_csv_transport(text)
                finally:
                    if script != source:
                        script.unlink(missing_ok=True)

    def check_csv_transport(self, text):
        parsed = runner.parse_hayashi_csv_from_string(text)
        header = text.index("variable,coef,std_err")
        rows = list(csv.reader(io.StringIO(text[header:])))[1:579]
        labels = [f"{key}_{i}" for i in range(1, 145)
                  for key in ("observed", *ref.COMPONENTS)] + ["nobs", "reconstruction_max"]
        self.assertEqual([row[0] for row in rows], labels)
        self.assertEqual(set(parsed["coefficients"]), set(labels))
        self.assertEqual(set(parsed["standard_errors"]), set(labels))
        self.assertTrue(all(value == 0 for value in parsed["standard_errors"].values()))
        values = parsed["coefficients"]
        vectors = {key: ref.vector([values[f"{key}_{i}"] for i in range(1, 145)])
                   for key in ("observed", *ref.COMPONENTS)}
        np.testing.assert_allclose(vectors["observed"], self.observed,
                                   atol=1e-12, rtol=0, equal_nan=False)
        error = ref.check_components(vectors["observed"],
                                    {key: vectors[key] for key in ref.COMPONENTS},
                                    vectors["observed"])
        self.assertEqual(values["nobs"], 144)
        self.assertLessEqual(values["reconstruction_max"], 1e-12)
        self.assertAlmostEqual(error, values["reconstruction_max"], delta=1e-15)


if __name__ == "__main__":
    unittest.main()
