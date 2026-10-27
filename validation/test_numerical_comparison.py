"""Regression checks for invalid numerical evidence in the validation runner."""

import importlib.util
import unittest
from pathlib import Path


def load_runner():
    spec = importlib.util.spec_from_file_location("comparison_runner", Path(__file__).with_name("run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NumericalComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def compare(self, actual, reference, tolerances=None):
        return self.runner.compare_quantities(
            {"coefficients": actual}, {"coefficients": reference},
            {"coefficients": 1e-8} if tolerances is None else tolerances,
        )

    def test_nonfinite_values_never_establish_agreement(self):
        for value in (float("nan"), float("inf"), -float("inf"), "nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.assertEqual(self.compare({"x": value}, {"x": value})[0], "fail")

    def test_absent_null_and_empty_evidence_fail(self):
        for actual, reference in ((12345, {}), ([], []), ({}, {"x": None}),
                                  ({"x": None}, {"x": None}), ({}, {"x": 1}),
                                  ({"x": []}, {"x": []}), ({"x": {}}, {"x": {}})):
            with self.subTest(actual=actual, reference=reference):
                self.assertEqual(self.compare(actual, reference)[0], "fail")

    def test_invalid_runtime_tolerances_fail_without_exception(self):
        for tolerance in (-1, float("nan"), float("inf"), True, None, "1e-8", [], {}):
            with self.subTest(tolerance=tolerance):
                self.assertEqual(self.compare({"x": 1}, {"x": 1}, {"coefficients": tolerance})[0], "fail")
        for contract in ({}, [], None, {1: 0}):
            with self.subTest(contract=contract):
                self.assertEqual(self.runner.compare_quantities({"x": 1}, {"x": 1}, contract)[0], "fail")

    def test_boolean_is_not_a_numerical_result(self):
        for actual, reference in ((True, 1), (1, True), (True, True), ("nan", float("nan"))):
            with self.subTest(actual=actual, reference=reference):
                self.assertEqual(self.compare({"x": actual}, {"x": reference})[0], "fail")

    def test_integer_precision_cannot_create_a_zero_tolerance_pass(self):
        exact = 2**53
        self.assertEqual(self.compare({"x": exact}, {"x": exact + 1}, {"coefficients": 0})[0], "fail")
        self.assertEqual(self.compare({"x": exact + 1}, {"x": float(exact)}, {"coefficients": 0})[0], "fail")
        self.assertEqual(self.compare({"x": exact + 1}, {"x": exact + 1}, {"coefficients": 0})[0], "pass")
        self.assertEqual(self.compare({"x": exact}, {"x": exact + 1}, {"coefficients": 1})[0], "pass")
        self.assertEqual(self.compare({"x": str(exact + 1)}, {"x": exact}, {"coefficients": 0})[0], "fail")
        self.assertEqual(self.compare({"x": str(exact)}, {"x": str(exact + 1)}, {"coefficients": 0})[0], "fail")
        self.assertEqual(self.compare({"x": "9.007199254740993e15"}, {"x": exact}, {"coefficients": 0})[0], "fail")

    def test_finite_values_and_selected_subsets_still_pass(self):
        self.assertEqual(self.compare({"x": [1, 2.5], "unselected": None}, {"x": [1, 2.5]})[0], "pass")
        self.assertEqual(self.compare({"x": "1.25"}, {"x": 1.25}, {"coefficients": 0})[0], "pass")
        self.assertEqual(self.compare({"name": "x"}, {"name": "x"})[0], "pass")
        self.assertEqual(self.compare({"x": 1.1}, {"x": 1}, {"coefficients": 0})[0], "fail")
        self.assertEqual(self.runner.compare_quantities(
            {"coefficients": {"x": 1, "unselected": 100}},
            {"coefficients": {"x": 1}}, {"coefficients.x": 0},
        )[0], "pass")

    def test_invalid_reference_cannot_hide_behind_passing_reference(self):
        failures, by_reference = self.runner.compare_against_references(
            {"coefficients": {"x": 1}},
            {"valid": {"coefficients": {"x": 1}}, "invalid": {"coefficients": {}}},
            {"coefficients": 0},
        )
        self.assertTrue(failures)
        self.assertEqual(by_reference["valid"], [])
        self.assertTrue(by_reference["invalid"])


if __name__ == "__main__":
    unittest.main()
