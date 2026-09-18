#!/usr/bin/env python3
"""Offline regression tests for Arellano-Bond reference JSON output."""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


RUN_PY = Path(__file__).resolve().parent / "cases/ab_grunfeld/reference/run.py"


class ReferenceOutputTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("ab_reference", RUN_PY)
        self.module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(self.module)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ref_dir = Path(tmp.name) / "reference"
        self.enterContext(patch.object(self.module, "REF_DIR", self.ref_dir))
        data = pd.DataFrame({
            "firm": [1, 1, 1],
            "year": [2000, 2001, 2002],
            "inv": [1.0, 2.0, 4.0],
            "value": [2.0, 4.0, 8.0],
            "capital": [3.0, 6.0, 12.0],
        })
        self.ensure_data = self.enterContext(
            patch.object(self.module, "ensure_data", return_value=data)
        )
        self.fit = self.enterContext(patch.object(
            self.module,
            "fit_arellano_bond",
            return_value=(
                np.array([0.5, -1.25, 2.0]),
                np.array([0.125, 0.25, 0.5]),
            ),
        ))
        self.expected = {
            "coefficients": {"LD.y": 0.5, "\u0394value": -1.25, "\u0394capital": 2.0},
            "standard_errors": {"LD.y": 0.125, "\u0394value": 0.25, "\u0394capital": 0.5},
        }

    def test_file_uses_utf8_with_cp1252_default(self):
        # Explicit encodings override this simulated Windows locale default.
        with patch.object(
            self.module, "open", partial(open, encoding="cp1252", errors="strict"),
            create=True,
        ), redirect_stdout(io.StringIO()):
            result = self.module.compute_reference()

        self.assertEqual(result, self.expected)
        payload = (self.ref_dir / "expected.json").read_bytes()
        self.assertIn("\u0394value".encode("utf-8"), payload)
        self.assertIn("\u0394capital".encode("utf-8"), payload)
        self.assertEqual(json.loads(payload.decode("utf-8")), self.expected)
        self.ensure_data.assert_called_once_with()
        self.fit.assert_called_once()

    def test_stdout_is_ascii_safe_on_strict_cp1252(self):
        # Keep file output UTF-8 independently, so it cannot mask stdout failure.
        with io.BytesIO() as buffer, io.TextIOWrapper(
            buffer, encoding="cp1252", errors="strict"
        ) as stdout, patch.object(
            self.module, "open", partial(open, encoding="utf-8"), create=True,
        ), redirect_stdout(stdout):
            result = self.module.compute_reference()
            stdout.flush()
            payload = buffer.getvalue()

        self.assertEqual(result, self.expected)
        self.assertTrue(payload.isascii())
        self.assertEqual(json.loads(payload.decode("cp1252")), self.expected)
        self.assertEqual(
            json.loads((self.ref_dir / "expected.json").read_text(encoding="utf-8")),
            self.expected,
        )
        self.ensure_data.assert_called_once_with()
        self.fit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
