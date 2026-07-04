import importlib.util
import unittest
from pathlib import Path


def load_runner_module():
    run_py = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("validation_run", run_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {run_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load_runner_module()


class ValidationRunnerCliTests(unittest.TestCase):
    def test_blocked_status_exits_nonzero_by_default(self):
        self.assertEqual(run.exit_code_for_status("blocked"), 1)

    def test_allow_blocked_makes_blocked_status_successful(self):
        self.assertEqual(run.exit_code_for_status("blocked", allow_blocked=True), 0)

    def test_pass_and_fail_exit_codes_are_stable(self):
        self.assertEqual(run.exit_code_for_status("pass"), 0)
        self.assertEqual(run.exit_code_for_status("fail"), 1)

    def test_allow_blocked_flag_is_parsed(self):
        args = run.parse_args(["--allow-blocked"])
        self.assertTrue(args.allow_blocked)


if __name__ == "__main__":
    unittest.main()
