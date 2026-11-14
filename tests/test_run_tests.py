"""Regression tests for sample run selection; no CUDA installation is required."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import run_tests


class RunSelectionTests(unittest.TestCase):
    def run_runner(self, config=None, names=("sample",)):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "run_tests.py", "--dir", str(root),
                "--output", str(root / "results"),
            ]
            if config is not None:
                config_path = root / "config.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                argv.extend(["--config", str(config_path)])

            output = io.StringIO()
            with patch.object(sys, "argv", argv), \
                    patch.object(run_tests, "get_gpu_count", return_value=1), \
                    patch.object(run_tests, "find_executables",
                                 return_value=[root / name for name in names]), \
                    patch.object(run_tests.subprocess, "run",
                                 return_value=subprocess.CompletedProcess([], 0)) as launch, \
                    redirect_stdout(output):
                return_code = run_tests.main()

            self.assertEqual(return_code, 0)
            commands = [call.args[0] for call in launch.call_args_list]
            self.assertIn(f"Ran {len(commands)} test runs", output.getvalue())
            return commands

    def test_all_runs_skipped(self):
        for count in (1, 2):
            with self.subTest(count=count):
                config = {"sample": {"runs": [
                    {"args": [f"--disabled={i}"], "skip": True}
                    for i in range(count)
                ]}}
                self.assertEqual(self.run_runner(config), [])

    def test_only_enabled_runs_execute(self):
        config = {"sample": {"runs": [
            {"args": ["--disabled"], "skip": True},
            {"args": ["--first"], "skip": False},
            {"args": ["--second"]},
        ]}}
        self.assertEqual(self.run_runner(config), [
            ["./sample", "--first"], ["./sample", "--second"],
        ])

    def test_executable_skip_overrides_enabled_runs(self):
        config = {"sample": {"skip": True, "runs": [{"args": ["--enabled"]}]}}
        self.assertEqual(self.run_runner(config), [])

    def test_missing_config_runs_default(self):
        self.assertEqual(self.run_runner(), [["./sample"]])

    def test_config_without_run_arguments_runs_default(self):
        for config in ({}, {"sample": {}}, {"sample": {"min_gpus": 1}}):
            with self.subTest(config=config):
                self.assertEqual(self.run_runner(config), [["./sample"]])

    def test_empty_runs_list_does_not_execute(self):
        self.assertEqual(self.run_runner({"sample": {"runs": []}}), [])

    def test_enabled_run_without_arguments_executes(self):
        config = {"sample": {"runs": [{"skip": True}, {}]}}
        self.assertEqual(self.run_runner(config), [["./sample"]])

    def test_top_level_arguments_take_precedence_over_runs(self):
        for args in ([], ["--single"]):
            with self.subTest(args=args):
                config = {"sample": {"args": args, "runs": [{"skip": True}]}}
                self.assertEqual(self.run_runner(config), [["./sample"] + args])

    def test_invalid_run_arguments_do_not_create_a_default_run(self):
        config = {"sample": {"runs": [{"args": "--invalid"}]}}
        self.assertEqual(self.run_runner(config), [])

    def test_skipped_sample_does_not_skip_other_executables(self):
        config = {"sample": {"runs": [{"skip": True}]}}
        self.assertEqual(self.run_runner(config, names=("sample", "other")), [
            ["./other"],
        ])


if __name__ == "__main__":
    unittest.main()
