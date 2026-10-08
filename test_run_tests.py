#!/usr/bin/env python3
"""Unit tests for the path-construction logic in run_tests.py.

Regression tests for NVIDIA/cuda-samples issue #453: the harness used to
invoke test executables with a POSIX-style relative path ("./clock.exe"),
which fails on Windows.  The harness now builds an absolute, OS-native
path instead.  These tests run on any platform (Linux included) and do
not require a GPU, a CUDA toolkit, or the samples to be built.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Import the module under test from this repository's root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tests  # noqa: E402


class FakeResult:
    def __init__(self, returncode):
        self.returncode = returncode


class TestBuildTestCommand(unittest.TestCase):
    """Tests for run_tests.build_test_command()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="test_run_tests_")
        self.addCleanup(self.tmp.cleanup)
        self.exe = Path(self.tmp.name) / "Samples" / "0_Introduction" / "clock.exe"
        self.exe.parent.mkdir(parents=True)
        self.exe.write_bytes(b"fake-binary")
        os.chmod(self.exe, 0o755)

    def test_command_is_absolute(self):
        """The executable must be launched via an absolute path."""
        cmd = run_tests.build_test_command(self.exe)
        self.assertTrue(Path(cmd[0]).is_absolute(), f"not absolute: {cmd[0]}")

    def test_relative_input_becomes_absolute(self):
        """find_executables() can yield relative paths (e.g. with --dir .);
        those must still become absolute."""
        rel = Path(os.path.relpath(self.exe))
        cmd = run_tests.build_test_command(rel)
        self.assertTrue(Path(cmd[0]).is_absolute(), f"not absolute: {cmd[0]}")
        self.assertEqual(Path(cmd[0]).resolve(), self.exe.resolve())

    def test_command_resolves_to_same_file(self):
        cmd = run_tests.build_test_command(self.exe)
        self.assertEqual(Path(cmd[0]).resolve(), self.exe.resolve())
        self.assertEqual(Path(cmd[0]).name, "clock.exe")

    def test_no_dot_slash_prefix(self):
        """Regression check for issue #453: never emit "./name"-style paths."""
        cmd = run_tests.build_test_command(self.exe)
        self.assertFalse(cmd[0].startswith("./"))
        self.assertFalse(cmd[0].startswith(".\\"))

    def test_accepts_string_path(self):
        cmd = run_tests.build_test_command(str(self.exe))
        self.assertTrue(Path(cmd[0]).is_absolute())


class TestRunSingleTestInstance(unittest.TestCase):
    """Tests for run_tests.run_single_test_instance() path handling."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="test_run_tests_")
        self.addCleanup(self.tmp.cleanup)
        self.exe = Path(self.tmp.name) / "clock.exe"
        self.exe.write_bytes(b"fake-binary")
        os.chmod(self.exe, 0o755)
        self.out = Path(self.tmp.name) / "out.txt"
        self.captured = {}

    def _fake_run(self, returncode=0):
        def _run(cmd, **kwargs):
            self.captured["cmd"] = cmd
            self.captured["cwd"] = kwargs.get("cwd")
            return FakeResult(returncode)

        return _run

    def _run_instance(self, *args, **kwargs):
        with patch.object(run_tests.subprocess, "run", side_effect=self._fake_run(**kwargs)):
            return run_tests.run_single_test_instance(self.exe, ["--a"], str(self.out), ["--global"], "")

    def test_subprocess_receives_absolute_executable(self):
        self._run_instance(returncode=0)
        exe_arg = self.captured["cmd"][0]
        self.assertTrue(Path(exe_arg).is_absolute(), f"not absolute: {exe_arg}")
        self.assertFalse(exe_arg.startswith("./"))
        self.assertEqual(Path(exe_arg).resolve(), self.exe.resolve())

    def test_cwd_is_executable_directory(self):
        self._run_instance(returncode=0)
        cwd = self.captured["cwd"]
        self.assertTrue(os.path.isabs(cwd), f"cwd not absolute: {cwd}")
        self.assertEqual(Path(cwd).resolve(), self.exe.parent.resolve())

    def test_args_are_forwarded(self):
        self._run_instance(returncode=0)
        cmd = self.captured["cmd"]
        self.assertEqual(cmd[1:], ["--a", "--global"])

    def test_status_mapping(self):
        cases = [
            (0, "Passed"),
            (run_tests.EXIT_WAIVED, "Waived"),
            (1, "Failed"),
        ]
        for returncode, expected in cases:
            self.captured.clear()
            with patch.object(run_tests.subprocess, "run",
                              side_effect=self._fake_run(returncode=returncode)):
                result = run_tests.run_single_test_instance(self.exe, [], str(self.out), None, "")
            self.assertEqual(result["status"], expected,
                             f"returncode {returncode} should map to {expected}")

    def test_timeout(self):
        with patch.object(run_tests.subprocess, "run",
                          side_effect=subprocess.TimeoutExpired("cmd", 300)):
            result = run_tests.run_single_test_instance(self.exe, [], str(self.out), None, "")
        self.assertEqual(result["status"], "Timeout")

    def test_unexpected_error(self):
        with patch.object(run_tests.subprocess, "run", side_effect=OSError("boom")):
            result = run_tests.run_single_test_instance(self.exe, [], str(self.out), None, "")
        self.assertTrue(result["status"].startswith("Error:"), result["status"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
