## Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions
## are met:
##  * Redistributions of source code must retain the above copyright
##    notice, this list of conditions and the following disclaimer.
##  * Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
##  * Neither the name of NVIDIA CORPORATION nor the names of its
##    contributors may be used to endorse or promote products derived from
##    this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
## EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
## PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
## NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
## SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUN_TESTS_PATH = Path(__file__).parents[1] / "run_tests.py"
SPEC = importlib.util.spec_from_file_location("run_tests", RUN_TESTS_PATH)
run_tests = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_tests)


class RunTestsCommandTests(unittest.TestCase):
    def test_build_command_uses_absolute_executable_path(self):
        executable = Path("build") / "clock.exe"

        command = run_tests.build_command(executable, ["--quick"], ["--verbose"])

        self.assertEqual(command, [str(executable.resolve()), "--quick", "--verbose"])

    def test_run_single_test_instance_uses_executable_path_with_working_directory(self):
        executable = Path("build") / "clock.exe"
        with tempfile.TemporaryDirectory() as output_dir:
            output_file = Path(output_dir) / "clock-output.txt"
            with mock.patch.object(run_tests.subprocess, "run") as run:
                run.return_value.returncode = 0
                result = run_tests.run_single_test_instance(
                    executable, [], str(output_file), [], ""
                )

        self.assertEqual(result["status"], "Passed")
        command = run.call_args.args[0]
        self.assertEqual(command[0], str(executable.resolve()))
        self.assertEqual(run.call_args.kwargs["cwd"], executable.resolve().parent)


if __name__ == "__main__":
    unittest.main()
