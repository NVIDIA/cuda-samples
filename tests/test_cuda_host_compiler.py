"""Check custom-command host compiler arguments without a CUDA toolkit."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COMMON_CMAKE = Path(__file__).resolve().parents[1] / "cmake" / "CudaSampleCommon.cmake"


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class CudaHostCompilerTests(unittest.TestCase):
    def check_arguments(self, host_compiler, nested):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            build = Path(temporary) / "build"
            output = build / "arguments.json"
            capture = source / "capture.py"
            capture.write_text(
                "import json, pathlib, sys\n"
                "pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n"
            )
            setup = 'cmake_minimum_required(VERSION 3.20)\nproject(host_args NONE)\n'
            if host_compiler is not None:
                setup += f'set(CMAKE_CUDA_HOST_COMPILER "{host_compiler}")\n'
            include = f'include("{COMMON_CMAKE.as_posix()}")\n'
            command = (
                'add_custom_target(capture ALL\n'
                f'    COMMAND "{Path(sys.executable).as_posix()}" "{capture.as_posix()}"\n'
                f'        "{output.as_posix()}" ${{CUDA_HOST_COMPILER_ARGS}}\n'
                '    VERBATIM)\n'
            )
            if nested:
                # Root setup is inherited by leaves even with the global include guard.
                leaf = source / "leaf"
                leaf.mkdir()
                (leaf / "CMakeLists.txt").write_text(include + command)
                (source / "CMakeLists.txt").write_text(setup + include + 'add_subdirectory(leaf)\n')
            else:
                (source / "CMakeLists.txt").write_text(setup + include + command)
            subprocess.run(
                ["cmake", "-S", str(source), "-B", str(build)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                ["cmake", "--build", str(build)],
                check=True, capture_output=True, text=True,
            )
            expected = [f"-ccbin={host_compiler}"] if host_compiler else []
            self.assertEqual(json.loads(output.read_text()), expected)

    def test_standalone_arguments(self):
        for host_compiler in (None, "", "/toolchains/g++", "/toolchain with spaces/g++"):
            with self.subTest(host_compiler=host_compiler):
                self.check_arguments(host_compiler, nested=False)

    def test_root_arguments_are_inherited_by_leaf(self):
        for host_compiler in (None, "", "/toolchains/g++", "/toolchain with spaces/g++"):
            with self.subTest(host_compiler=host_compiler):
                self.check_arguments(host_compiler, nested=True)


if __name__ == "__main__":
    unittest.main()
