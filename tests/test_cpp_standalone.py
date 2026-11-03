"""Exercise aggregator CMake scope without CUDA compiler discovery."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class CppStandaloneTests(unittest.TestCase):
    def test_common_setup_is_available_to_all_categories(self):
        for standalone in (True, False):
            for debug in (False, True):
                with self.subTest(standalone=standalone, debug=debug):
                    self.check_configuration(standalone, debug)

    def check_configuration(self, standalone, debug):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            shutil.copytree(REPOSITORY / "cmake", source / "cmake")
            (source / "cpp").mkdir()
            for relative in ("CMakeLists.txt", "cpp/CMakeLists.txt"):
                content = (REPOSITORY / relative).read_text()
                self.assertIn("LANGUAGES C CXX CUDA", content)
                # Keep the real aggregator logic; only omit compiler detection.
                content = content.replace("LANGUAGES C CXX CUDA", "LANGUAGES NONE")
                (source / relative).write_text(content)
            cpp_source = (source / "cpp/CMakeLists.txt").read_text()
            categories = re.findall(r"add_subdirectory\(([^)]+)\)", cpp_source)
            self.assertGreaterEqual(len(categories), 2)
            common = (source / "cmake/CudaSampleCommon.cmake").as_posix()
            expected_flag = "-G" if debug else "-lineinfo"
            for category in categories:
                directory = source / "cpp" / category
                directory.mkdir(parents=True)
                (directory / "CMakeLists.txt").write_text(
                    f'include("{common}")\n'
                    'include(InstallSamples)\n'
                    'if(NOT COMMAND setup_samples_install)\n'
                    '    message(FATAL_ERROR "Missing sample installation function")\n'
                    'endif()\n'
                    'string(STRIP "${CMAKE_CUDA_FLAGS}" actual_flags)\n'
                    f'if(NOT actual_flags STREQUAL "-Wno-deprecated-gpu-targets {expected_flag}")\n'
                    '    message(FATAL_ERROR "Missing or duplicated common flags: ${actual_flags}")\n'
                    'endif()\n'
                )
            configured_source = source / "cpp" if standalone else source
            result = subprocess.run(
                ["cmake", "-S", str(configured_source), "-B", str(Path(temporary) / "build"),
                 f"-DENABLE_CUDA_DEBUG={'ON' if debug else 'OFF'}", "-DBUILD_TEGRA=ON"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
