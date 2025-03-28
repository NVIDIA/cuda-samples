## Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
##    contributors may be used to endorse or promote products derived
##    from this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
## EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
## PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
## OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##
## For additional information on the license terms, see the CUDA EULA at
## https://docs.nvidia.com/cuda/eula/index.html

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

def normalize_exe_name(name):
    """Normalize executable name across platforms by removing .exe if present"""
    return Path(name).stem

def load_args_config(config_file):
    """Load arguments configuration from JSON file"""
    if not config_file or not os.path.exists(config_file):
        return {}

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Validate the config format
        if not isinstance(config, dict):
            print("Warning: Config file must contain a dictionary/object")
            return {}

        return config
    except json.JSONDecodeError:
        print("Warning: Failed to parse config file as JSON")
        return {}
    except Exception as e:
        print(f"Warning: Error reading config file: {str(e)}")
        return {}

def find_executables(root_dir):
    """Find all executable files recursively"""
    executables = []

    for path in Path(root_dir).rglob('*'):
        # Skip directories
        if not path.is_file():
            continue

        # Check if file is executable
        if os.access(path, os.X_OK):
            # Skip if it's a library file
            if path.suffix.lower() in ('.dll', '.so', '.dylib'):
                continue
            executables.append(path)

    return executables

def run_test(executable, output_dir, args_config, global_args=None):
    """Run a single test and capture output"""
    exe_path = str(executable)
    exe_name = executable.name
    base_name = normalize_exe_name(exe_name)

    # Check if this executable should be skipped
    if base_name in args_config and args_config[base_name].get("skip", False):
        print(f"Skipping {exe_name} (marked as skip in config)")
        return 0

    # Get argument sets for this executable
    arg_sets = []
    if base_name in args_config:
        config = args_config[base_name]
        if "args" in config:
            # Single argument set (backwards compatibility)
            if isinstance(config["args"], list):
                arg_sets.append(config["args"])
            else:
                print(f"Warning: Arguments for {base_name} must be a list")
        elif "runs" in config:
            # Multiple argument sets
            for run in config["runs"]:
                if isinstance(run.get("args", []), list):
                    arg_sets.append(run.get("args", []))
                else:
                    print(f"Warning: Arguments for {base_name} run must be a list")

    # If no specific args defined, run once with no args
    if not arg_sets:
        arg_sets.append([])

    # Run for each argument set
    failed = False
    run_number = 1
    for args in arg_sets:
        # Create output file name with run number if multiple runs
        if len(arg_sets) > 1:
            output_file = os.path.abspath(f"{output_dir}/APM_{exe_name}.run{run_number}.txt")
            print(f"Running {exe_name} (run {run_number}/{len(arg_sets)})")
        else:
            output_file = os.path.abspath(f"{output_dir}/APM_{exe_name}.txt")
            print(f"Running {exe_name}")

        try:
            # Prepare command with arguments
            cmd = [f"./{exe_name}"]
            cmd.extend(args)

            # Add global arguments if provided
            if global_args:
                cmd.extend(global_args)

            print(f"    Command: {' '.join(cmd)}")

            # Store current directory
            original_dir = os.getcwd()

            try:
                # Change to executable's directory
                os.chdir(os.path.dirname(exe_path))

                # Run the executable and capture output
                with open(output_file, 'w') as f:
                    result = subprocess.run(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=300  # 5 minute timeout
                    )

                if result.returncode != 0:
                    failed = True
                print(f"    Test completed with return code {result.returncode}")

            finally:
                # Always restore original directory
                os.chdir(original_dir)

        except subprocess.TimeoutExpired:
            print(f"Error: {exe_name} timed out after 5 minutes")
            failed = True
        except Exception as e:
            print(f"Error running {exe_name}: {str(e)}")
            failed = True

        run_number += 1

    return 1 if failed else 0

def main():
    parser = argparse.ArgumentParser(description='Run all executables and capture output')
    parser.add_argument('--dir', default='.', help='Root directory to search for executables')
    parser.add_argument('--config', help='JSON configuration file for executable arguments')
    parser.add_argument('--output', default='.',  # Default to current directory
                       help='Output directory for test results')
    parser.add_argument('--args', nargs=argparse.REMAINDER,
                       help='Global arguments to pass to all executables')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Load arguments configuration
    args_config = load_args_config(args.config)

    executables = find_executables(args.dir)
    if not executables:
        print("No executables found!")
        return 1

    print(f"Found {len(executables)} executables")

    failed = []
    for exe in executables:
        ret_code = run_test(exe, args.output, args_config, args.args)
        if ret_code != 0:
            failed.append((exe.name, ret_code))

    # Print summary
    print("\nTest Summary:")
    print(f"Ran {len(executables)} tests")
    if failed:
        print(f"Failed tests ({len(failed)}):")
        for name, code in failed:
            print(f"  {name}: returned {code}")
        return failed[0][1]  # Return first failure code
    else:
        print("All tests passed!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
