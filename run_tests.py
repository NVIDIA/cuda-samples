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
import concurrent.futures
import threading

print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

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

def run_single_test_instance(executable, args, output_file, global_args, run_description):
    """Run a single instance of a test executable with specific arguments."""
    exe_path = str(executable)
    exe_name = executable.name

    safe_print(f"Starting {exe_name} {run_description}")

    try:
        cmd = [f"./{exe_name}"]
        cmd.extend(args)
        if global_args:
            cmd.extend(global_args)

        safe_print(f"    Command ({exe_name} {run_description}): {' '.join(cmd)}")

        # Run the executable in its own directory using cwd
        with open(output_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=300,  # 5 minute timeout
                cwd=os.path.dirname(exe_path) # Execute in the executable's directory
            )

        status = "Passed" if result.returncode == 0 else "Failed"
        safe_print(f"    Finished {exe_name} {run_description}: {status} (code {result.returncode})")
        return {"name": exe_name, "description": run_description, "return_code": result.returncode, "status": status}

    except subprocess.TimeoutExpired:
        safe_print(f"Error ({exe_name} {run_description}): Timed out after 5 minutes")
        return {"name": exe_name, "description": run_description, "return_code": -1, "status": "Timeout"}
    except Exception as e:
        safe_print(f"Error running {exe_name} {run_description}: {str(e)}")
        return {"name": exe_name, "description": run_description, "return_code": -1, "status": f"Error: {str(e)}"}

def get_gpu_count():
    """Return the number of NVIDIA GPUs visible on the system.

    The function first tries to use the `nvidia-smi` CLI which should be
    available on most systems with a CUDA-capable driver installed.  If the
    command is not present or fails we fall back to checking the
    CUDA_VISIBLE_DEVICES environment variable.  The fallback is conservative
    – if we cannot determine the GPU count we assume 0."""

    # Try the recommended NVML/nvidia-smi approach first
    try:
        smi = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if smi.returncode == 0:
            # Each GPU is reported on its own line that starts with "GPU 0:" etc.
            gpu_lines = [ln for ln in smi.stdout.strip().splitlines() if ln.strip().lower().startswith("gpu ")]
            if gpu_lines:
                return len(gpu_lines)
    except FileNotFoundError:
        # nvidia-smi is missing – may be WSL/no driver inside container etc.
        pass
    except Exception:
        # Any unexpected error – treat as unknown → 0
        pass

    # Fallback: attempt to infer from CUDA_VISIBLE_DEVICES if it is set and not empty
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"no", "none"}:
        # Handles comma-separated list like "0,1,2" or single values
        return len([v for v in visible.split(',') if v])

    # Unable to determine, assume no GPUs
    return 0

def main():
    parser = argparse.ArgumentParser(description='Run all executables and capture output')
    parser.add_argument('--dir', default='.', help='Root directory to search for executables')
    parser.add_argument('--config', help='JSON configuration file for executable arguments')
    parser.add_argument('--output', default='.',  # Default to current directory
                       help='Output directory for test results')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel tests to run')
    parser.add_argument('--args', nargs=argparse.REMAINDER,
                       help='Global arguments to pass to all executables')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Load arguments configuration
    args_config = load_args_config(args.config)

    # Determine how many GPUs are available
    gpu_count = get_gpu_count()
    if gpu_count == 0:
        print("No NVIDIA GPU detected – cannot run CUDA samples. Exiting.")
        return 1
    else:
        print(f"Detected {gpu_count} GPU(s).")

    executables = find_executables(args.dir)
    if not executables:
        print("No executables found!")
        return 1

    print(f"Found {len(executables)} executables")
    print(f"Running tests with up to {args.parallel} parallel tasks")
    print("----------------------------------------" + "-" * len(str(args.parallel)) + "\n")

    tasks = []
    for exe in executables:
        exe_name = exe.name
        base_name = normalize_exe_name(exe_name)

        # Check if this executable should be skipped globally
        if base_name in args_config and args_config[base_name].get("skip", False):
            safe_print(f"Skipping {exe_name} (marked as skip in config)")
            continue

        # Skip if the sample requires more GPUs than available
        required_gpus = args_config.get(base_name, {}).get("min_gpus", 1)
        if required_gpus > gpu_count:
            safe_print(
                f"Skipping {exe_name} (requires {required_gpus} GPU(s), only {gpu_count} available)"
            )
            continue

        arg_sets_configs = []
        if base_name in args_config:
            config = args_config[base_name]
            if "args" in config:
                if isinstance(config["args"], list):
                    arg_sets_configs.append({"args": config["args"]}) # Wrap in dict for consistency
                else:
                    safe_print(f"Warning: Arguments for {base_name} must be a list")
            elif "runs" in config:
                for i, run_config in enumerate(config["runs"]):
                    if run_config.get("skip", False):
                         safe_print(f"Skipping run {i+1} for {exe_name} (marked as skip in config)")
                         continue
                    if isinstance(run_config.get("args", []), list):
                        arg_sets_configs.append(run_config)
                    else:
                        safe_print(f"Warning: Arguments for {base_name} run {i+1} must be a list")

        # If no specific args defined, create one run with no args
        if not arg_sets_configs:
            arg_sets_configs.append({"args": []})

        # Create tasks for each run configuration
        num_runs = len(arg_sets_configs)
        for i, run_config in enumerate(arg_sets_configs):
            current_args = run_config.get("args", [])
            run_desc = f"(run {i+1}/{num_runs})" if num_runs > 1 else ""

            # Create output file name
            if num_runs > 1:
                output_file = os.path.abspath(f"{args.output}/APM_{exe_name}.run{i+1}.txt")
            else:
                output_file = os.path.abspath(f"{args.output}/APM_{exe_name}.txt")

            tasks.append({
                "executable": exe,
                "args": current_args,
                "output_file": output_file,
                "global_args": args.args,
                "description": run_desc
            })

    failed = []
    total_runs = len(tasks)
    completed_runs = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_task = {
            executor.submit(run_single_test_instance,
                            task["executable"],
                            task["args"],
                            task["output_file"],
                            task["global_args"],
                            task["description"]): task
            for task in tasks
        }

        for future in concurrent.futures.as_completed(future_to_task):
            task_info = future_to_task[future]
            completed_runs += 1
            safe_print(f"Progress: {completed_runs}/{total_runs} runs completed.")
            try:
                result = future.result()
                if result["return_code"] != 0:
                    failed.append(result)
            except Exception as exc:
                safe_print(f'Task {task_info["executable"].name} {task_info["description"]} generated an exception: {exc}')
                failed.append({
                    "name": task_info["executable"].name,
                    "description": task_info["description"],
                    "return_code": -1,
                    "status": f"Execution Exception: {exc}"
                })

    # Print summary
    print("\nTest Summary:")
    print(f"Ran {total_runs} test runs for {len(executables)} executables.")
    if failed:
        print(f"Failed runs ({len(failed)}):")
        for fail in failed:
            print(f"  {fail['name']} {fail['description']}: {fail['status']} (code {fail['return_code']})")
        # Return the return code of the first failure, or 1 if only exceptions occurred
        first_failure_code = next((f["return_code"] for f in failed if f["return_code"] != -1), 1)
        return first_failure_code
    else:
        print("All test runs passed!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
