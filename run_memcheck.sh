#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <path-to-cuda-binary>"
  exit 1
fi

BIN_PATH="$1"

echo "Running compute-sanitizer on $BIN_PATH"
compute-sanitizer --tool memcheck "$BIN_PATH"
