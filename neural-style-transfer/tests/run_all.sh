#!/usr/bin/env bash
# Run the full test suite from the project root:  bash tests/run_all.sh
set -e
cd "$(dirname "$0")/.."
python tests/test_pipeline.py
echo
python tests/test_api.py
