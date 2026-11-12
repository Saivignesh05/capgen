#!/usr/bin/env bash
# ./round0 test: run every test, each in the environment it needs.
# Anything this machine cannot run is skipped, with the reason listed above each summary line.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_environments
cd "$CODE_DIR"

# The ASR's code ships inside its model folder; its tests need that folder on PYTHONPATH.
ASR_DIR="$("$ASR_CAPTIONER_PYTHON" -c 'import sys; from pipeline.settings import load_paths, load_settings, model_folder; print(model_folder(load_paths(sys.argv[2]), load_settings(sys.argv[1]).downloads.asr))' "$SETTINGS_FILE" "$PATHS_FILE")" \
    || fail "could not read settings.yaml or paths.yaml."

echo "=============== Part 1 of 2: asr-captioner environment (ASR, captioner, settings, downloads)"
PYTHONPATH="$ASR_DIR" "$ASR_CAPTIONER_PYTHON" -m pytest -rs --ignore=tests/models/test_generator.py
first=$?

echo "=============== Part 2 of 2: generator environment"
"$GENERATOR_PYTHON" -m pytest -rs tests/models/test_generator.py
second=$?

if [ "$first" -eq 0 ] && [ "$second" -eq 0 ]; then
    echo "=============== All tests passed"
else
    echo "=============== Some tests FAILED: see above" >&2
    exit 1
fi
