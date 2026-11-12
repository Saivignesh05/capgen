#!/usr/bin/env bash
# Run every test, each in the environment it needs. Reads the environments folder from
# ../paths.yaml and the ASR's folder from ../settings.yaml. Run from anywhere:
#
#     code/test.sh
#
# Anything this machine cannot run is skipped, with the reason listed at the end of each part.
set -uo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUND0_DIR="$(dirname "$CODE_DIR")"
PATHS_FILE="$ROUND0_DIR/paths.yaml"

ENVIRONMENTS="$(sed -n 's/^environments:[[:space:]]*//p' "$PATHS_FILE" | sed 's/[[:space:]]*#.*//; s/[[:space:]]*$//')"
if [[ "$ENVIRONMENTS" != /* ]] || [[ "$ENVIRONMENTS" == /path/to/* ]]; then
    echo "Set 'environments' in $PATHS_FILE to an absolute path (see README, Step 6)." >&2
    exit 1
fi
ASR_CAPTIONER_PYTHON="$ENVIRONMENTS/asr-captioner/bin/python"
GENERATOR_PYTHON="$ENVIRONMENTS/generator/bin/python"
for python in "$ASR_CAPTIONER_PYTHON" "$GENERATOR_PYTHON"; do
    [ -x "$python" ] || { echo "No Python at $python (check 'environments' in $PATHS_FILE)." >&2; exit 1; }
done

cd "$CODE_DIR"
# The ASR's code ships inside its model folder; its tests need that folder on PYTHONPATH.
ASR_DIR="$("$ASR_CAPTIONER_PYTHON" -c 'import sys; from pipeline.settings import load_settings, model_folder; s = load_settings(sys.argv[1]); print(model_folder(s, s.downloads.asr))' "$ROUND0_DIR/settings.yaml")" \
    || { echo "Could not read $ROUND0_DIR/settings.yaml (is model_dir set?)." >&2; exit 1; }

echo "=============== Part 1 of 2: asr-captioner environment (ASR, captioner, settings, downloads)"
PYTHONPATH="$ASR_DIR" "$ASR_CAPTIONER_PYTHON" -m pytest -rs --ignore=tests/models/test_generator.py
first=$?

echo "=============== Part 2 of 2: generator environment"
"$GENERATOR_PYTHON" -m pytest -rs tests/models/test_generator.py
second=$?

if [ "$first" -eq 0 ] && [ "$second" -eq 0 ]; then
    echo "=============== All tests passed (skips are listed above each summary line)"
else
    echo "=============== Some tests FAILED: see above" >&2
    exit 1
fi
