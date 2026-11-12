#!/usr/bin/env bash
# The short way to run the pipeline. Reads the environments and runs folders from ../paths.yaml,
# then calls scripts/run.py with them. Run from anywhere:
#
#     code/run.sh new --stages transcribe --settings settings.yaml --clips /path/to/devset
#     code/run.sh new --stages transcribe,describe,generate --settings settings.yaml --clips /path/to/devset
#     code/run.sh resume --stages describe,generate --run latest
#
# --settings and --clips may be relative to the folder you are in.
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATHS_FILE="$(dirname "$CODE_DIR")/paths.yaml"

read_path() {   # read_path <key>: the value of "<key>: /some/path" in paths.yaml
    local value
    value="$(sed -n "s/^$1:[[:space:]]*//p" "$PATHS_FILE" | sed 's/[[:space:]]*#.*//; s/[[:space:]]*$//')"
    if [[ "$value" != /* ]] || [[ "$value" == /path/to/* ]]; then
        echo "Set '$1' in $PATHS_FILE to an absolute path (see README, Step 6)." >&2
        exit 1
    fi
    echo "$value"
}
ENVIRONMENTS="$(read_path environments)"
RUNS="$(read_path runs)"
[ -x "$ENVIRONMENTS/asr-captioner/bin/python" ] || { echo "No asr-captioner environment in $ENVIRONMENTS (check 'environments' in $PATHS_FILE)." >&2; exit 1; }
mkdir -p "$RUNS"

# Make --settings and --clips absolute before changing folder, so relative paths keep working.
args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --settings|--clips)
            [ "$#" -ge 2 ] || { echo "$1 needs a value" >&2; exit 1; }
            args+=("$1" "$(realpath "$2")"); shift 2 ;;
        *)
            args+=("$1"); shift ;;
    esac
done

cd "$CODE_DIR"
exec "$ENVIRONMENTS/asr-captioner/bin/python" -m scripts.run "${args[@]}" --runs "$RUNS" --environments "$ENVIRONMENTS"
