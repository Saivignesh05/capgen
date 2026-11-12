# Shared by every command: reads paths.yaml and names the files and Pythons the commands use.
# Sourced by ./round0 and the scripts in this folder; not run on its own.

ROUND0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_DIR="$ROUND0_DIR/code"
PATHS_FILE="$ROUND0_DIR/paths.yaml"
SETTINGS_FILE="$ROUND0_DIR/settings.yaml"
EVAL_SETTINGS_FILE="$ROUND0_DIR/settings_eval.yaml"
HASHES_FILE="$ROUND0_DIR/checkpoint_hashes.json"
ENVIRONMENT_DEFINITIONS="$CODE_DIR/environments"
TORCH_INDEX_URL="$(tr -d "[:space:]" < "$ENVIRONMENT_DEFINITIONS/torch-index.txt")"

fail() {
    echo "Error: $*" >&2
    exit 1
}

read_path() {   # read_path <key>: the folder given for <key> in paths.yaml
    local value
    value="$(sed -n "s/^$1:[[:space:]]*//p" "$PATHS_FILE" | sed 's/[[:space:]]#.*//; s/[[:space:]]*$//')"
    [ -n "$value" ] || fail "'$1' is missing from $PATHS_FILE."
    [[ "$value" != /absolute/path/* ]] || fail "'$1' in $PATHS_FILE is still the example value. Open the file and replace it with a real folder."
    [[ "$value" == /* ]] || fail "'$1' in $PATHS_FILE must be a full path starting with /, got: $value"
    echo "$value"
}

[ -f "$PATHS_FILE" ] || fail "paths.yaml is missing. Copy paths.example.yaml to paths.yaml and put your folders in it (see README)."
MODELS="$(read_path models)" || exit 1
ENVIRONMENTS="$(read_path environments)" || exit 1
RUNS="$(read_path runs)" || exit 1
ASR_CAPTIONER_ENV="$ENVIRONMENTS/asr-captioner"
GENERATOR_ENV="$ENVIRONMENTS/generator"
ASR_CAPTIONER_PYTHON="$ASR_CAPTIONER_ENV/bin/python"
GENERATOR_PYTHON="$GENERATOR_ENV/bin/python"

require_environments() {
    [ -x "$ASR_CAPTIONER_PYTHON" ] && [ -x "$GENERATOR_PYTHON" ] \
        || fail "the environments are not in $ENVIRONMENTS. Run ./round0 setup first (or check 'environments' in paths.yaml)."
}

# Hugging Face waits only 10 s for a reply by default, too short on slow connections.
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120
