#!/usr/bin/env bash
# Build the "scoring" conda environment, download the scoring models, and pack the environment for
# the HPC. Run on a Linux machine WITH internet, after round 0's setup_envs.sh:
#
#     bash midterm/setup_scoring.sh <midterm/settings.yaml> <folder for packed environments> <PyTorch index URL>
#     bash midterm/setup_scoring.sh midterm/settings.yaml "$PACKED" https://download.pytorch.org/whl/cu121
#
# Safe to re-run: the environment is reused and finished downloads are skipped. Same approach as
# round0/code/scripts/setup_envs.sh (lock file, conda-pack); see its CAREFUL codes.
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: bash midterm/setup_scoring.sh <settings.yaml> <folder for packed environments> <PyTorch index URL>" >&2
    exit 1
fi
SETTINGS="$(realpath "$1")"
PACKED_DIR="$(realpath -m "$2")"
TORCH_INDEX_URL="$3"
MIDTERM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$MIDTERM_DIR/environment"
LOCK="$ENV_DIR/requirements.lock.txt"
NAME=scoring

log() { echo "$(date '+%F %T') [setup_scoring] $*"; }
command -v conda >/dev/null || { echo "conda not found" >&2; exit 1; }
command -v conda-pack >/dev/null || { echo "conda-pack not found: pip install conda-pack" >&2; exit 1; }
mkdir -p "$PACKED_DIR"
in_env() { conda run --no-capture-output -n "$NAME" "$@"; }

if conda env list | awk '{print $1}' | grep -qx "$NAME"; then
    log "environment $NAME exists, updating it"
else
    log "creating environment $NAME"
    conda env create -n "$NAME" -f "$ENV_DIR/environment.yml"
fi
if [ -f "$LOCK" ]; then
    log "installing the exact versions in $LOCK"
    in_env pip install --no-deps -r "$LOCK" --extra-index-url "$TORCH_INDEX_URL"
else
    log "no lock file yet: installing from requirements.txt"
    in_env pip install torch torchaudio --index-url "$TORCH_INDEX_URL"
    in_env pip install -r "$ENV_DIR/requirements.txt"
    in_env pip freeze > "$LOCK"
    log "wrote $LOCK: commit it"
fi

if ! in_env hf auth whoami >/dev/null 2>&1; then
    echo "Not logged in to Hugging Face (IndicConformer is gated). Run this, then run this script again:" >&2
    echo "    conda run --no-capture-output -n $NAME hf auth login" >&2
    exit 1
fi
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120
cd "$MIDTERM_DIR"
in_env python -m scoring.download --settings "$SETTINGS"

log "packing $NAME"
conda-pack -n "$NAME" -o "$PACKED_DIR/$NAME.tar.gz" --force
log "done. Copy $PACKED_DIR/$NAME.tar.gz and the scoring model_dir to the HPC, then unpack with round0/code/scripts/unpack_envs.sh"
