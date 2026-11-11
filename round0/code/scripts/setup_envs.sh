#!/usr/bin/env bash
# Build the conda environments defined in environments/, download the models, and pack the
# environments for the cluster. Run on a machine WITH internet, from anywhere:
#
#     bash scripts/setup_envs.sh <settings.yaml> <folder for packed environments> <PyTorch index URL>
#     bash scripts/setup_envs.sh ../settings.yaml /path/to/packed-envs https://download.pytorch.org/whl/cu121
#
# Safe to re-run: existing environments are reused, finished model downloads are skipped.
# See the CAREFUL codes at the bottom.
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: bash scripts/setup_envs.sh <settings.yaml> <folder for packed environments> <PyTorch index URL>" >&2
    echo "example: bash scripts/setup_envs.sh ../settings.yaml /path/to/packed-envs https://download.pytorch.org/whl/cu121" >&2
    exit 1
fi
SETTINGS="$(realpath "$1")"
PACKED_DIR="$(realpath -m "$2")"
TORCH_INDEX_URL="$3"        # CAREFUL E1
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENTS_DIR="$CODE_DIR/environments"

command -v conda >/dev/null || { echo "conda not found" >&2; exit 1; }
command -v conda-pack >/dev/null || { echo "conda-pack not found: conda install -n base -c conda-forge conda-pack" >&2; exit 1; }
mkdir -p "$PACKED_DIR"
cd "$CODE_DIR"       # so `python -m scripts.download` and `import pipeline` work

in_env() {      # in_env <environment> <command...>
    conda run --no-capture-output -n "$1" "${@:2}"
}

lock_file() {   # lock_file <environment>
    echo "$ENVIRONMENTS_DIR/$1/requirements.lock.txt"
}

build_env() {   # build_env <environment>: create from environments/<environment>/, then install packages
    local name="$1"
    if conda env list | awk '{print $1}' | grep -qx "$name"; then
        echo "== Environment $name exists, updating it"
    else
        echo "== Creating environment $name"
        conda env create -n "$name" -f "$ENVIRONMENTS_DIR/$name/environment.yml"
    fi
    if [ -f "$(lock_file "$name")" ]; then        # CAREFUL E5
        echo "== Installing the exact versions in $(lock_file "$name")"
        in_env "$name" pip install --no-deps -r "$(lock_file "$name")" --extra-index-url "$TORCH_INDEX_URL"
    else
        echo "== No lock file yet: installing from requirements.txt"
        in_env "$name" pip install torch torchaudio --index-url "$TORCH_INDEX_URL"      # torch first. CAREFUL E2
        in_env "$name" pip install -r "$ENVIRONMENTS_DIR/$name/requirements.txt"
        local overrides="$ENVIRONMENTS_DIR/$name/requirements-overrides.txt"
        if [ -f "$overrides" ]; then        # CAREFUL E8
            echo "== Installing $overrides (a warning about a version conflict is expected)"
            in_env "$name" pip install -r "$overrides"
        fi
    fi
}

# 1. Environment for the captioner, ASR, downloads and tests.
build_env asr-captioner

# 2. Download the models (needs model_dir in settings.yaml set and a Hugging Face login). CAREFUL E3
if ! in_env asr-captioner hf auth whoami >/dev/null 2>&1; then
    echo "Not logged in to Hugging Face. Run this, then run this script again:" >&2
    echo "    conda run --no-capture-output -n asr-captioner hf auth login" >&2
    exit 1
fi
# Hugging Face waits only 10 s for a reply by default, too short on slow connections. CAREFUL E7
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120
in_env asr-captioner python -m scripts.download --settings "$SETTINGS"

# 3. The ASR's own requirements, from its model folder, without changing any fixed version.
#    Skipped with a lock file, which already contains them. CAREFUL E4
if [ ! -f "$(lock_file asr-captioner)" ]; then
    ASR_DIR="$(in_env asr-captioner python -c 'import sys; from pipeline.settings import load_settings, model_folder; s = load_settings(sys.argv[1]); print(model_folder(s, s.downloads.asr))' "$SETTINGS")"
    TORCH_VERSION="$(in_env asr-captioner python -c 'import torch; print(torch.__version__)')"
    CONSTRAINTS="$(mktemp)"
    cat "$ENVIRONMENTS_DIR/asr-captioner/requirements.txt" > "$CONSTRAINTS"
    echo "torch==$TORCH_VERSION" >> "$CONSTRAINTS"
    in_env asr-captioner pip install -r "$ASR_DIR/requirements.txt" -c "$CONSTRAINTS"
    rm "$CONSTRAINTS"
fi

# 4. Environment for the generator.
build_env generator

# 5. Record the exact version of every package, the first time only. CAREFUL E5
for name in asr-captioner generator; do
    if [ ! -f "$(lock_file "$name")" ]; then
        in_env "$name" pip freeze > "$(lock_file "$name")"
        echo "== Wrote $(lock_file "$name"): commit it"
    fi
done

# 6. Pack each environment into one file to copy to the cluster. CAREFUL E6
for name in asr-captioner generator; do
    conda-pack -n "$name" -o "$PACKED_DIR/$name.tar.gz" --force
done

echo "== Done. Copy $PACKED_DIR and the models folder to the cluster, then run scripts/unpack_envs.sh there."

# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# E1   The PyTorch index decides which CUDA build of torch is installed. Run nvidia-smi on a cluster
#      GPU node: the "CUDA Version" it shows must be at least the index's CUDA version (cu121 needs
#      12.1 or newer). Build on Linux x86_64, like the cluster, or the packed environments will
#      not run there.
# E2   torch is installed before everything else, so the other packages do not pull in a torch
#      build from the default index instead.
# E3   The ASR and generator model repos are gated: accept their terms on Hugging Face first.
#      The script stops early if you are not logged in, and says how to log in.
# E4   The ASR ships its own requirements.txt. The versions in environments/asr-captioner/requirements.txt
#      and the installed torch are passed as constraints: if the ASR needs a different version of
#      any of them, pip stops with a conflict error instead of silently replacing it. If that
#      happens, the ASR needs its own environment.
# E5   The first build installs from requirements.txt and writes requirements.lock.txt with the
#      exact version of every package. Commit it. Every later build, on any machine, installs
#      exactly those versions instead, so everyone gets the same environment. --no-deps installs
#      the list as it is, without re-checking dependencies, because the lock already lists every
#      package (and would otherwise trip over the conflict in E8). The lock pins the torch build
#      too (e.g. +cu121), so the PyTorch index given must be the same. To upgrade on purpose,
#      delete the lock file and build again.
# E6   conda-pack makes an environment that can be moved to another folder or machine. --force
#      replaces an older packed file, so the packed file always matches the environment.
# E7   A download that still times out is retried a few times and then fails. Run the script
#      again: finished files are kept and half-finished ones continue.
# E8   requirements-overrides.txt holds versions that contradict another package's declared
#      dependencies but work at runtime (see the file). Installed in a separate pip command,
#      pip warns instead of refusing.
# ---------------------------------------------------------------------------------------------
