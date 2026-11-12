#!/usr/bin/env bash
# ./round0 setup: build both environments, log in to Hugging Face if needed, download the models.
# Needs internet and conda. Safe to re-run: finished work is skipped. See CAREFUL codes at the bottom.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

command -v conda >/dev/null || fail "conda is not installed. Install Miniconda first (see README)."
mkdir -p "$MODELS" "$ENVIRONMENTS" "$RUNS"

lock_file() {   # lock_file <environment>
    echo "$ENVIRONMENT_DEFINITIONS/$1/requirements.lock.txt"
}

has_lock() {    # has_lock <environment>: true if its lock file exists and is not empty
    [ -s "$(lock_file "$1")" ]
}

build_env() {   # build_env <environment>
    local name="$1" env="$ENVIRONMENTS/$1"
    if [ -x "$env/bin/python" ]; then
        echo "== Environment $name already exists in $env"
    else
        echo "== Creating environment $name in $env"
        conda env create -p "$env" -f "$ENVIRONMENT_DEFINITIONS/$name/environment.yml"
    fi
    if has_lock "$name"; then       # CAREFUL E1
        echo "== Installing the exact package versions of $name"
        "$env/bin/python" -m pip install --no-deps -r "$(lock_file "$name")" --extra-index-url "$TORCH_INDEX_URL"
    else
        echo "== Installing the packages of $name (first build)"
        "$env/bin/python" -m pip install torch torchaudio torchvision --index-url "$TORCH_INDEX_URL"        # torch first. CAREFUL E2
        "$env/bin/python" -m pip install -r "$ENVIRONMENT_DEFINITIONS/$name/requirements.txt"
        local overrides="$ENVIRONMENT_DEFINITIONS/$name/requirements-overrides.txt"
        if [ -f "$overrides" ]; then        # CAREFUL E3
            echo "== Installing $(basename "$overrides") (a pip message about descript-audiotools and protobuf is expected)"
            "$env/bin/python" -m pip install -r "$overrides"
        fi
    fi
}

cd "$CODE_DIR"      # so `python -m scripts.<name>` and `import pipeline` work

# 1. Environment for the ASR, the captioner, downloads and tests.
build_env asr-captioner

# 2. Log in to Hugging Face if needed: two models are gated. CAREFUL E4
if ! "$ASR_CAPTIONER_ENV/bin/hf" auth whoami >/dev/null 2>&1; then
    echo "== Log in to Hugging Face: paste your access token (see README) and press Enter."
    "$ASR_CAPTIONER_ENV/bin/hf" auth login
fi

# 3. Download every model at its pinned commit.
"$ASR_CAPTIONER_PYTHON" -m scripts.download --settings "$SETTINGS_FILE" --paths "$PATHS_FILE"

# 4. The ASR's own requirements, from its model folder, without changing any fixed version.
#    A lock file already contains them. CAREFUL E5
if ! has_lock asr-captioner; then
    ASR_DIR="$("$ASR_CAPTIONER_PYTHON" -c 'import sys; from pipeline.settings import load_paths, load_settings, model_folder; print(model_folder(load_paths(sys.argv[2]), load_settings(sys.argv[1]).downloads.asr))' "$SETTINGS_FILE" "$PATHS_FILE")"
    TORCH_VERSION="$("$ASR_CAPTIONER_PYTHON" -c 'import torch; print(torch.__version__)')"
    CONSTRAINTS="$(mktemp)"
    cat "$ENVIRONMENT_DEFINITIONS/asr-captioner/requirements.txt" > "$CONSTRAINTS"
    echo "torch==$TORCH_VERSION" >> "$CONSTRAINTS"
    "$ASR_CAPTIONER_PYTHON" -m pip install -r "$ASR_DIR/requirements.txt" -c "$CONSTRAINTS"
    rm "$CONSTRAINTS"
fi

# 5. Environment for the generator.
build_env generator

# 6. Record the exact version of every package, the first time only. CAREFUL E6
for name in asr-captioner generator; do
    if ! has_lock "$name"; then
        "$ENVIRONMENTS/$name/bin/python" -m scripts.write_lock "$(lock_file "$name")"
    fi
done

echo "== Setup done. Next: ./round0 check"

# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# E1   With a lock file, every package is installed at exactly the recorded version, so everyone
#      gets the same environment. --no-deps installs the list as it is, because the lock already
#      lists every package (and would otherwise trip over the conflict in E3). The lock pins the
#      torch build too (+cu121), which is why environments/torch-index.txt names the CUDA 12.1 index.
# E2   torch is installed before everything else, so the other packages do not pull in a torch
#      build from the default index instead.
# E3   environments/generator/requirements-overrides.txt holds versions that contradict another
#      package's declared dependencies but work at runtime (see the file). Installed in a separate
#      pip command, pip prints a message instead of refusing.
# E4   The token is saved in your home folder, so you are only asked once per machine.
# E5   The ASR ships its own requirements.txt. The fixed versions and the installed torch are passed
#      as constraints: if the ASR needs something different, pip stops instead of silently
#      replacing it.
# E6   scripts/write_lock.py writes the lock atomically and refuses to write an empty one. Commit
#      the lock files. To upgrade on purpose, delete a lock file and run ./round0 setup again.
# ---------------------------------------------------------------------------------------------