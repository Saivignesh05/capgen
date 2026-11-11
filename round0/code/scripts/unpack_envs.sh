#!/usr/bin/env bash
# Unpack every environment packed by setup_envs.sh. Run on the cluster (no internet needed):
#
#     bash scripts/unpack_envs.sh <folder with packed environments> <folder to unpack into>
#     bash scripts/unpack_envs.sh /path/to/packed-envs /path/to/envs
#
# Safe to re-run: an environment that is already unpacked is skipped.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: bash scripts/unpack_envs.sh <folder with packed environments> <folder to unpack into>" >&2
    exit 1
fi
PACKED_DIR="$1"
TARGET_DIR="$2"

shopt -s nullglob
packed_files=("$PACKED_DIR"/*.tar.gz)
if [ "${#packed_files[@]}" -eq 0 ]; then
    echo "no packed environments (*.tar.gz) in $PACKED_DIR" >&2
    exit 1
fi

for packed in "${packed_files[@]}"; do
    name="$(basename "$packed" .tar.gz)"
    target="$TARGET_DIR/$name"
    if [ -f "$target/.unpack_complete" ]; then
        echo "== $name already unpacked in $target"
        continue
    fi
    echo "== Unpacking $name into $target"
    rm -rf "$target"        # a half-finished unpack from an earlier run, if any
    mkdir -p "$target"
    tar -xzf "$packed" -C "$target"
    "$target/bin/python" "$target/bin/conda-unpack"     # fixes paths inside the environment for its new location
    touch "$target/.unpack_complete"
done

echo "== Done. Activate one with: source $TARGET_DIR/<name>/bin/activate"
