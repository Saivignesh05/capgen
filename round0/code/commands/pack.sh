#!/usr/bin/env bash
# ./round0 pack <folder>: pack both environments into one file each, to copy to a machine without internet.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ "$#" -eq 1 ] || fail "usage: ./round0 pack <folder to put the packed environments in>"
require_environments
command -v conda-pack >/dev/null || fail "conda-pack is not installed. Install it with: pip install conda-pack"

PACKED="$(realpath -m "$1")"
mkdir -p "$PACKED"
for name in asr-captioner generator; do
    echo "== Packing $name"
    conda-pack -p "$ENVIRONMENTS/$name" -o "$PACKED/$name.tar.gz" --force     # --force: always match the environment
done
echo "== Done: $PACKED holds asr-captioner.tar.gz and generator.tar.gz"
