#!/usr/bin/env bash
# ./round0 unpack <folder>: unpack the environments made by ./round0 pack into the 'environments'
# folder of paths.yaml. No internet needed. Safe to re-run: unpacked environments are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ "$#" -eq 1 ] || fail "usage: ./round0 unpack <folder with asr-captioner.tar.gz and generator.tar.gz>"

PACKED="$(realpath "$1")"
for name in asr-captioner generator; do
    packed="$PACKED/$name.tar.gz"
    target="$ENVIRONMENTS/$name"
    [ -f "$packed" ] || fail "$packed is missing."
    if [ -f "$target/.unpack_complete" ]; then
        echo "== $name is already unpacked in $target"
        continue
    fi
    echo "== Unpacking $name into $target"
    rm -rf "$target"        # a half-finished unpack from an earlier attempt, if any
    mkdir -p "$target"
    tar -xzf "$packed" -C "$target"
    "$target/bin/python" "$target/bin/conda-unpack"     # fixes paths inside the environment for its new place
    touch "$target/.unpack_complete"
done
echo "== Done. Next: ./round0 check"
