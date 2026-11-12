"""SHA-256 hashes of every downloaded model checkpoint: write them once, check them anywhere.

A checkpoint is every file of a model's folder. Each file gets a SHA-256 hash, and the checkpoint
as a whole gets the SHA-256 of the list of "<file hash>  <file name>" lines (the format of
sha256sum). Matching hashes prove that the exact same weights are on disk.

Run from the code folder:
    python -m scripts.hash_checkpoints --settings ../settings.yaml --write ../checkpoint_hashes.json
    python -m scripts.hash_checkpoints --settings ../settings.yaml --check ../checkpoint_hashes.json
"""
import argparse
import hashlib
import json
import os
import sys

from pipeline.settings import load_settings, model_folder
from scripts.download import MARKER_NAME

SKIPPED_FOLDERS = {".cache", "__pycache__"}     # download bookkeeping and Python bytecode, not part of the checkpoint. CAREFUL H1, H2
CHUNK_BYTES = 64 * 1024 * 1024


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(folder: str) -> list[str]:
    """Every file of a checkpoint, as paths relative to its folder, sorted."""
    files = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in SKIPPED_FOLDERS]
        for name in names:
            if name != MARKER_NAME:
                files.append(os.path.relpath(os.path.join(root, name), folder))
    return sorted(files)


def hash_checkpoint(folder: str) -> dict:
    files = {name: file_sha256(os.path.join(folder, name)) for name in checkpoint_files(folder)}
    listing = "".join(f"{files[name]}  {name}\n" for name in files)
    return {"sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(), "files": files}


def hash_all(settings_path: str) -> dict:
    settings = load_settings(settings_path)
    hashes = {}
    for name, repo in settings.downloads.named():
        folder = model_folder(settings, repo)
        print(f"Hashing {name}: {folder}")
        hashes[name] = {"repo_id": repo.repo_id, "revision": repo.revision, **hash_checkpoint(folder)}
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", required=True, help="path to settings.yaml")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", help="write the hashes to this JSON file")
    action.add_argument("--check", help="compare the models on disk with the hashes in this JSON file")
    args = parser.parse_args()

    hashes = hash_all(args.settings)

    if args.write:
        with open(args.write, "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2)
        print(f"Wrote {args.write}")
        return

    with open(args.check, encoding="utf-8") as f:
        expected = json.load(f)
    if hashes == expected:
        print("All checkpoints match.")
        return
    for name in sorted(set(expected) | set(hashes)):
        if expected.get(name) != hashes.get(name):
            print(f"MISMATCH: {name}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# H1   Hugging Face keeps download bookkeeping in a .cache folder inside each model folder, and
#      download.py writes its own marker. Neither is part of the model, and both can differ
#      between machines, so neither is hashed.
# H2   The ASR's code runs from inside its model folder (it is put on PYTHONPATH), so Python writes
#      __pycache__ folders there the first time the ASR is imported. They are not part of the
#      checkpoint, and without skipping them --check fails after any run or test.
# ---------------------------------------------------------------------------------------------
