"""Download every evaluation model at its pinned revision, and record or check their SHA-256.

Run from the repository root with the evaluation environment, on a machine with internet:

    python -m scripts.download_models download --settings settings/phase0.yaml --paths settings/paths.yaml
    python -m scripts.download_models verify   --settings settings/phase0.yaml --paths settings/paths.yaml

download  fetches each Hugging Face repo to <models>/<repo_id>, clones SpeechMOS at its pinned
          commit to <models>/SpeechMOS, loads UTMOS once so its checkpoint lands in
          <models>/torch_hub, and writes settings/checkpoint_hashes.json. Safe to re-run:
          finished files are skipped. Commit checkpoint_hashes.json after the first download.
verify    recomputes the hashes (no internet needed) and compares them with
          settings/checkpoint_hashes.json. Run it on the cluster after copying <models> there.
"""
import argparse
import json
import os
import subprocess
import sys

import torch
from huggingface_hub import snapshot_download

from evaluation.manifest import folder_sha256
from evaluation.settings import EvaluationSettings, Paths, load_paths, load_settings, model_folder, speechmos_folder, torch_hub_folder

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HASHES_PATH = os.path.join(REPO_DIR, "settings", "checkpoint_hashes.json")
SPEECHMOS_URL = "https://github.com/tarepan/SpeechMOS.git"


def checkpoint_folders(settings: EvaluationSettings, paths: Paths) -> dict[str, str]:
    """{name: folder} of everything whose hash is recorded."""
    folders = {}
    for name, download in settings.downloads.named():
        folders[name] = model_folder(paths, download)
    folders["speechmos"] = speechmos_folder(paths)
    folders["torch_hub"] = torch_hub_folder(paths)
    return folders


def compute_hashes(settings: EvaluationSettings, paths: Paths) -> dict[str, str]:
    hashes = {}
    for name, folder in checkpoint_folders(settings, paths).items():
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"{name}: no such folder {folder}. Run the download first.")
        print(f"[hash] {name}: {folder}", flush=True)
        hashes[name] = folder_sha256(folder)
    return hashes


def download(settings: EvaluationSettings, paths: Paths) -> None:
    os.makedirs(paths.models, exist_ok=True)
    for name, repo in settings.downloads.named():
        folder = model_folder(paths, repo)
        print(f"[download] {name}: {repo.repo_id}@{repo.revision} -> {folder}", flush=True)
        snapshot_download(repo_id=repo.repo_id, revision=repo.revision, local_dir=folder, allow_patterns=repo.allow_patterns)

    speechmos = speechmos_folder(paths)
    if not os.path.isdir(speechmos):
        print(f"[download] SpeechMOS@{settings.naturalness.speechmos_commit} -> {speechmos}", flush=True)
        subprocess.run(["git", "clone", SPEECHMOS_URL, speechmos], check=True)
    subprocess.run(["git", "-C", speechmos, "checkout", "--quiet", settings.naturalness.speechmos_commit], check=True)

    hub = torch_hub_folder(paths)
    os.makedirs(hub, exist_ok=True)
    torch.hub.set_dir(hub)
    print(f"[download] UTMOS checkpoint -> {hub}", flush=True)
    torch.hub.load(speechmos, settings.naturalness.speechmos_model, source="local", trust_repo=True)

    hashes = compute_hashes(settings, paths)
    with open(HASHES_PATH, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[download] done. Hashes written to {HASHES_PATH}: commit that file.", flush=True)


def verify(settings: EvaluationSettings, paths: Paths) -> bool:
    with open(HASHES_PATH, encoding="utf-8") as f:
        expected = json.load(f)
    actual = compute_hashes(settings, paths)
    all_match = True
    for name in sorted(set(expected) | set(actual)):
        if expected.get(name) == actual.get(name):
            print(f"[verify] ok        {name}")
        else:
            print(f"[verify] MISMATCH  {name}: expected {expected.get(name)}, found {actual.get(name)}")
            all_match = False
    return all_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["download", "verify"])
    parser.add_argument("--settings", required=True)
    parser.add_argument("--paths", required=True)
    args = parser.parse_args()
    settings = load_settings(args.settings)
    paths = load_paths(args.paths)
    if args.action == "download":
        download(settings, paths)
    elif not verify(settings, paths):
        sys.exit(1)


if __name__ == "__main__":
    main()
