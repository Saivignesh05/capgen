"""Download every model in settings.yaml, each pinned to its exact commit. Safe to re-run.

Started by `./round0 setup`. On its own, from the code folder, on a machine with internet:
    python -m scripts.download --settings ../settings.yaml --paths ../paths.yaml
"""
import argparse
import os

from huggingface_hub import snapshot_download     # CAREFUL D1

from pipeline.settings import DownloadSettings, load_paths, load_settings, model_folder

MARKER_NAME = ".download_complete.json"     # written into the model's folder after a download finishes


def download(repo: DownloadSettings, local_dir: str) -> None:
    marker_path = os.path.join(local_dir, MARKER_NAME)

    # A finished download leaves a marker holding its settings. CAREFUL D3
    if os.path.isfile(marker_path):
        with open(marker_path, encoding="utf-8") as f:
            finished = DownloadSettings.model_validate_json(f.read())
        if finished == repo:
            print(f"Already downloaded: {repo.repo_id} -> {local_dir}")
            return
        raise ValueError(f"{local_dir} already holds {finished.repo_id} at {finished.revision} with other settings. Delete it to download again.")

    # No marker: start, or resume an interrupted download. CAREFUL D4
    print(f"Downloading: {repo.repo_id} at {repo.revision} -> {local_dir}")
    snapshot_download(
        repo_id=repo.repo_id,
        revision=repo.revision,
        local_dir=local_dir,
        allow_patterns=repo.allow_patterns)

    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(repo.model_dump_json(indent=2))
    print(f"Finished: {repo.repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", required=True, help="path to settings.yaml")
    parser.add_argument("--paths", required=True, help="path to paths.yaml")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    paths = load_paths(args.paths)
    for _name, repo in settings.downloads.named():
        download(repo, model_folder(paths, repo))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# D1   Needs internet. The ASR and generator repos are gated: accept their terms on Hugging Face and
#      log in first (./round0 setup asks you to log in when needed).
# D2   A branch name like "main" changes over time; a commit hash never does, so every run gets the
#      exact same files. The hashes in settings.yaml were the latest commits on 2026-11-11.
# D3   The marker is written only after snapshot_download succeeds, so its presence means the
#      download finished. A second run with the same settings skips the repo without contacting
#      Hugging Face. A changed revision or file list for the same repo raises instead of mixing
#      two versions.
# D4   snapshot_download resumes on its own: files already downloaded are checked and skipped, and a
#      half-downloaded file continues where it stopped. So after a crash, just run the script again.
# D5   The generator loads its description tokenizer from model.config.text_encoder._name_or_path.
#      For the pinned Indic Parler-TTS revision that is google/flan-t5-large (checked in its
#      config.json). If you change the generator's revision, check it again.
# ---------------------------------------------------------------------------------------------
