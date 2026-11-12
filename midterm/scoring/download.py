"""Download every scoring model, pinned, into model_dir. Run once on a machine WITH internet:

    python -m scoring.download --settings settings.yaml

Safe to re-run: finished files are skipped. Then copy model_dir to the HPC.

    <model_dir>/hf/          Hugging Face cache: judge, IndicConformer (gated), WavLM-SV, XLS-R
    <model_dir>/SpeechMOS/   tarepan/SpeechMOS at its pinned commit (torch.hub code for UTMOS)
    <model_dir>/torch/hub/checkpoints/utmos22_strong_step7459_v1.pt
    <model_dir>/sigmos/      sigmos.py + its ONNX model from microsoft/SIG-Challenge, pinned
The Qwen3-Omni captioner is NOT downloaded here: caption_generated uses round 0's own model folder.
"""
import argparse
import io
import logging
import os
import shutil
import sys
import tarfile
import urllib.request

from scoring.settings import load_settings

UTMOS_URL = "https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt"
SIGMOS_FILES = ["sigmos.py", "model-sigmos_1697718653_41d092e8-epo-200.onnx"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s [download] %(message)s", stream=sys.stdout)
log = logging.getLogger("download")


def fetch(url: str, path: str, min_bytes: int = 1) -> None:
    if os.path.isfile(path) and os.path.getsize(path) >= min_bytes:
        log.info("already there: %s", path)
        return
    log.info("downloading %s", url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(path + ".partial", "wb") as f:
        shutil.copyfileobj(response, f)
    if os.path.getsize(path + ".partial") < min_bytes:
        raise RuntimeError(f"{url} gave only {os.path.getsize(path + '.partial')} bytes (a Git LFS pointer?)")
    os.replace(path + ".partial", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", required=True)
    args = parser.parse_args()
    settings = load_settings(args.settings)
    os.makedirs(settings.model_dir, exist_ok=True)

    from huggingface_hub import snapshot_download
    for name, repo in settings.models.named():
        log.info("%s: %s@%s", name, repo.repo_id, repo.revision)
        try:
            folder = snapshot_download(repo_id=repo.repo_id, revision=repo.revision, cache_dir=settings.hf_cache)
        except Exception as error:
            log.error("%s failed: %r. Gated repo? Accept its terms on huggingface.co and run `hf auth login`.", name, error)
            raise
        log.info("%s: done -> %s", name, folder)

    # UTMOS: torch.hub code (pinned commit) + checkpoint, both loaded offline later.
    speechmos = settings.speechmos_dir
    if not os.path.isfile(os.path.join(speechmos, "hubconf.py")):
        url = f"https://github.com/tarepan/SpeechMOS/archive/{settings.naturalness.speechmos_commit}.tar.gz"
        log.info("downloading %s", url)
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
        top = archive.getnames()[0].split("/")[0]
        archive.extractall(settings.model_dir)
        os.replace(os.path.join(settings.model_dir, top), speechmos)
    fetch(UTMOS_URL, os.path.join(settings.torch_home, "hub", "checkpoints", os.path.basename(UTMOS_URL)), min_bytes=1_000_000)

    # SIGMOS: the ONNX file is in Git LFS; github.com/.../raw/... follows the LFS redirect.
    for name in SIGMOS_FILES:
        url = f"https://github.com/microsoft/SIG-Challenge/raw/{settings.naturalness.sigmos_commit}/ICASSP2024/sigmos/{name}"
        fetch(url, os.path.join(settings.sigmos_dir, name), min_bytes=1_000_000 if name.endswith(".onnx") else 100)

    log.info("all scoring models are in %s. psp-eval is not downloaded: set naturalness.psp_eval_dir by hand.", settings.model_dir)


if __name__ == "__main__":
    main()
