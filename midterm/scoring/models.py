"""Loading the scoring models, offline, from the pinned Hugging Face cache (<model_dir>/hf).

Nothing here touches the network: scoring.run sets HF_HUB_OFFLINE=1, and every load passes the
pinned revision with local_files_only=True. A model that was not downloaded fails with a clear
error naming the repo and revision. Run scoring/download.py on a machine with internet first.
"""
import logging
import os

import numpy as np

from scoring.settings import Repo, Settings


def snapshot_folder(settings: Settings, repo: Repo) -> str:
    """Local folder of a downloaded repo at its pinned revision."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo_id=repo.repo_id, revision=repo.revision, cache_dir=settings.hf_cache,
                                 local_files_only=True)
    except Exception as error:
        raise FileNotFoundError(f"{repo.repo_id}@{repo.revision} is not in {settings.hf_cache}. "
                                f"Run scoring/download.py on a machine with internet, then copy model_dir here.") from error


def fix_weight_norm(model, folder: str) -> int:
    """Copy weight-norm tensors that from_pretrained can leave randomly initialised.

    PyTorch 2.1 renamed weight-norm parameters (weight_g / weight_v became
    parametrizations.weight.original0 / original1). Older wav2vec2 / WavLM checkpoints can then load
    their positional-convolution weights as random numbers, which quietly ruins every embedding.
    Recent transformers versions map the names themselves, so this is usually a no-op (same values);
    it is kept as insurance. Same fix as features-extractor/src/models.py. Returns tensors copied.
    """
    import torch
    state = {}
    if os.path.isfile(os.path.join(folder, "model.safetensors")):
        from safetensors.torch import load_file
        state = load_file(os.path.join(folder, "model.safetensors"))
    elif os.path.isfile(os.path.join(folder, "pytorch_model.bin")):
        state = torch.load(os.path.join(folder, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    params = dict(model.named_parameters())
    copied = 0
    for key, tensor in state.items():
        for old, new in ((".weight_g", ".parametrizations.weight.original0"),
                         (".weight_v", ".parametrizations.weight.original1")):
            target = params.get(key.replace(old, new)) if key.endswith(old) else None
            if target is not None and tuple(target.shape) == tuple(tensor.shape):
                with torch.no_grad():
                    target.copy_(tensor.to(target.dtype))
                copied += 1
    return copied


def load_wave(path: str, sample_rate: int) -> np.ndarray:
    import librosa
    wave, _ = librosa.load(path, sr=sample_rate, mono=True)
    if wave.size == 0:
        raise ValueError(f"{path}: empty audio")
    return wave


# ---------------------------------------------------------------------------------------------
# IndicConformer: the ASR Rime uses for intelligibility
# ---------------------------------------------------------------------------------------------

class IndicConformer:
    def __init__(self, settings: Settings, logger: logging.Logger):
        import torch
        from transformers import AutoModel
        repo = settings.models.asr
        snapshot_folder(settings, repo)       # clear error if missing
        self.torch = torch
        self.decoding = settings.intelligibility.decoding
        self.model = AutoModel.from_pretrained(repo.repo_id, revision=repo.revision, cache_dir=settings.hf_cache,
                                               trust_remote_code=True, local_files_only=True)    # CAREFUL M1
        logger.info("IndicConformer loaded: %s@%s, decoding=%s", repo.repo_id, repo.revision, self.decoding)

    def transcribe(self, path: str, language: str) -> str:
        wave = self.torch.from_numpy(load_wave(path, 16000)).unsqueeze(0)     # (1, samples), 16 kHz mono
        with self.torch.no_grad():
            output = self.model(wave, language, self.decoding)
        if isinstance(output, (list, tuple)):       # CAREFUL M2
            output = output[0] if output else ""
        return str(output).strip()


# ---------------------------------------------------------------------------------------------
# Speaker embeddings: WavLM-base-plus-sv x-vectors
# ---------------------------------------------------------------------------------------------

class SpeakerEmbedder:
    def __init__(self, settings: Settings, logger: logging.Logger):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        folder = snapshot_folder(settings, settings.models.speaker)
        self.torch = torch
        self.device = settings.device
        self.features = AutoFeatureExtractor.from_pretrained(folder)
        self.model = WavLMForXVector.from_pretrained(folder).to(self.device).eval()
        copied = fix_weight_norm(self.model, folder)
        logger.info("WavLM-SV loaded from %s (weight-norm tensors re-copied: %d)", folder, copied)

    def embed(self, path: str) -> np.ndarray:
        wave = load_wave(path, 16000)
        inputs = self.features(wave, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            embedding = self.model(**inputs).embeddings[0]
        embedding = self.torch.nn.functional.normalize(embedding, dim=-1)
        return embedding.float().cpu().numpy()


def count_speakers(embeddings: list[np.ndarray], threshold: float) -> int:
    """Distinct speakers among unit-length embeddings: complete-linkage clustering at `threshold`.

    Two clusters merge only if EVERY pair across them has cosine similarity >= threshold, so one
    cluster is a set of samples that all sound like the same speaker to the verifier. Order does
    not matter, and a chain of slightly different voices is not collapsed into one."""
    n = len(embeddings)
    if n == 0:
        return 0
    similarity = np.array([[float(np.dot(a, b)) for b in embeddings] for a in embeddings])
    clusters = [[i] for i in range(n)]
    while True:
        best, pair = None, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                link = min(similarity[i, j] for i in clusters[a] for j in clusters[b])
                if link >= threshold and (best is None or link > best):
                    best, pair = link, (a, b)
        if pair is None:
            return len(clusters)
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# M1   IndicConformer is gated and ships its own code (trust_remote_code). Loading by repo id and
#      revision from the cache (not from a plain folder) lets that code find its ONNX assets in the
#      same cache. It needs onnxruntime and runs on CPU through ONNX; slow but deterministic.
# M2   The model card calls model(wav, lang, "ctc") and gets a string. Some versions return a list
#      (one string per batch item), so both are handled. Check scoring.check's output once on the HPC.
# ---------------------------------------------------------------------------------------------
