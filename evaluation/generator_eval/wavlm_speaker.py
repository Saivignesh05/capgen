"""microsoft/wavlm-base-plus-sv as the speaker-verification model for speaker similarity and
speaker diversity. Needs torch, transformers and librosa, and the model folder downloaded by
scripts/download_models.py.
"""
from typing import Literal

import librosa
import torch
from pydantic import field_validator
from transformers import AutoFeatureExtractor, WavLMForXVector

from evaluation.generator_eval.interfaces import SpeakerEmbedderInterface
from evaluation.settings import Strict, must_be_directory

SAMPLE_RATE = 16000


class WavLMSpeakerSettings(Strict):
    model_dir: str
    device: Literal["cuda", "cpu"]

    @field_validator("model_dir")
    @classmethod
    def model_dir_must_exist(cls, model_dir: str) -> str:
        return must_be_directory(model_dir)


class WavLMSpeakerEmbedder(SpeakerEmbedderInterface):
    def __init__(self, settings: WavLMSpeakerSettings):
        self.settings = settings
        self.features = AutoFeatureExtractor.from_pretrained(settings.model_dir, local_files_only=True)
        self.model, loading = WavLMForXVector.from_pretrained(settings.model_dir, local_files_only=True, output_loading_info=True)
        if loading["missing_keys"]:     # CAREFUL V1
            raise ValueError(f"weights missing from {settings.model_dir}, left at random values: {loading['missing_keys']}")
        self.model.to(settings.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, audio_path: str) -> list[float]:
        wave, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        if len(wave) == 0:
            raise ValueError(f"{audio_path}: empty audio")
        inputs = self.features(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).to(self.settings.device)
        return self.model(**inputs).embeddings[0].float().cpu().tolist()    # raw: the measure takes the cosine


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# V1   Older wav2vec2 / WavLM checkpoints name their weight-norm tensors differently from current
#      PyTorch. A transformers version that does not map the names leaves those weights at random
#      values and only prints a warning, which quietly ruins every embedding. So a missing weight
#      is an error here.
# ---------------------------------------------------------------------------------------------
