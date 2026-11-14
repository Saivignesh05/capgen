"""UTMOS (through the SpeechMOS repository) as a naturalness proxy. Needs torch and librosa, the
SpeechMOS repository cloned at the pinned commit to <models>/SpeechMOS, and its checkpoint in the
torch hub cache (TORCH_HOME), both put there by scripts/download_models.py.

A proxy only: the official measure is a listener test. UTMOS was trained mostly on non-Indian
speech, so compare rounds with it; do not read it as an absolute quality.
"""
from typing import Literal

import librosa
import torch
from pydantic import field_validator

from evaluation.generator_eval.interfaces import NaturalnessScorerInterface
from evaluation.settings import Strict, must_be_directory

SAMPLE_RATE = 16000


class UtmosSettings(Strict):
    speechmos_dir: str
    torch_hub_dir: str      # holds checkpoints/<the UTMOS checkpoint>, so nothing is downloaded
    speechmos_model: str
    device: Literal["cuda", "cpu"]

    @field_validator("speechmos_dir", "torch_hub_dir")
    @classmethod
    def folders_must_exist(cls, directory: str) -> str:
        return must_be_directory(directory)


class UtmosScorer(NaturalnessScorerInterface):
    def __init__(self, settings: UtmosSettings):
        self.settings = settings
        torch.hub.set_dir(settings.torch_hub_dir)
        # source="local": the repository is read from the folder, nothing is downloaded.
        self.model = torch.hub.load(settings.speechmos_dir, settings.speechmos_model, source="local", trust_repo=True)
        self.model.to(settings.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, audio_path: str) -> float:
        wave, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        if len(wave) == 0:
            raise ValueError(f"{audio_path}: empty audio")
        batch = torch.from_numpy(wave).unsqueeze(0).to(self.settings.device)
        return float(self.model(batch, SAMPLE_RATE)[0])
