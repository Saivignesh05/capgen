"""Whisper large-v3 as an intelligibility ASR (doubt session Q31). Needs torch, transformers and
librosa, and the model folder downloaded by scripts/download_models.py.
"""
from typing import Literal

import librosa
import torch
from pydantic import PositiveFloat, PositiveInt, field_validator
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from evaluation.generator_eval.interfaces import SpeechRecogniserInterface
from evaluation.settings import Language, Strict, must_be_directory

SAMPLE_RATE = 16000     # what Whisper was trained on


class WhisperSettings(Strict):
    model_dir: str
    device: Literal["cuda", "cpu"]
    dtype: Literal["float16", "float32"]
    max_new_tokens: PositiveInt
    max_clip_seconds: PositiveFloat

    @field_validator("model_dir")
    @classmethod
    def model_dir_must_exist(cls, model_dir: str) -> str:
        return must_be_directory(model_dir)


class WhisperRecogniser(SpeechRecogniserInterface):
    def __init__(self, settings: WhisperSettings):
        self.settings = settings
        self.processor = WhisperProcessor.from_pretrained(settings.model_dir, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            settings.model_dir, torch_dtype=getattr(torch, settings.dtype), local_files_only=True)
        self.model.to(settings.device)
        self.model.eval()

    @torch.no_grad()
    def transcribe(self, audio_path: str, language: Language) -> str:
        wave, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        seconds = len(wave) / SAMPLE_RATE
        if seconds > self.settings.max_clip_seconds:        # CAREFUL W1
            raise ValueError(f"{audio_path} is {seconds:.1f}s, max_clip_seconds is {self.settings.max_clip_seconds}s")
        features = self.processor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        features = features.to(self.settings.device, dtype=getattr(torch, self.settings.dtype))
        tokens = self.model.generate(
            features, language=language, task="transcribe",     # the language is given, never detected
            max_new_tokens=self.settings.max_new_tokens, do_sample=False, num_beams=1)
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0]     # raw: the measure normalises it


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# W1   The Whisper processor silently cuts audio at 30 s. A cut transcript would look like the
#      generator dropping words, so longer audio is an error here. Long-form decoding would be
#      another implementation of the interface.
# ---------------------------------------------------------------------------------------------
