from abc import ABC, abstractmethod
from typing import get_args

import soundfile
from indic_transcribe import IndicTranscribe     # CAREFUL A3

from pipeline.audio_types import AudioClip, AudioLanguage, AudioTranscript
from pipeline.settings import ASRSettings


class ASRInterface(ABC):
    """Interface for ASR models."""

    @abstractmethod
    def identify(self, clip: AudioClip) -> AudioLanguage:
        """Return the language spoken in the clip."""

    @abstractmethod
    def transcribe(self, clip: AudioClip, language: AudioLanguage) -> AudioTranscript:
        """Return what was said in the clip, in the native script of `language`."""


class ASR(ASRInterface):
    def __init__(self, settings: ASRSettings):
        self.settings = settings
        self.model = IndicTranscribe.from_pretrained(settings.model_dir)

    def identify(self, clip: AudioClip) -> AudioLanguage:
        self._check_duration(clip)
        ranked = self.model.identify(
            clip.path,
            allowed_langs=list(get_args(AudioLanguage)))     # CAREFUL A4
        language, _probability = ranked[0]
        return language

    def transcribe(self, clip: AudioClip, language: AudioLanguage) -> AudioTranscript:
        self._check_duration(clip)
        text = self.model(
            clip.path,
            lang=language)      # CAREFUL A5
        return AudioTranscript(text=text, language=language)

    def _check_duration(self, clip: AudioClip) -> None:
        duration = soundfile.info(clip.path).duration
        if duration > self.settings.max_clip_seconds:
            raise ValueError(f"{clip.path} is {duration:.1f}s, max_clip_seconds is {self.settings.max_clip_seconds}s")


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# A1   The model repo is gated (Indic Open Model License): accept its terms on Hugging Face before
#      scripts/download.py can fetch it. Its folder holds the model's Python code as well as weights.
# A2   The model itself refuses audio longer than 45 s, but it was trained on clips of at most 30 s.
#      Use 30 for best accuracy (the same as the captioner).
# A3   indic_transcribe is not a pip package; its code ships inside the model folder. Put that folder
#      on PYTHONPATH before Python starts (scripts/run.py does this):
#          export PYTHONPATH=/path/to/models/bodhan-ai/indic-transcribe-core:$PYTHONPATH
#      Without it, this import fails with ModuleNotFoundError. Every .py file in that folder becomes
#      importable, so a file there with the same name as one of yours can shadow it.
# A4   allowed_langs restricts the answer to the languages in AudioLanguage, so identify always returns
#      one of them, even for audio in some other language. Check once that ranked[0] is the best
#      allowed language.
# A5   The model resamples to 16 kHz mono itself, so no sample-rate handling is needed here.
#      With lang given, the transcript is in that language's script; check how English words in
#      Hindi/Tamil speech come out (in English or transliterated).
# ---------------------------------------------------------------------------------------------
