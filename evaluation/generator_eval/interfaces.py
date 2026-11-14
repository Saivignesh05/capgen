"""Interfaces of the models that measure generated speech. One interface per kind of model, so an
implementation can be swapped without touching the measures that use it.

Every model returns its raw output. Normalising text, comparing embeddings and averaging scores
belong to the measures (intelligibility.py, diversity.py, similarity.py).
"""
from abc import ABC, abstractmethod

from evaluation.settings import Language


class SpeechRecogniserInterface(ABC):
    """An ASR used to measure intelligibility. Never the pipeline's own ASR: a model is not graded
    with itself (docs/evaluation_plan.md, ground rule 2)."""

    @abstractmethod
    def transcribe(self, audio_path: str, language: Language) -> str:
        """Return what the model hears in the file, in the native script of `language`."""


class SpeakerEmbedderInterface(ABC):
    """A speaker-verification model."""

    @abstractmethod
    def embed(self, audio_path: str) -> list[float]:
        """Return the speaker embedding of the file, as the model gives it (not normalised)."""


class NaturalnessScorerInterface(ABC):
    """A model that predicts a mean opinion score."""

    @abstractmethod
    def score(self, audio_path: str) -> float:
        """Return the predicted mean opinion score of the file."""
