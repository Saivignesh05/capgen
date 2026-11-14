"""Evaluation settings: the classes that check the `evaluation` section of settings/phase0.yaml and
settings/paths.yaml, and the functions that load them.

The settings file describes the results (models, revisions, decoding, seeds, thresholds) and is the
same on every machine. paths.yaml says where things are on this machine. Nothing here has a default.
"""
import os
import re
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, field_validator

Language = Literal["hi", "ta"]


def must_be_directory(directory: str) -> str:
    if not os.path.isdir(directory):
        raise ValueError(f"No such directory: {directory}")
    return directory


class Strict(BaseModel):
    """Base for every settings class: unknown or misspelled keys are errors, values cannot change."""
    model_config = ConfigDict(extra="forbid", frozen=True)


class DownloadSettings(Strict):
    """One model repo to download, pinned to an exact commit."""
    repo_id: str
    revision: str
    allow_patterns: list[str] | None    # null downloads every file in the repo

    @field_validator("revision")
    @classmethod
    def must_be_commit_hash(cls, revision: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"revision must be a full commit hash, got: {revision}")
        return revision


class Downloads(Strict):
    grader: DownloadSettings
    whisper: DownloadSettings
    indic_conformer: DownloadSettings
    speaker: DownloadSettings

    def named(self) -> list[tuple[str, DownloadSettings]]:
        """Every download with its name, in the order of the fields above."""
        return [(name, getattr(self, name)) for name in type(self).model_fields]


class GraderOptions(Strict):
    dtype: Literal["bfloat16", "float16", "float32"]
    max_new_tokens: PositiveInt
    enable_thinking: bool
    max_description_chars: PositiveInt
    max_description_words: PositiveInt
    timing_collar_seconds: NonNegativeFloat
    max_non_ascii_letter_share: NonNegativeFloat


class BootstrapOptions(Strict):
    seed: int
    replicates: PositiveInt
    lower_position: NonNegativeInt      # 0-based position of the interval's lower end in the sorted replicates
    upper_position: NonNegativeInt

    @field_validator("upper_position")
    @classmethod
    def positions_must_be_inside_the_replicates(cls, upper_position: int, info) -> int:
        replicates = info.data.get("replicates")
        lower_position = info.data.get("lower_position")
        if replicates is not None and upper_position >= replicates:
            raise ValueError(f"upper_position {upper_position} is not below replicates {replicates}")
        if lower_position is not None and lower_position > upper_position:
            raise ValueError(f"lower_position {lower_position} is above upper_position {upper_position}")
        return upper_position


class GapClosedOptions(Strict):
    level_3_from: PositiveFloat
    level_4_from: PositiveFloat


class IntelligibilityOptions(Strict):
    whisper_max_new_tokens: PositiveInt
    indic_conformer_decoding: Literal["ctc", "rnnt"]


class SpeakerOptions(Strict):
    same_speaker_threshold: PositiveFloat
    samples_per_description: PositiveInt


class NaturalnessOptions(Strict):
    speechmos_commit: str
    speechmos_model: str


class EvaluationSettings(Strict):
    """Everything in the `evaluation` section of the settings file."""
    languages: list[Language]
    device: Literal["cuda", "cpu"]
    downloads: Downloads
    grader: GraderOptions
    bootstrap: BootstrapOptions
    gap_closed: GapClosedOptions
    intelligibility: IntelligibilityOptions
    speaker: SpeakerOptions
    naturalness: NaturalnessOptions


def load_settings(path: str) -> EvaluationSettings:
    """Read the settings file and check its `evaluation` section."""
    with open(path, encoding="utf-8") as f:
        return EvaluationSettings.model_validate(yaml.safe_load(f)["evaluation"])


class Paths(Strict):
    """Everything in paths.yaml: folders on this machine. Not part of the results."""
    models: str
    outputs: str

    @field_validator("models", "outputs")
    @classmethod
    def must_be_an_absolute_path(cls, path: str) -> str:
        if path.startswith("/absolute/path/"):
            raise ValueError(f"still the example value in paths.yaml: {path}. Replace it with a real folder.")
        if not os.path.isabs(path):
            raise ValueError(f"must be an absolute path, got: {path}")
        return path


def load_paths(path: str) -> Paths:
    """Read and check paths.yaml."""
    with open(path, encoding="utf-8") as f:
        return Paths.model_validate(yaml.safe_load(f))


def model_folder(paths: Paths, download: DownloadSettings) -> str:
    """Where a downloaded repo lives: <models>/<repo_id>."""
    return os.path.join(paths.models, download.repo_id)


def speechmos_folder(paths: Paths) -> str:
    """Where the SpeechMOS repository (UTMOS) is cloned: <models>/SpeechMOS."""
    return os.path.join(paths.models, "SpeechMOS")
