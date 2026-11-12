"""Scoring settings: the classes that check midterm/settings.yaml, and load_settings().

Relative paths in the file are resolved against the folder the settings file is in, so the same
file works from any working directory (on a laptop or on the HPC).
"""
import os
import re
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, field_validator

STAGES = ("intelligibility", "speaker_diversity", "naturalness", "caption_generated", "judge")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Repo(Strict):
    repo_id: str
    revision: str

    @field_validator("revision")
    @classmethod
    def must_be_commit_hash(cls, revision: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"revision must be a full 40-character commit hash, got: {revision}")
        return revision


class Models(Strict):
    judge: Repo
    asr: Repo
    speaker: Repo
    xlsr: Repo

    def named(self) -> list[tuple[str, Repo]]:
        return [(name, getattr(self, name)) for name in type(self).model_fields]


class Reference(Strict):
    name: str
    measure: str
    files: list[str]
    text_field: str


class JudgeOptions(Strict):
    dtype: Literal["bfloat16", "float16", "float32"]
    max_new_tokens: PositiveInt
    max_description_words: PositiveInt


class IntelligibilityOptions(Strict):
    decoding: Literal["ctc", "rnnt"]
    cer_metric: Literal["cer_ns", "cer", "gcer"]
    hindi_lenient: bool


class SpeakerOptions(Strict):
    threshold: PositiveFloat


class NaturalnessOptions(Strict):
    metrics: list[Literal["utmos", "sigmos", "fad", "psd"]]
    xlsr_layer: PositiveInt
    speechmos_commit: str
    sigmos_commit: str
    psp_eval_dir: str | None


class RelevanceOptions(Strict):
    samples_per_item: PositiveInt


class Settings(Strict):
    model_dir: str
    device: Literal["cuda", "cpu"]
    seed: int
    bootstrap_replicates: PositiveInt
    devset: str
    stages: list[Literal[STAGES]]  # type: ignore[valid-type]
    references: list[Reference]
    models: Models
    judge: JudgeOptions
    intelligibility: IntelligibilityOptions
    speaker_diversity: SpeakerOptions
    naturalness: NaturalnessOptions
    relevance: RelevanceOptions

    @field_validator("model_dir")
    @classmethod
    def must_be_absolute(cls, model_dir: str) -> str:
        if not os.path.isabs(model_dir) and not re.match(r"^[A-Za-z]:[\\/]", model_dir):
            raise ValueError(f"model_dir must be an absolute path, got: {model_dir}")
        return model_dir

    # ---- derived folders -------------------------------------------------------------------
    @property
    def hf_cache(self) -> str:
        """Hugging Face cache holding every scoring model. Exported as HF_HUB_CACHE by scoring.run."""
        return os.path.join(self.model_dir, "hf")

    @property
    def torch_home(self) -> str:
        """TORCH_HOME: holds the UTMOS checkpoint (hub/checkpoints)."""
        return os.path.join(self.model_dir, "torch")

    @property
    def speechmos_dir(self) -> str:
        return os.path.join(self.model_dir, "SpeechMOS")

    @property
    def sigmos_dir(self) -> str:
        return os.path.join(self.model_dir, "sigmos")


def _resolve(base: str, path: str | None) -> str | None:
    if path is None or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base, path))


def load_settings(path: str) -> Settings:
    """Read and check the settings file, resolving relative paths against its folder."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    raw["devset"] = _resolve(base, raw.get("devset"))
    for reference in raw.get("references") or []:
        reference["files"] = [_resolve(base, file) for file in reference.get("files", [])]
    naturalness = raw.get("naturalness") or {}
    naturalness["psp_eval_dir"] = _resolve(base, naturalness.get("psp_eval_dir"))
    return Settings.model_validate(raw)
