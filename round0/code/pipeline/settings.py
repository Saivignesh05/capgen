"""Settings: the classes that check settings.yaml and paths.yaml, and the functions that load them.

settings.yaml describes the results: models, revisions, decoding, seeds. It is the same on every
machine. paths.yaml says where things are on this machine: models, environments, runs. Both live
outside the code folder; scripts take them as --settings and --paths. Nothing here has a default.
"""
import os
import re
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, field_validator


def must_be_directory(directory: str) -> str:
    if not os.path.isdir(directory):
        raise ValueError(f"No such directory: {directory}")
    return directory


class Strict(BaseModel):
    """Base for every settings class: unknown or misspelled keys are errors, values cannot change."""
    model_config = ConfigDict(extra="forbid", frozen=True)


# =============================================================================================
# The sections of settings.yaml
# =============================================================================================

class DownloadSettings(Strict):
    """One model repo to download, pinned to an exact commit."""

    repo_id: str        # "owner/name", e.g. "Qwen/Qwen3-Omni-30B-A3B-Captioner"
    revision: str       # full 40-character commit hash, not a branch name. CAREFUL D2 (scripts/download.py)
    allow_patterns: list[str] | None    # null downloads every file in the repo

    @field_validator("revision")
    @classmethod
    def must_be_commit_hash(cls, revision: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"revision must be a full commit hash, got: {revision}")
        return revision

    @field_validator("repo_id")
    @classmethod
    def must_be_owner_and_name(cls, repo_id: str) -> str:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
            raise ValueError(f"repo_id must look like owner/name, got: {repo_id}")
        return repo_id

    @field_validator("allow_patterns")
    @classmethod
    def must_not_be_empty(cls, allow_patterns: list[str] | None) -> list[str] | None:
        if allow_patterns is not None and len(allow_patterns) == 0:
            raise ValueError("allow_patterns is empty, which would download nothing; use null for every file")
        return allow_patterns


class Downloads(Strict):
    captioner: DownloadSettings
    asr: DownloadSettings
    generator: DownloadSettings
    description_tokenizer: DownloadSettings     # CAREFUL D5 (scripts/download.py)

    def named(self) -> list[tuple[str, DownloadSettings]]:
        """Every download with its name, in the order of the fields above."""
        return [(name, getattr(self, name)) for name in type(self).model_fields]


class CaptionerOptions(Strict):
    dtype: Literal["bfloat16", "float16", "float32"]    # CAREFUL C1 (pipeline/models/captioner.py)
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"]  # CAREFUL C2 (pipeline/models/captioner.py)
    max_new_tokens: PositiveInt     # CAREFUL C3 (pipeline/models/captioner.py)
    max_clip_seconds: PositiveFloat     # the model card recommends at most 30


class ASROptions(Strict):
    max_clip_seconds: PositiveFloat     # CAREFUL A2 (pipeline/models/asr.py)


class GeneratorOptions(Strict):
    device: Literal["cuda", "cpu"]


class GenerationOptions(Strict):
    """How the runner (scripts/run.py) uses the generator."""
    samples_per_item: PositiveInt       # the competition asks for 10 per description
    seed: int       # every sample's seed is derived from this, the item id and the sample number


class Settings(Strict):
    """Everything in settings.yaml."""

    downloads: Downloads
    captioner: CaptionerOptions
    asr: ASROptions
    generator: GeneratorOptions
    generation: GenerationOptions


def load_settings(path: str) -> Settings:
    """Read and check settings.yaml."""
    with open(path, encoding="utf-8") as f:
        return Settings.model_validate(yaml.safe_load(f))


class Paths(Strict):
    """Everything in paths.yaml: folders on this machine. Not part of the results."""

    models: str         # every model sits in models/<repo_id>
    environments: str   # holds the asr-captioner and generator environments
    runs: str           # holds every run

    @field_validator("models", "environments", "runs")
    @classmethod
    def must_be_an_absolute_path(cls, path: str) -> str:
        # A relative path would depend on the folder a script happens to run from. CAREFUL S2
        if path.startswith("/absolute/path/"):
            raise ValueError(f"still the example value in paths.yaml: {path}. Replace it with a real folder.")
        if not os.path.isabs(path):
            raise ValueError(f"must be an absolute path (starting with /), got: {path}")
        return path


def load_paths(path: str) -> Paths:
    """Read and check paths.yaml."""
    with open(path, encoding="utf-8") as f:
        return Paths.model_validate(yaml.safe_load(f))


# =============================================================================================
# What each model is built with: its options plus its folder. CAREFUL S1
# =============================================================================================

class CaptionerSettings(CaptionerOptions):
    model_dir: str

    @field_validator("model_dir")
    @classmethod
    def model_dir_must_exist(cls, model_dir: str) -> str:
        return must_be_directory(model_dir)


class ASRSettings(ASROptions):
    model_dir: str      # CAREFUL A1 (pipeline/models/asr.py)

    @field_validator("model_dir")
    @classmethod
    def model_dir_must_exist(cls, model_dir: str) -> str:
        return must_be_directory(model_dir)


class GeneratorSettings(GeneratorOptions):
    model_dir: str
    description_tokenizer_dir: str      # CAREFUL G2 (pipeline/models/generator.py)

    @field_validator("model_dir", "description_tokenizer_dir")
    @classmethod
    def folders_must_exist(cls, directory: str) -> str:
        return must_be_directory(directory)


def model_folder(paths: Paths, download: DownloadSettings) -> str:
    """Where a downloaded repo lives: models/<repo_id>."""
    return os.path.join(paths.models, download.repo_id)


def captioner_settings(settings: Settings, paths: Paths) -> CaptionerSettings:
    return CaptionerSettings(
        model_dir=model_folder(paths, settings.downloads.captioner),
        **settings.captioner.model_dump())


def asr_settings(settings: Settings, paths: Paths) -> ASRSettings:
    return ASRSettings(
        model_dir=model_folder(paths, settings.downloads.asr),
        **settings.asr.model_dump())


def generator_settings(settings: Settings, paths: Paths) -> GeneratorSettings:
    return GeneratorSettings(
        model_dir=model_folder(paths, settings.downloads.generator),
        description_tokenizer_dir=model_folder(paths, settings.downloads.description_tokenizer),
        **settings.generator.model_dump())


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# S1   Loading the settings does not check the model folders, because scripts/download.py loads
#      them before the folders exist. Each model's settings are built, and its folders checked, only
#      when that model is about to load: Captioner(captioner_settings(settings, paths)).
# S2   Paths must be absolute. A relative path like ../models would be read from the folder a
#      script runs in, not from where paths.yaml is, and point somewhere unexpected.
# ---------------------------------------------------------------------------------------------
