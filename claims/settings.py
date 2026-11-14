"""Claims settings: the classes that check the `claims` section of settings/phase0.yaml.
Nothing here has a default.
"""
from typing import Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveInt, field_validator

from evaluation.rubric.schema import ItemType


class Strict(BaseModel):
    """Base for every settings class: unknown or misspelled keys are errors, values cannot change."""
    model_config = ConfigDict(extra="forbid", frozen=True)


def must_be_a_probability(value: float) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"must be between 0 and 1, got: {value}")
    return value


class ComposerOptions(Strict):
    max_chars: PositiveInt
    min_probability: float
    separator: str
    type_priors: dict[ItemType, NonNegativeFloat]

    @field_validator("min_probability")
    @classmethod
    def min_probability_must_be_a_probability(cls, value: float) -> float:
        return must_be_a_probability(value)

    @field_validator("type_priors")
    @classmethod
    def every_item_type_needs_a_prior(cls, type_priors: dict) -> dict:
        missing = [item_type for item_type in get_args(ItemType) if item_type not in type_priors]
        if missing:
            raise ValueError(f"type_priors has no value for: {missing}")
        return type_priors


class ExtractorOptions(Strict):
    implementation: Literal["rules"]
    probability: float
    min_chars: PositiveInt
    fallback_type: ItemType
    type_keywords: dict[ItemType, list[str]]    # checked in the order written in the settings file

    @field_validator("probability")
    @classmethod
    def probability_must_be_a_probability(cls, value: float) -> float:
        return must_be_a_probability(value)


class ClaimsSettings(Strict):
    """Everything in the `claims` section of the settings file."""
    composer: ComposerOptions
    extractor: ExtractorOptions


def load_settings(path: str) -> ClaimsSettings:
    """Read the settings file and check its `claims` section."""
    with open(path, encoding="utf-8") as f:
        return ClaimsSettings.model_validate(yaml.safe_load(f)["claims"])
