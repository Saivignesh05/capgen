"""The training manifest: one row for every training example of every round, kept or removed.

It answers, for any model, "what exactly was it trained on, and who labelled it?". The loop and
its control run are compared by their manifests, and the SHA-256 of a manifest goes in the
submission, so the file must be the same bytes whenever it is written from the same rows.
"""
import csv
import hashlib
import json
import os

from pydantic import BaseModel, ConfigDict, NonNegativeInt, field_validator

COLUMNS = ["source_id", "audio_sha256", "round", "label_producer", "label_producer_checkpoint_sha256",
           "settings", "removal_reason"]


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str                          # where the audio comes from: a pool clip id or a synthesis request id
    audio_sha256: str                       # SHA-256 of the audio file's bytes
    round: NonNegativeInt                   # the round whose training uses this example
    label_producer: str                     # what wrote the label, e.g. "captioner_round_0", "tool:f0", "transform:noise"
    label_producer_checkpoint_sha256: str   # SHA-256 of that producer's checkpoint folder; "none" for a transform
    settings: dict[str, str]                # the settings that made the example (decoding, transform parameters, seed)
    removal_reason: str | None              # null if the example is used; otherwise why it was removed

    @field_validator("audio_sha256")
    @classmethod
    def must_be_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"not a SHA-256 in lowercase hex: {value}")
        return value

    @field_validator("removal_reason")
    @classmethod
    def removal_reason_must_say_something(cls, removal_reason: str | None) -> str | None:
        if removal_reason is not None and not removal_reason.strip():
            raise ValueError("removal_reason is empty: use null for an example that is kept")
        return removal_reason


def example_key(example: TrainingExample) -> tuple[int, str, str]:
    return (example.round, example.source_id, example.label_producer)


def write_training_manifest(examples: list[TrainingExample], path: str) -> None:
    """Write the manifest as CSV. Rows are sorted by round, source id and label producer, and the
    settings are written as JSON with sorted keys, so the same examples always give the same bytes
    whatever order they arrive in. The same (round, source, producer) twice is an error."""
    ordered = sorted(examples, key=example_key)
    for previous, current in zip(ordered, ordered[1:]):
        if example_key(previous) == example_key(current):
            raise ValueError(f"the same example appears twice: {example_key(current)}")
    with open(path + ".partial", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for example in ordered:
            removal_reason = "" if example.removal_reason is None else example.removal_reason
            writer.writerow([example.source_id, example.audio_sha256, example.round, example.label_producer,
                             example.label_producer_checkpoint_sha256,
                             json.dumps(example.settings, sort_keys=True, ensure_ascii=False), removal_reason])
    os.replace(path + ".partial", path)


def training_manifest_sha256(path: str) -> str:
    """The SHA-256 of a manifest file's bytes."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
