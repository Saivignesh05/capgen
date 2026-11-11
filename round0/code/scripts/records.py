"""The files of a run: one record per line (JSON Lines). Shared by scripts/run.py and its stages."""
import os

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from pipeline.audio_types import AudioLanguage


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClipRecord(Record):
    """clips.jsonl: the run's input clips, fixed when the run starts."""
    clip_id: str        # file name without .wav
    path: str
    sha256: str         # proves which audio the outputs came from


class TranscriptRecord(Record):
    """transcripts.jsonl: the ASR's output."""
    clip_id: str
    language: AudioLanguage
    transcript: str


class DescriptionRecord(Record):
    """descriptions.jsonl: the captioner's output, in the competition's describe format."""
    clip_id: str
    description: str


class GenerationTask(Record):
    """One sample for the generator to make. Built by scripts/run.py from a transcript and a description."""
    clip_id: str
    sample: NonNegativeInt
    seed: int
    language: AudioLanguage
    transcript: str
    description: str


class AudioRecord(Record):
    """audio.jsonl: the generator's output, one line per sample written."""
    clip_id: str
    sample: NonNegativeInt
    seed: int
    file: str           # relative to the run folder


class ErrorRecord(Record):
    """errors.jsonl: every failure of every attempt, kept as history."""
    stage: str
    clip_id: str
    sample: NonNegativeInt | None       # only for the generate stage
    error: str


def read_records(path: str, record_type: type[Record]) -> list:
    """Every record in a JSONL file, or [] if the file does not exist yet."""
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [record_type.model_validate_json(line) for line in f if line.strip()]


def write_records(path: str, records: list[Record]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def append_record(path: str, record: Record) -> None:
    """Add one line and flush it, so a crash loses at most the item in progress."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
        f.flush()
