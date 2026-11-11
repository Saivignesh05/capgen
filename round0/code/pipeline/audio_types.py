import os
from typing import Annotated, Literal

import numpy as np
import soundfile
from pydantic import BaseModel, ConfigDict, PositiveInt, StringConstraints, field_validator

TrimmedStr = Annotated[str, StringConstraints(strip_whitespace=True)]
AudioLanguage = Literal["hi", "ta", "en"]     # Hindi, Tamil, English


class AudioClip(BaseModel):
    """One audio file. If an AudioClip exists, its path points to a readable audio file."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str

    @field_validator("path")
    @classmethod
    def must_be_readable_audio(cls, path: str) -> str:
        if not os.path.isfile(path):
            raise ValueError(f"no such file: {path}")
        # Reads only the file header, so this is fast. Fails for anything that isn't audio.
        try:
            soundfile.info(path)
        except RuntimeError as error:
            raise ValueError(f"not a readable audio file: {path} ({error})")
        return path


class AudioDescription(BaseModel):
    """English description of an audio clip. Surrounding whitespace is removed."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: TrimmedStr


class AudioTranscript(BaseModel):
    """What was said in an audio clip, in the native script of its language. Surrounding whitespace is removed."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: TrimmedStr
    language: AudioLanguage


class AudioOut(BaseModel):
    """Audio made by a generator: a mono waveform and its sample rate."""
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    waveform: np.ndarray     # 1-D float32 samples
    sample_rate: PositiveInt

