"""Generator with the real model. Skips unless the model and tokenizer are downloaded and the device exists."""
import numpy as np
import pytest
from pydantic import ValidationError

from helpers import repo_settings, require_cuda

settings = repo_settings()
for package in ["torch", "transformers"]:
    pytest.importorskip(package)
pytest.importorskip("parler_tts", reason="needs the generator environment")

from pipeline.audio_types import AudioDescription, AudioOut, AudioTranscript
from pipeline.models.generator import Generator
from pipeline.settings import generator_settings

TRANSCRIPT = AudioTranscript(text="नमस्ते, आप कैसे हैं?", language="hi")
DESCRIPTION = AudioDescription(text="A female speaker speaks clearly at a moderate pace in a quiet room.")


@pytest.fixture(scope="module")
def generator():
    try:
        model_settings = generator_settings(settings)
    except ValidationError:
        pytest.skip("generator model or description tokenizer not downloaded (run scripts/download.py)")
    if model_settings.device == "cuda":
        require_cuda(min_total_gb=8)
    return Generator(model_settings)


def test_generates_mono_audio(generator):
    out = generator.generate(TRANSCRIPT, DESCRIPTION)
    assert isinstance(out, AudioOut)
    assert out.waveform.ndim == 1
    assert out.waveform.size > 0
    assert np.isfinite(out.waveform).all()
    assert out.sample_rate == generator.sample_rate


def test_rejects_an_empty_transcript(generator):
    with pytest.raises(ValueError, match="Transcript is empty"):
        generator.generate(AudioTranscript(text="", language="hi"), DESCRIPTION)


def test_rejects_an_empty_description(generator):
    with pytest.raises(ValueError, match="Description is empty"):
        generator.generate(TRANSCRIPT, AudioDescription(text=""))
