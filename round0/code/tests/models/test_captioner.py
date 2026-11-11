"""Captioner with the real model. Skips unless the model is downloaded and the GPUs are big enough."""
import pytest
from pydantic import ValidationError

from helpers import repo_settings, require_cuda, write_tone

settings = repo_settings()
for package in ["torch", "librosa", "soundfile"]:
    pytest.importorskip(package)
pytest.importorskip("transformers", minversion="4.57", reason="needs the asr-captioner environment (transformers 4.57 or newer)")

from pipeline.audio_types import AudioClip, AudioDescription
from pipeline.models.captioner import Captioner
from pipeline.settings import captioner_settings


@pytest.fixture(scope="module")
def model_settings():
    try:
        return captioner_settings(settings)
    except ValidationError:
        pytest.skip("captioner model not downloaded (run scripts/download.py)")


@pytest.fixture(scope="module")
def captioner(model_settings):
    # About 31B parameters: ~70 GB in bfloat16/float16, ~135 GB in float32.
    require_cuda(min_total_gb=135 if model_settings.dtype == "float32" else 70)
    return Captioner(model_settings)


def test_describes_a_clip(captioner, tone_path):
    description = captioner.describe(AudioClip(path=tone_path))
    assert isinstance(description, AudioDescription)
    assert description.text != ""


def test_same_clip_gives_the_same_description(captioner, tone_path):
    clip = AudioClip(path=tone_path)
    assert captioner.describe(clip).text == captioner.describe(clip).text


def test_rejects_a_clip_longer_than_max_clip_seconds(captioner, model_settings, tmp_path):
    long_path = write_tone(str(tmp_path / "long.wav"), seconds=model_settings.max_clip_seconds + 1, sample_rate=16000)
    with pytest.raises(ValueError, match="max_clip_seconds"):
        captioner.describe(AudioClip(path=long_path))
