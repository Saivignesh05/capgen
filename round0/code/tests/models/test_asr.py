"""ASR with the real model. Skips unless the model is downloaded and its folder is on PYTHONPATH."""
from typing import get_args

import pytest
from pydantic import ValidationError

from helpers import repo_settings_and_paths, speech_path, write_tone

settings, paths = repo_settings_and_paths()
pytest.importorskip("soundfile")
pytest.importorskip("indic_transcribe", reason="run the tests with ./round0 test, which puts the ASR model folder on PYTHONPATH")

from pipeline.audio_types import AudioClip, AudioLanguage, AudioTranscript
from pipeline.models.asr import ASR
from pipeline.settings import asr_settings


@pytest.fixture(scope="module")
def model_settings():
    try:
        return asr_settings(settings, paths)
    except ValidationError:
        pytest.skip("ASR model not downloaded (run ./round0 setup)")


@pytest.fixture(scope="module")
def asr(model_settings):
    return ASR(model_settings)


def test_identify_returns_a_supported_language(asr, tone_path):
    assert asr.identify(AudioClip(path=tone_path)) in get_args(AudioLanguage)


def test_transcribe_returns_a_transcript_in_the_given_language(asr, tone_path):
    transcript = asr.transcribe(AudioClip(path=tone_path), "hi")
    assert isinstance(transcript, AudioTranscript)
    assert transcript.language == "hi"


def test_rejects_a_clip_longer_than_max_clip_seconds(asr, model_settings, tmp_path):
    clip = AudioClip(path=write_tone(str(tmp_path / "long.wav"), seconds=model_settings.max_clip_seconds + 1, sample_rate=16000))
    with pytest.raises(ValueError, match="max_clip_seconds"):
        asr.identify(clip)
    with pytest.raises(ValueError, match="max_clip_seconds"):
        asr.transcribe(clip, "hi")


@pytest.mark.parametrize("language", ["hi", "ta"])
def test_identifies_real_speech(asr, language):
    assert asr.identify(AudioClip(path=speech_path(language))) == language


@pytest.mark.parametrize("language", ["hi", "ta"])
def test_transcribes_real_speech(asr, language):
    transcript = asr.transcribe(AudioClip(path=speech_path(language)), language)
    assert transcript.text != ""
