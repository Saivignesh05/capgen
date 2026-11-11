"""Shared input/output types. No models needed."""
import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("soundfile")

from pipeline.audio_types import AudioClip, AudioDescription, AudioOut, AudioTranscript


def test_clip_accepts_a_real_audio_file(tone_path):
    assert AudioClip(path=tone_path).path == tone_path


def test_clip_rejects_a_missing_file(tmp_path):
    with pytest.raises(ValidationError, match="no such file"):
        AudioClip(path=str(tmp_path / "missing.wav"))


def test_clip_rejects_a_file_that_is_not_audio(tmp_path):
    fake = tmp_path / "fake.wav"
    fake.write_text("this is not audio")
    with pytest.raises(ValidationError, match="not a readable audio file"):
        AudioClip(path=str(fake))


def test_clip_rejects_unknown_fields(tone_path):
    with pytest.raises(ValidationError):
        AudioClip(path=tone_path, language="hi")


def test_clip_cannot_be_changed(tone_path):
    clip = AudioClip(path=tone_path)
    with pytest.raises(ValidationError):
        clip.path = "other.wav"


def test_description_is_trimmed_and_may_be_empty():
    assert AudioDescription(text="  a calm voice \n").text == "a calm voice"
    assert AudioDescription(text="   ").text == ""


def test_transcript_is_trimmed_and_may_be_empty():
    assert AudioTranscript(text=" नमस्ते ", language="hi").text == "नमस्ते"
    assert AudioTranscript(text="", language="ta").text == ""


def test_transcript_rejects_an_unsupported_language():
    with pytest.raises(ValidationError):
        AudioTranscript(text="bonjour", language="fr")


def test_audio_out_holds_a_waveform():
    out = AudioOut(waveform=np.zeros(100, dtype=np.float32), sample_rate=44100)
    assert out.waveform.shape == (100,)
    assert out.sample_rate == 44100

