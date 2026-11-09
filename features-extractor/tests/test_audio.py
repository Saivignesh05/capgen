"""Loading: formats, mono mix, resampling, truncated and broken files."""

import numpy as np
import pytest
from conftest import SR, rms, sine, wav_bytes

from src import audio


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32, np.uint8])
def test_formats_decode_to_the_same_signal(dtype):
    x = sine(440, 0.5, 0.5)
    if dtype == np.uint8:
        from scipy.io import wavfile
        import io
        b = io.BytesIO()
        wavfile.write(b, SR, np.round(x * 127 + 128).astype(np.uint8))
        data = b.getvalue()
    else:
        data = wav_bytes(x, SR, dtype)
    c = audio.load(data)
    assert c.sr == SR and c.duration == pytest.approx(0.5)
    assert np.max(np.abs(c.audio - x)) < (0.02 if dtype == np.uint8 else 1e-3)


def test_stereo_is_mixed_and_original_kept():
    left, right = sine(300, 1.0, 0.4), np.zeros(SR, np.float32)
    c = audio.load(wav_bytes(np.stack([left, right])))
    assert c.original.shape == (2, SR)
    assert rms(c.audio) == pytest.approx(rms(left) / 2, rel=0.01)


@pytest.mark.parametrize("src", [8000, 22050, 44100, 48000])
def test_resampling_keeps_frequency_and_length(src):
    c = audio.load(wav_bytes(sine(440, 2.0, sr=src), src))
    assert c.sr == SR and c.original_sr == src and len(c.audio) == pytest.approx(2 * SR, abs=2)
    spec = np.abs(np.fft.rfft(c.audio))
    assert np.fft.rfftfreq(len(c.audio), 1 / SR)[np.argmax(spec)] == pytest.approx(440, abs=1)


@pytest.mark.parametrize("cut", [0.5, 0.9])
def test_truncated_wav_is_rejected(tmp_path, cut):
    data = wav_bytes(sine(300, 2.0))
    short = data[:int(len(data) * cut)]
    (tmp_path / "cut.wav").write_bytes(short)
    for src in (short, tmp_path / "cut.wav"):
        with pytest.raises(audio.AudioError, match="incomplete"):
            audio.load(src)
    audio.check_complete(data)                       # complete file passes


def test_garbage_missing_and_empty(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"RIFF1234WAVEjunk")
    with pytest.raises(audio.AudioError):
        audio.load(tmp_path / "bad.wav")
    with pytest.raises(audio.AudioError, match="not found"):
        audio.load(tmp_path / "nope.wav")
    with pytest.raises(audio.AudioError):
        audio.load(wav_bytes(np.zeros(0, np.float32)))
