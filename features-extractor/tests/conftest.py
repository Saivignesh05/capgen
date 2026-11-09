"""Shared test setup: finds the project, makes its `src` package importable
(no install needed), synthetic audio with known properties, and a model-free config.

    pytest tests                       everything that needs no model files
    FX_MODELS=1 pytest tests           + the real frozen models in models/ (fetch-models first)

Works with tests/ inside features-extractor/ or next to it (or set FX_PROJECT).
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).parent
for cand in ([Path(os.environ["FX_PROJECT"])] if os.environ.get("FX_PROJECT") else []) + [
        TESTS.parent, TESTS.parent / "features-extractor"]:
    if (cand / "src" / "__init__.py").is_file() and (cand / "config.yaml").is_file():
        ROOT = cand.resolve()
        break
else:
    raise RuntimeError("features-extractor project not found next to tests/; set FX_PROJECT")

sys.path.insert(0, str(ROOT))                       # `import src` = the features-extractor package

SR = 16000
MODELS = os.environ.get("FX_MODELS") == "1"
needs_models = pytest.mark.skipif(not MODELS, reason="set FX_MODELS=1 (after fetch-models) to test the real models")


# ------------------------------------------------------------------ synthetic audio


def t_axis(seconds, sr=SR):
    return np.arange(int(round(seconds * sr))) / sr


def sine(freq, seconds, amp=0.5, sr=SR):
    return (amp * np.sin(2 * np.pi * freq * t_axis(seconds, sr))).astype(np.float32)


def harmonic(f0, seconds, amp=0.5, sr=SR, n=12):
    """Voice-like tone; f0 constant or one value per sample."""
    k = int(round(seconds * sr))
    f0 = np.full(k, float(f0)) if np.isscalar(f0) else np.asarray(f0, float)[:k]
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(((h * f0) < sr / 2) * np.sin(h * ph) / h for h in range(1, n + 1))
    return (amp * x / np.max(np.abs(x))).astype(np.float32)


def noise(seconds, rms_dbfs, sr=SR, seed=0):
    x = np.random.default_rng(seed).standard_normal(int(round(seconds * sr)))
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (rms_dbfs / 20)).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def syllables(rate, seconds, f0=150.0, amp=0.5, sr=SR):
    """Harmonic carrier, full-depth amplitude modulation: exactly rate*seconds peaks."""
    t = t_axis(seconds, sr)
    return (harmonic(f0, seconds, 1.0, sr) * amp * 0.5 * (1 - np.cos(2 * np.pi * rate * t))).astype(np.float32)


def place(spans, seconds, make, sr=SR, floor=None):
    out = np.zeros(int(round(seconds * sr)), np.float32) if floor is None else floor.copy()
    for a, b in spans:
        seg = make(b - a)
        i = int(round(a * sr))
        out[i:i + len(seg)] += seg[:len(out) - i]
    return out


def wav_bytes(x, sr=SR, dtype=np.int16) -> bytes:
    from scipy.io import wavfile
    x = np.asarray(x)
    data = x.T if x.ndim == 2 else x
    if np.issubdtype(np.dtype(dtype), np.integer):
        data = np.clip(data, -1, 1 - 1 / 32768) * (np.iinfo(dtype).max + 1)
    buf = io.BytesIO()
    wavfile.write(buf, sr, data.astype(dtype))
    return buf.getvalue()


def utterance(seconds=5.0, sr=SR):
    """Two phrases (0.5-2.0 s at 130 Hz; 2.8-4.5 s at 130 then 200 Hz from 3.6 s), 4 syl/s,
    quiet noise floor. Ground truth for the pipeline tests."""
    t = t_axis(seconds, sr)
    v = harmonic(np.where(t < 3.6, 130.0, 200.0), seconds, 1.0, sr) * 0.5 * (1 - np.cos(2 * np.pi * 4 * t))
    gate = ((t >= 0.5) & (t < 2.0)) | ((t >= 2.8) & (t < 4.5))
    return (0.3 * v * gate + noise(seconds, -60, sr)).astype(np.float32)


# ------------------------------------------------------------------ model-free config


def config(**over) -> dict:
    cfg = {"sample_rate": SR, "device": "cpu", "models_dir": "models",
           "vad": {"impl": "energy", "threshold": 0.5, "neg_threshold": 0.35, "min_speech_s": 0.15,
                   "min_silence_s": 0.2, "pad_s": 0.05},
           "language": {"impl": "folder", "allowed": ["hi", "ta"], "min_confidence": 0.6},
           "pitch": {"method": "autocorr", "fmin_hz": 70, "fmax_hz": 450},
           "voice": {}, "space": {},
           "checks": {"min_duration_s": 0.25, "max_duration_s": 30, "min_sample_rate": 16000,
                      "max_clipping_pct": 1, "max_silence_pct": 80, "min_speech_s": 0.5, "min_snr_db": 10,
                      "language": ["hi", "ta"], "reject": ["min_duration_s", "min_speech_s", "language"]}}
    cfg.update(over)
    return cfg
