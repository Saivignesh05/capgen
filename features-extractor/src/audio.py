"""Load one audio file, check it decoded completely, make the 16 kHz mono analysis copy."""

from __future__ import annotations

import io
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np


class AudioError(Exception):
    pass


@dataclass
class Clip:
    audio: np.ndarray          # mono float32 at `sr` (what every feature uses)
    sr: int
    original: np.ndarray       # (channels, frames) float32 as decoded (levels / clipping)
    original_sr: int
    path: str | None = None
    format: str | None = None

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sr


def load(source, sample_rate: int = 16000) -> Clip:
    """source: path or WAV bytes. Raises AudioError if unreadable, truncated or empty."""
    path = None
    if not isinstance(source, (bytes, bytearray)):
        path = str(Path(source).expanduser())
        if not Path(path).is_file():
            raise AudioError(f"file not found: {path}")
    check_complete(source)
    x, sr, fmt = _decode(source)
    if x.shape[1] == 0:
        raise AudioError("audio has zero frames")
    mono = x.mean(axis=0)
    if sr != sample_rate:
        from scipy.signal import resample_poly
        g = gcd(sr, sample_rate)
        mono = resample_poly(mono, sample_rate // g, sr // g)
        peak = float(np.max(np.abs(mono)))
        if peak > 1.0:                                   # resampling overshoot
            mono = mono / peak
    return Clip(np.ascontiguousarray(mono, np.float32), sample_rate, x, sr, path, fmt)


def _decode(source):
    handle = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    try:
        if sf is not None:
            data, sr = sf.read(handle, dtype="float32", always_2d=True)
            fmt = sf.info(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source).format
            return np.ascontiguousarray(data.T), int(sr), fmt
        import warnings
        from scipy.io import wavfile
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sr, data = wavfile.read(handle)
    except Exception as exc:
        raise AudioError(f"cannot decode: {type(exc).__name__}: {exc}") from exc
    if data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128
    elif np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max + 1)
    data = data.astype(np.float32)
    return np.ascontiguousarray(data[None] if data.ndim == 1 else data.T), int(sr), "WAV"


def check_complete(source) -> None:
    """A RIFF/WAVE whose `data` chunk declares more bytes than the file holds was cut off
    (interrupted copy/download). Decoders read it silently, so check the header."""
    if isinstance(source, (bytes, bytearray)):
        head, total = bytes(source[:65536]), len(source)
    else:
        total = Path(source).stat().st_size
        with open(source, "rb") as f:
            head = f.read(65536)
    if len(head) < 12 or head[8:12] != b"WAVE" or head[:4] not in (b"RIFF", b"RIFX"):
        return
    order = "little" if head[:4] == b"RIFF" else "big"
    o = 12
    while o + 8 <= len(head):
        cid, size = head[o:o + 4], int.from_bytes(head[o + 4:o + 8], order)
        if cid == b"data":
            present = total - (o + 8)
            if size not in (0, 0xFFFFFFFF) and size > present:
                raise AudioError(f"incomplete file: data chunk declares {size} bytes, file has {present}")
            return
        o += 8 + size + (size & 1)
