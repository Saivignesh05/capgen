"""Round 0 ASR: Indic-Transcribe-Core (bodhan-ai/indic-transcribe-core), frozen.

One .wav in, one transcript string out, in the native script of the language.
Languages are restricted to Tamil (ta), Hindi (hi) and English (en).
The same model does both language identification and transcription.

Usage as a library:
    from asr import IndicTranscribeASR
    asr = IndicTranscribeASR()
    lang = asr.identify("devset/ta/clips/y.wav")                # "ta", "hi" or "en"
    text = asr.transcribe("devset/hi/clips/x.wav", lang="hi")   # language given
    text = asr.transcribe("devset/ta/clips/y.wav")              # language identified first

Usage from the command line (quick test):
    python round0/code/asr.py path/to/clip.wav --lang hi

The model repo is gated: accept its terms on Hugging Face and run `hf auth login`
once, or point `model_dir` (round0/settings.py, ASR dict) at a local copy.
"""
import argparse
import os
import sys
from abc import ABC, abstractmethod

import soundfile as sf

# ---- settings: read from round0/settings.py (ASR dict) if present ----------------
DEFAULTS = {
    "model_id": "bodhan-ai/indic-transcribe-core",
    "model_revision": None,       # pin a commit hash here for reproducible runs
    "model_dir": None,            # local copy; if None, downloaded from the Hub
    "max_clip_seconds": 45.0,     # the model refuses audio longer than 45 s
}
ROUND0_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROUND0_DIR)
try:
    from settings import ASR as _USER
except Exception:
    _USER = {}
SETTINGS = {**DEFAULTS, **_USER}

LANGS = ("ta", "hi", "en")


class ASRModel(ABC):
    """Interface: identify the language of a .wav clip, and transcribe it."""

    @abstractmethod
    def identify(self, wav_path: str) -> str:
        """Return the language of `wav_path`, one of LANGS."""

    @abstractmethod
    def transcribe(self, wav_path: str, lang: str | None = None) -> str:
        """Return the transcript of `wav_path`; lang=None means identify it first."""


class IndicTranscribeASR(ASRModel):
    """Indic-Transcribe-Core, limited to Tamil, Hindi and English."""

    def __init__(self, settings: dict | None = None):
        self.s = {**SETTINGS, **(settings or {})}
        model_dir = self.s["model_dir"]
        if not model_dir:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(self.s["model_id"], revision=self.s["model_revision"])
        # The model code ships inside the model repo, so it is imported from there.
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from indic_transcribe import IndicTranscribe
        self.model = IndicTranscribe.from_pretrained(model_dir)

    def identify(self, wav_path: str) -> str:
        self._check_wav(wav_path)
        # The model ranks all 25 of its languages; return the best one we allow.
        ranked = self.model.identify(wav_path)
        for code, _prob in ranked:
            if code in LANGS:
                return code
        raise RuntimeError(f"no supported language found for {wav_path}: {ranked[:5]}")

    def transcribe(self, wav_path: str, lang: str | None = None) -> str:
        self._check_wav(wav_path)
        if lang is None:
            lang = self.identify(wav_path)
        elif lang not in LANGS:
            raise ValueError(f"language '{lang}' not supported; use one of {LANGS}")
        # Always pass lang explicitly, so the model never decodes in an unsupported language.
        return self.model(wav_path, lang=lang).strip()

    def _check_wav(self, wav_path: str) -> None:
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(wav_path)
        if os.path.splitext(wav_path)[1].lower() != ".wav":
            raise ValueError(f"expected a .wav file, got {wav_path}")
        dur = sf.info(wav_path).duration
        if dur > self.s["max_clip_seconds"]:
            raise ValueError(f"{wav_path} is {dur:.1f}s; the model accepts at most "
                             f"{self.s['max_clip_seconds']:.0f}s")


def main():
    ap = argparse.ArgumentParser(description="Transcribe a .wav clip (Tamil, Hindi or English).")
    ap.add_argument("wav", help="path to a .wav file")
    ap.add_argument("--lang", choices=LANGS, default=None,
                    help="language of the clip; omit to identify it among ta/hi/en")
    args = ap.parse_args()

    asr = IndicTranscribeASR()
    print(asr.transcribe(args.wav, lang=args.lang))


if __name__ == "__main__":
    main()
