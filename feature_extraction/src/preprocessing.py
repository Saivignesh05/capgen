"""
preprocessing.py - Stage 1: Audio Preprocessing and Metadata Extraction.
Inspects raw audio properties, computes initial signal health metrics,
and creates a normalized 16kHz mono analysis copy while preserving original audio.
"""

import math
import csv
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import soundfile as sf
import librosa

from feature_extraction.src.config import PipelineConfig


class AudioPreprocessor:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.config.output_audio_dir.mkdir(parents=True, exist_ok=True)

    def process(self, audio_path: str | Path) -> Dict[str, Any]:
        """
        Inspects original audio, computes metadata and quality indicators,
        and generates a standardized 16kHz mono analysis copy.
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        vaani_metadata = self._load_vaani_metadata(audio_path)

        # 1. Read original audio details using soundfile info without modifying file
        info = sf.info(str(audio_path))
        orig_sr = info.samplerate
        orig_channels = info.channels
        orig_frames = info.frames
        orig_duration = info.duration

        # Read samples in native format
        raw_audio, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")

        # 2. Compute signal statistics on original audio
        peak_amp = float(np.max(np.abs(raw_audio)))
        rms_amp = float(np.sqrt(np.mean(raw_audio**2)))
        peak_dbfs = 20.0 * math.log10(max(peak_amp, 1e-9))
        rms_dbfs = 20.0 * math.log10(max(rms_amp, 1e-9))

        # Clipping percentage: fraction of samples whose absolute value >= threshold
        clipped_samples = int(np.sum(np.abs(raw_audio) >= self.config.clipping_threshold))
        total_samples = raw_audio.size
        clipping_pct = float((clipped_samples / total_samples) * 100.0) if total_samples > 0 else 0.0

        # Silence percentage: compute short-time frame energy and measure frames below silence_db_threshold
        # Frame size = 25ms, hop = 10ms
        mono_raw = np.mean(raw_audio, axis=1)
        frame_len = int(orig_sr * 0.025)
        hop_len = int(orig_sr * 0.010)
        
        if len(mono_raw) < frame_len:
            silence_pct = 100.0 if rms_dbfs < self.config.silence_db_threshold else 0.0
        else:
            # Vectorized frame energy computation
            num_frames = 1 + (len(mono_raw) - frame_len) // hop_len
            frames = np.lib.stride_tricks.as_strided(
                mono_raw,
                shape=(num_frames, frame_len),
                strides=(mono_raw.strides[0] * hop_len, mono_raw.strides[0])
            )
            frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
            frame_db = 20.0 * np.log10(frame_rms)
            silent_frames = np.sum(frame_db < self.config.silence_db_threshold)
            silence_pct = float((silent_frames / num_frames) * 100.0)

        # 3. Create standardized analysis copy (16 kHz, Mono)
        target_sr = self.config.target_sample_rate
        if orig_channels > 1:
            analysis_mono = np.mean(raw_audio, axis=1)
        else:
            analysis_mono = raw_audio[:, 0]

        # Resample if sample rate differs from target_sr
        if orig_sr != target_sr:
            analysis_audio = librosa.resample(analysis_mono, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq")
        else:
            analysis_audio = analysis_mono.copy()

        # Prevent digital clipping during normalization/storage
        max_val = np.max(np.abs(analysis_audio))
        if max_val > 1.0:
            analysis_audio = analysis_audio / max_val

        # Save analysis copy to output_audio_dir
        analysis_filename = f"{audio_path.stem}_16k_mono.wav"
        analysis_path = self.config.output_audio_dir / analysis_filename
        sf.write(str(analysis_path), analysis_audio, target_sr, subtype="PCM_16")

        metadata = {
            "clip_id": audio_path.stem,
            "language": vaani_metadata.get("language", audio_path.parent.parent.name.title()),
            "vaani": vaani_metadata,
            "original_path": str(audio_path),
            "analysis_path": str(analysis_path),
            "duration_seconds": round(orig_duration, 4),
            "original_sample_rate": orig_sr,
            "analysis_sample_rate": target_sr,
            "num_channels": orig_channels,
            "num_samples": orig_frames,
            "peak_amplitude": {
                "linear": round(peak_amp, 6),
                "dbfs": round(peak_dbfs, 2)
            },
            "rms_amplitude": {
                "linear": round(rms_amp, 6),
                "dbfs": round(rms_dbfs, 2)
            },
            "clipping_percentage": round(clipping_pct, 4),
            "silence_percentage": round(silence_pct, 2),
            "is_analysis_copy_resampled": bool(orig_sr != target_sr)
        }

        return {
            "metadata": metadata,
            "analysis_audio": analysis_audio,
            "sample_rate": target_sr
        }

    def _load_vaani_metadata(self, audio_path: Path) -> Dict[str, Any]:
        """Load the matching local Vaani record without contacting the network."""
        language = audio_path.parent.parent.name.lower()
        metadata_path = self.config.project_root / "data" / "vaani" / "metadata" / f"{language}.csv"
        if not metadata_path.exists():
            return {}

        relative_audio = audio_path.relative_to(self.config.project_root).as_posix()
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row_path = (row.get("audio_file_path") or "").replace("\\", "/")
                if row_path == relative_audio or Path(row_path).stem == audio_path.stem:
                    return {
                        key: value
                        for key, value in row.items()
                        if value not in (None, "")
                    }
        return {}
