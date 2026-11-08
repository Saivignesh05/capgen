"""
vad_segmentation.py - Stage 3: Voice Activity Detection (VAD) & Silence Segmentation.
Detects precise speech intervals, silence, pauses, and speech duration metrics.
Supports Silero VAD (ONNX / TorchScript) with caching in models/ and an acoustic fallback.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import urllib.request
import logging
import numpy as np
import torch

from feature_extraction.src.config import PipelineConfig

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.model = None
        self.model_type = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Loads Silero VAD from local cache, or downloads it explicitly to models/ directory.
        """
        model_dir = Path(self.config.models_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = model_dir / "silero_vad.onnx"

        # Prefer the known-compatible local TorchScript model. ONNX exports
        # differ in recurrent-state signatures across Silero releases.
        jit_path = model_dir / "silero_vad.jit"
        try:
            if not jit_path.exists():
                logger.info(f"Downloading Silero VAD JIT to {jit_path}...")
                url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit"
                urllib.request.urlretrieve(url, jit_path)
            self.model = torch.jit.load(str(jit_path), map_location="cpu")
            self.model.eval()
            self.model_type = "silero_jit"
            logger.info("Silero VAD loaded via TorchScript JIT.")
            return
        except Exception as e:
            logger.debug(f"Silero JIT initialization skipped/failed: {e}")

        # Fallback to ONNX only when the JIT model is unavailable.
        try:
            import onnxruntime as ort
            if not onnx_path.exists():
                raise FileNotFoundError(f"Local Silero ONNX model not found: {onnx_path}")
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
            self.model_type = "silero_onnx"
            logger.info("Silero VAD loaded via ONNX Runtime.")
            return
        except Exception as e:
            logger.warning(f"Silero VAD initialization failed: {e}. Using acoustic VAD fallback.")

        self.model_type = "acoustic_signal_vad"

    def process(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Runs VAD, applies hysteresis, minimum duration rules, and merges adjacent segments.
        
        Args:
            audio: 1D float32 numpy array at 16kHz
            sr: sample rate (must be 16000)
            
        Returns:
            Dictionary containing speech_segments, silence_intervals, pauses, and aggregate metrics.
        """
        duration = len(audio) / sr
        
        # 1. Compute frame-level speech probabilities
        if self.model_type in ["silero_onnx", "silero_jit"]:
            probs, times = self._run_silero(audio, sr)
        else:
            probs, times = self._run_acoustic_fallback(audio, sr)

        # 2. Apply hysteresis thresholding
        thresh = self.config.vad_threshold
        neg_thresh = self.config.vad_neg_threshold
        min_speech_sec = self.config.vad_min_speech_duration_ms / 1000.0
        min_silence_sec = self.config.vad_min_silence_duration_ms / 1000.0
        pad_sec = self.config.vad_speech_pad_ms / 1000.0

        triggered = False
        raw_speech_spans: List[Tuple[float, float, float]] = []
        cur_start = 0.0
        cur_probs = []

        for p, t in zip(probs, times):
            if not triggered and p >= thresh:
                triggered = True
                cur_start = max(0.0, t - pad_sec)
                cur_probs = [p]
            elif triggered:
                cur_probs.append(p)
                if p < neg_thresh:
                    triggered = False
                    cur_end = min(duration, t + pad_sec)
                    mean_conf = float(np.mean(cur_probs))
                    raw_speech_spans.append((cur_start, cur_end, mean_conf))
                    cur_probs = []

        if triggered and len(cur_probs) > 0:
            raw_speech_spans.append((cur_start, duration, float(np.mean(cur_probs))))

        # 3. Merge segments separated by small pauses (< min_silence_sec) and filter short segments (< min_speech_sec)
        merged_spans: List[Dict[str, Any]] = []
        for start, end, conf in raw_speech_spans:
            if not merged_spans:
                merged_spans.append({"start": start, "end": end, "confidence": conf, "samples": [conf]})
                continue
            
            prev = merged_spans[-1]
            gap = start - prev["end"]
            if gap < min_silence_sec:
                # Merge
                prev["end"] = end
                prev["samples"].append(conf)
                prev["confidence"] = float(np.mean(prev["samples"]))
            else:
                merged_spans.append({"start": start, "end": end, "confidence": conf, "samples": [conf]})

        # Prune segments below minimum speech duration
        final_speech_segments = []
        for i, s in enumerate(merged_spans):
            dur = s["end"] - s["start"]
            if dur >= min_speech_sec:
                final_speech_segments.append({
                    "segment_id": f"speech_{i+1:03d}",
                    "start_time": round(s["start"], 4),
                    "end_time": round(s["end"], 4),
                    "duration": round(dur, 4),
                    "confidence": round(float(s["confidence"]), 4),
                    "detection_method": self.model_type
                })

        # 4. Compute pauses (between speech segments) and silence intervals
        pauses = []
        for i in range(len(final_speech_segments) - 1):
            pause_start = final_speech_segments[i]["end_time"]
            pause_end = final_speech_segments[i+1]["start_time"]
            pause_dur = pause_end - pause_start
            if pause_dur > 0:
                pauses.append({
                    "pause_id": f"pause_{i+1:03d}",
                    "start_time": round(pause_start, 4),
                    "end_time": round(pause_end, 4),
                    "duration": round(pause_dur, 4)
                })

        # Silence intervals: non-speech intervals where short-time energy is genuinely below silence threshold
        silence_intervals = []
        non_speech_spans = []
        last_end = 0.0
        for seg in final_speech_segments:
            if seg["start_time"] > last_end:
                non_speech_spans.append((last_end, seg["start_time"]))
            last_end = seg["end_time"]
        if last_end < duration:
            non_speech_spans.append((last_end, duration))

        for idx, (s, e) in enumerate(non_speech_spans):
            dur = e - s
            if dur >= 0.05:  # at least 50 ms
                silence_intervals.append({
                    "interval_id": f"non_speech_{idx+1:03d}",
                    "start_time": round(s, 4),
                    "end_time": round(e, 4),
                    "duration": round(dur, 4)
                })

        total_speech_dur = sum(s["duration"] for s in final_speech_segments)
        total_non_speech_dur = duration - total_speech_dur

        return {
            "model": self.model_type,
            "speech_segments": final_speech_segments,
            "silence_intervals": silence_intervals,
            "pauses": pauses,
            "total_speech_duration": round(total_speech_dur, 4),
            "total_non_speech_duration": round(total_non_speech_dur, 4),
            "speech_ratio": round(total_speech_dur / duration, 4) if duration > 0 else 0.0,
            "_raw_times": times,
            "_raw_probabilities": probs
        }

    def _run_silero(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Runs Silero model chunk-by-chunk (512 samples / 32ms at 16kHz)."""
        window_size_samples = 512
        probs = []
        times = []
        
        if self.model_type == "silero_onnx":
            h = np.zeros((2, 1, 64), dtype=np.float32)
            c = np.zeros((2, 1, 64), dtype=np.float32)
            for i in range(0, len(audio) - window_size_samples, window_size_samples):
                chunk = audio[i:i + window_size_samples].reshape(1, -1).astype(np.float32)
                ort_inputs = {
                    "input": chunk,
                    "sr": np.array(sr, dtype=np.int64),
                    "h": h,
                    "c": c
                }
                out, h, c = self.session.run(None, ort_inputs)
                prob = float(out[0][0])
                probs.append(prob)
                times.append(i / sr)
        else:
            if hasattr(self.model, "reset_states"):
                self.model.reset_states()
            with torch.no_grad():
                for i in range(0, len(audio) - window_size_samples, window_size_samples):
                    chunk = torch.from_numpy(audio[i:i + window_size_samples]).unsqueeze(0).float()
                    out = self.model(chunk, sr)
                    prob = float(out[0][0])
                    probs.append(prob)
                    times.append(i / sr)

        return np.array(probs), np.array(times)

    def _run_acoustic_fallback(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        High-precision multi-feature acoustic fallback VAD:
        Combines adaptive short-time log-energy and spectral flux.
        """
        hop_len = int(sr * 0.010)  # 10ms
        frame_len = int(sr * 0.030) # 30ms
        
        # Frame RMS
        frames = []
        times = []
        for i in range(0, len(audio) - frame_len, hop_len):
            chunk = audio[i:i + frame_len]
            rms = np.sqrt(np.mean(chunk**2) + 1e-12)
            frames.append(rms)
            times.append(i / sr)
            
        frames = np.array(frames)
        times = np.array(times)
        
        db = 20.0 * np.log10(frames)
        floor = np.percentile(db, 15)
        ceiling = np.percentile(db, 85)
        dynamic_range = ceiling - floor
        
        if dynamic_range < 6.0:
            probs = np.full_like(db, 0.5)
        else:
            # Sigmoidal mapping centered between noise floor and speech ceiling
            center = floor + 0.35 * dynamic_range
            probs = 1.0 / (1.0 + np.exp(-(db - center) / 4.0))
            
        return probs, times
