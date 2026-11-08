"""
sound_event_detection.py - Stage 4: Local Sound-Event Detection.
Implements lightweight sound-event detection (YAMNet / AudioSet ontology).
Generates frame-level class probabilities, groups candidate events (vocal vs environmental),
applies class-specific thresholds and duration pruning, refines boundaries,
and outputs timestamp uncertainty and confidence.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import csv
import logging
import urllib.request
import numpy as np
import scipy.signal
import torch
import librosa

from feature_extraction.src.config import PipelineConfig

logger = logging.getLogger(__name__)

# Standard AudioSet / YAMNet ontology mapping for target candidate events
AUDIOSET_TARGET_MAPPING = {
    "laughter": {
        "category": "vocal_event",
        "audioset_names": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Chuckle, chortle"]
    },
    "cough": {
        "category": "vocal_event",
        "audioset_names": ["Cough", "Throat clearing"]
    },
    "sigh": {
        "category": "vocal_event",
        "audioset_names": ["Sigh"]
    },
    "breathing": {
        "category": "vocal_event",
        "audioset_names": ["Breathing", "Wheeze", "Snort", "Gasp", "Pant"]
    },
    "crying": {
        "category": "vocal_event",
        "audioset_names": ["Crying, sobbing", "Baby cry, infant cry", "Whimper"]
    },
    "music": {
        "category": "environmental_sound",
        "audioset_names": ["Music", "Singing", "Acoustic guitar", "Musical instrument", "Rattle"]
    },
    "background_speech": {
        "category": "vocal_event",
        "audioset_names": ["Speech", "Conversation", "Babble", "Whispering", "Chatter"]
    },
    "animal": {
        "category": "environmental_sound",
        "audioset_names": ["Animal", "Domestic animals, pets", "Dog", "Bark", "Cat", "Purr", "Bird", "Bird vocalization, bird call, bird song"]
    },
    "impact_sound": {
        "category": "environmental_sound",
        "audioset_names": ["Knock", "Door", "Tap", "Bang", "Slap, smack", "Clatter", "Slam", "Clicking", "Crash"]
    }
}


class SoundEventDetector:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.classes: List[str] = []
        self.target_indices: Dict[str, List[int]] = {}
        self.model = None
        self.model_type = None
        self._load_class_map()
        self._initialize_model()

    def _load_class_map(self):
        """Loads or creates local AudioSet/YAMNet 521 class ontology mapping."""
        class_map_file = self.config.models_dir / "yamnet_class_map.csv"
        
        # Download AudioSet class map if missing
        if not class_map_file.exists():
            try:
                url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
                urllib.request.urlretrieve(url, class_map_file)
            except Exception as e:
                logger.warning(f"Could not download yamnet_class_map.csv: {e}. Generating target subset.")
                self._create_builtin_class_map(class_map_file)

        # Parse CSV
        self.classes = []
        try:
            with open(class_map_file, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        self.classes.append(row[2].strip().strip('"'))
        except Exception:
            self._create_builtin_class_map(class_map_file)

        # Build index mapping for our target events
        for target_label, meta in AUDIOSET_TARGET_MAPPING.items():
            self.target_indices[target_label] = []
            for target_name in meta["audioset_names"]:
                for idx, cname in enumerate(self.classes):
                    if target_name.lower() in cname.lower():
                        self.target_indices[target_label].append(idx)

    def _create_builtin_class_map(self, path: Path):
        """Creates a minimal AudioSet label reference locally."""
        all_names = []
        for m in AUDIOSET_TARGET_MAPPING.values():
            all_names.extend(m["audioset_names"])
        self.classes = sorted(list(set(all_names)))
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "mid", "display_name"])
            for idx, name in enumerate(self.classes):
                writer.writerow([idx, f"/m/00{idx:03d}", name])

    def _initialize_model(self):
        """Initializes YAMNet ONNX or acoustic signal detector."""
        onnx_path = self.config.models_dir / "yamnet.onnx"
        
        # Only use an explicitly provisioned local model.
        try:
            import onnxruntime as ort
            if not onnx_path.exists():
                raise FileNotFoundError(f"Local YAMNet model not found: {onnx_path}")
            
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
            self.model_type = "yamnet_onnx"
            logger.info("YAMNet loaded via ONNX Runtime.")
            return
        except Exception as e:
            logger.info(f"YAMNet unavailable ({e}). Sound events remain candidate-only.")

        self.model_type = "acoustic_band_sed_candidate"

    def detect(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Runs sound event detection over audio, returns candidate events with timestamps and confidence.
        """
        duration = len(audio) / sr
        window_sec = self.config.yamnet_window_size
        hop_sec = self.config.yamnet_hop_size
        uncertainty_sec = round(hop_sec / 2.0, 4)

        # 1. Obtain frame-level or short-window probabilities
        time_stamps, prob_matrix = self._compute_probabilities(audio, sr, window_sec, hop_sec)

        # 2. Extract candidate event regions per target class
        candidate_events = []
        event_counter = 1

        for target_label, meta in AUDIOSET_TARGET_MAPPING.items():
            threshold = self.config.event_thresholds.get(target_label, 0.25)
            class_probs = prob_matrix[target_label]

            # Find consecutive frames >= threshold
            is_active = class_probs >= threshold
            runs = self._find_contiguous_runs(is_active)

            for start_idx, end_idx in runs:
                t_start = max(0.0, float(time_stamps[start_idx]))
                t_end = min(duration, float(time_stamps[end_idx] + window_sec))
                raw_duration = t_end - t_start

                # Minimum duration check
                if raw_duration < self.config.yamnet_min_event_duration:
                    continue

                # Refine onset/offset using local acoustic energy envelope
                refined_start, refined_end = self._refine_boundaries(audio, sr, t_start, t_end)
                event_dur = refined_end - refined_start

                mean_conf = float(np.mean(class_probs[start_idx:end_idx+1]))
                max_conf = float(np.max(class_probs[start_idx:end_idx+1]))

                candidate_events.append({
                    "event_id": f"sed_{event_counter:03d}",
                    "start_time": round(refined_start, 4),
                    "end_time": round(refined_end, 4),
                    "duration": round(event_dur, 4),
                    "event_category": meta["category"],
                    "event_label": target_label,
                    "detection_method": self.model_type,
                    "confidence": round(mean_conf, 4),
                    "max_confidence": round(max_conf, 4),
                    "timestamp_uncertainty_seconds": uncertainty_sec,
                    "verification_status": "unverified",
                    "note": "Candidate detection from sound event classifier; boundaries refined via energy envelope."
                })
                event_counter += 1

        # Sort candidate events chronologically
        candidate_events.sort(key=lambda x: x["start_time"])

        # Merge overlapping or close events of the SAME label
        merged_events = self._merge_same_label_events(candidate_events)

        return {
            "model": self.model_type,
            "events": [],
            "candidate_events": merged_events,
            "time_stamps": time_stamps,
            "class_probabilities": prob_matrix
        }

    def _compute_probabilities(self, audio: np.ndarray, sr: int, window_sec: float, hop_sec: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Computes time series of probabilities for target classes."""
        window_samples = int(sr * window_sec)
        hop_samples = int(sr * hop_sec)
        num_windows = max(1, 1 + (len(audio) - window_samples) // hop_samples)

        time_stamps = np.array([i * hop_sec for i in range(num_windows)])
        prob_matrix: Dict[str, np.ndarray] = {label: np.zeros(num_windows) for label in AUDIOSET_TARGET_MAPPING}

        if self.model_type == "yamnet_onnx":
            try:
                for i in range(num_windows):
                    start = i * hop_samples
                    chunk = audio[start:start + window_samples].astype(np.float32)
                    features = self._yamnet_features(chunk, sr)
                    inputs = {self.session.get_inputs()[0].name: features}
                    out = self.session.run(None, inputs)[0]  # shape: (1, 521) or (frames, 521)
                    if out.ndim > 1:
                        out = np.mean(out, axis=0)
                    for target_label, indices in self.target_indices.items():
                        if indices:
                            prob_matrix[target_label][i] = float(np.max(out[indices]))
                return time_stamps, prob_matrix
            except Exception as e:
                logger.warning(f"YAMNet ONNX inference error: {e}. No sound events will be reported.")
                self.model_type = "yamnet_onnx_error"
                return time_stamps, prob_matrix

        # Acoustic spectral band feature detector (heuristic candidate generator)
        # Analyzes temporal impulsiveness, spectral flux, harmonics, and envelope characteristics
        for i in range(num_windows):
            start = i * hop_samples
            chunk = audio[start:start + window_samples]
            if len(chunk) < window_samples:
                break
            
            rms = np.sqrt(np.mean(chunk**2) + 1e-12)
            fft_mag = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
            total_energy = np.sum(fft_mag) + 1e-9

            # Energy in specific acoustic bands
            low_energy = np.sum(fft_mag[freqs < 300]) / total_energy
            mid_energy = np.sum(fft_mag[(freqs >= 300) & (freqs < 3000)]) / total_energy
            high_energy = np.sum(fft_mag[freqs >= 3000]) / total_energy
            crest_factor = (np.max(np.abs(chunk)) / rms) if rms > 0 else 0.0

            # Laughter: rhythmic burst modulation in mid band (2-6 Hz modulation frequency)
            env = np.abs(scipy.signal.hilbert(chunk))
            env_spec = np.abs(np.fft.rfft(env - np.mean(env)))
            env_freqs = np.fft.rfftfreq(len(env), 1.0 / sr)
            mod_energy = np.sum(env_spec[(env_freqs >= 2) & (env_freqs <= 6)]) / (np.sum(env_spec) + 1e-9)
            
            prob_matrix["laughter"][i] = float(np.clip(mod_energy * 2.2 * mid_energy, 0.0, 0.95))
            
            # Cough: abrupt onset burst with high crest factor and broad spectrum
            prob_matrix["cough"][i] = float(np.clip(0.15 + (crest_factor / 12.0) * mid_energy * 0.8, 0.0, 0.90)) if crest_factor > 4.5 else 0.05
            
            # Breathing / Sigh: low RMS, high spectral tilt (predominantly mid/high noise without periodicity)
            prob_matrix["breathing"][i] = float(np.clip(high_energy * 1.5, 0.0, 0.85)) if rms < 0.04 else 0.02
            prob_matrix["sigh"][i] = float(np.clip(mid_energy * 0.8, 0.0, 0.80)) if rms < 0.03 else 0.02
            
            # Impact / transient: very high crest factor > 5.5 and brief duration
            prob_matrix["impact_sound"][i] = float(np.clip((crest_factor - 3.5) / 5.0, 0.0, 0.95)) if crest_factor > 4.0 else 0.02
            
            # Background speech: mid-band energy prominent with harmonic structure (from ZCR)
            zcr_val = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)
            # Background speech is likely only when primary VAD speaker is absent → use low RMS context
            prob_matrix["background_speech"][i] = float(np.clip(mid_energy * 0.5, 0.0, 0.80)) if zcr_val < 0.15 and rms < 0.06 else 0.02
            
            # Music / tonal: very sharp narrow spectral peaks, much higher ratio threshold
            spectral_max_ratio = np.max(fft_mag) / (np.mean(fft_mag) + 1e-9)
            prob_matrix["music"][i] = float(np.clip((spectral_max_ratio - 40.0) / 60.0, 0.0, 0.90)) if spectral_max_ratio > 40 else 0.02
            prob_matrix["animal"][i] = 0.02
            prob_matrix["crying"][i] = 0.02

        return time_stamps, prob_matrix

    @staticmethod
    def _yamnet_features(audio: np.ndarray, sr: int) -> np.ndarray:
        """Create the 96-frame by 64-bin log-mel input expected by this ONNX export."""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=64,
            fmin=125,
            fmax=7500,
            center=True,
            power=2.0,
        )
        log_mel = np.log(np.maximum(mel, 1e-10) + 0.001).T
        if log_mel.shape[0] < 96:
            log_mel = np.pad(log_mel, ((0, 96 - log_mel.shape[0]), (0, 0)))
        return log_mel[:96].astype(np.float32)[np.newaxis, np.newaxis, :, :]

    def _find_contiguous_runs(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Finds contiguous blocks of True in a boolean array."""
        runs = []
        start = None
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(mask) - 1))
        return runs

    def _refine_boundaries(self, audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> Tuple[float, float]:
        """Refines window-level timestamps using high-resolution energy envelope."""
        s_sample = max(0, int((start_sec - 0.10) * sr))
        e_sample = min(len(audio), int((end_sec + 0.10) * sr))
        if e_sample - s_sample < 256:
            return start_sec, end_sec

        sub_audio = audio[s_sample:e_sample]
        # Frame energy (10ms hop)
        hop = int(sr * 0.010)
        num_sub_frames = len(sub_audio) // hop
        if num_sub_frames < 2:
            return start_sec, end_sec

        sub_rms = np.array([np.sqrt(np.mean(sub_audio[k*hop:(k+1)*hop]**2) + 1e-12) for k in range(num_sub_frames)])
        threshold = 0.25 * np.max(sub_rms)
        
        active = np.where(sub_rms >= threshold)[0]
        if len(active) > 0:
            refined_s = (s_sample + active[0] * hop) / sr
            refined_e = (s_sample + (active[-1] + 1) * hop) / sr
            return max(0.0, refined_s), min(len(audio) / sr, refined_e)

        return start_sec, end_sec

    def _merge_same_label_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merges consecutive detections of the same label if separated by less than merge_gap_sec."""
        if not events:
            return []

        merged = []
        for ev in events:
            if not merged:
                merged.append(ev)
                continue
            prev = merged[-1]
            if prev["event_label"] == ev["event_label"] and (ev["start_time"] - prev["end_time"]) <= self.config.yamnet_merge_gap_sec:
                # Merge into previous
                prev["end_time"] = ev["end_time"]
                prev["duration"] = round(prev["end_time"] - prev["start_time"], 4)
                prev["confidence"] = round((prev["confidence"] + ev["confidence"]) / 2.0, 4)
            else:
                merged.append(ev)

        # Re-index event IDs
        for i, ev in enumerate(merged):
            ev["event_id"] = f"sed_{i+1:03d}"
        return merged
