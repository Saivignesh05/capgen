"""
acoustic_changes.py - Stage 5: Local Vocal-Delivery & Acoustic Changes.
Analyzes speech segments using short overlapping frames to detect sudden acoustic transitions:
- Sudden increase or decrease in pitch (F0)
- Sudden increase or decrease in loudness (RMS dB)
- Voice quality / spectral tilt changes
- Voicing state shifts and acoustic change points.
Does NOT classify emotions or intentions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import scipy.signal
import librosa

from feature_extraction.src.config import PipelineConfig


class AcousticChangeDetector:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

    def detect_changes(
        self,
        audio: np.ndarray,
        sr: int,
        speech_segments: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extracts short-time acoustic features and detects local change points.
        """
        duration = len(audio) / sr
        frame_len = int(sr * self.config.frame_length_sec)
        hop_len = int(sr * self.config.hop_length_sec)

        # 1. Frame-level feature extraction
        times = librosa.frames_to_time(np.arange(len(audio) // hop_len), sr=sr, hop_length=hop_len)
        
        # RMS Energy & dB
        rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6))
        
        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
        
        # Fundamental Frequency F0 & Voiced Flag via pYIN
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=self.config.f0_min,
            fmax=self.config.f0_max,
            sr=sr,
            frame_length=frame_len,
            hop_length=hop_len,
            fill_na=0.0
        )

        # Spectral Tilt: Ratio of high-band energy (>2000Hz) to low-band energy (<1000Hz)
        spec_tilt = self._compute_spectral_tilt(audio, sr, frame_len, hop_len)

        # Truncate arrays to uniform length
        min_len = min(len(times), len(rms_db), len(centroid), len(zcr), len(f0), len(spec_tilt))
        times = times[:min_len]
        rms_db = rms_db[:min_len]
        centroid = centroid[:min_len]
        zcr = zcr[:min_len]
        f0 = f0[:min_len]
        spec_tilt = spec_tilt[:min_len]
        voiced_flag = voiced_flag[:min_len]

        # 2. Build speech frame mask
        speech_mask = np.zeros(min_len, dtype=bool)
        if speech_segments:
            for seg in speech_segments:
                s_t = seg["start_time"]
                e_t = seg["end_time"]
                speech_mask |= ((times >= s_t) & (times <= e_t))
        else:
            # If no VAD provided, treat regions above 25th percentile RMS as active
            speech_mask = rms_db > np.percentile(rms_db, 25)

        # 3. Detect sudden acoustic changes
        changes = []
        # A. Sudden Pitch (F0) Changes
        pitch_changes = self._detect_pitch_jumps(times, f0, voiced_flag, speech_mask)
        changes.extend(pitch_changes)

        # B. Sudden Loudness (RMS dB) Changes
        loudness_changes = self._detect_loudness_jumps(times, rms_db, speech_mask)
        changes.extend(loudness_changes)

        # C. Voice Quality / Spectral Tilt Shifts
        tilt_changes = self._detect_tilt_shifts(times, spec_tilt, voiced_flag, speech_mask)
        changes.extend(tilt_changes)

        # Consolidate overlapping frame detections into caption-level events.
        changes.sort(key=lambda x: x["start_time"])
        filtered_changes = self._consolidate_changes(
            changes,
            merge_gap_sec=self.config.change_merge_gap_sec,
            max_duration_sec=self.config.change_max_duration_sec,
        )
        filtered_changes = self._group_delivery_changes(
            filtered_changes,
            merge_gap_sec=self.config.delivery_merge_gap_sec,
            max_duration_sec=self.config.change_max_duration_sec,
        )

        # Assign unique change IDs
        for i, ch in enumerate(filtered_changes):
            ch["change_id"] = f"change_{i+1:03d}"

        return {
            "acoustic_changes": filtered_changes,
            "raw_candidate_count": len(changes),
            "consolidated_event_count": len(filtered_changes),
            "feature_contours": {
                "times": times,
                "f0": f0,
                "rms_db": rms_db,
                "centroid": centroid,
                "spectral_tilt": spec_tilt,
                "voiced_flag": voiced_flag
            }
        }

    def _compute_spectral_tilt(self, audio: np.ndarray, sr: int, frame_len: int, hop_len: int) -> np.ndarray:
        """Computes ratio of high frequency (>2000Hz) to low frequency (<1000Hz) energy."""
        S = np.abs(librosa.stft(audio, n_fft=frame_len, hop_length=hop_len))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_len)
        low_band = np.sum(S[freqs < 1000, :], axis=0) + 1e-8
        high_band = np.sum(S[freqs > 2000, :], axis=0) + 1e-8
        tilt_ratio = high_band / low_band
        return tilt_ratio

    def _detect_pitch_jumps(self, times: np.ndarray, f0: np.ndarray, voiced: np.ndarray, speech_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Detects sudden pitch jumps (>5 semitones or >60Hz) across voiced speech over a 100ms window."""
        events = []
        hop_step = 10  # 100ms comparison window (10 frames × 10ms hop)

        f0_smooth = scipy.signal.medfilt(f0.copy(), kernel_size=7)

        for i in range(hop_step, len(f0_smooth)):
            if not (speech_mask[i] and speech_mask[i - hop_step] and voiced[i] and voiced[i - hop_step]):
                continue
            f1 = f0_smooth[i - hop_step]
            f2 = f0_smooth[i]
            if f1 <= 0 or f2 <= 0:
                continue

            semitones = 12.0 * np.log2(f2 / f1)
            delta_hz = f2 - f1

            if abs(semitones) >= self.config.pitch_change_semitones or abs(delta_hz) >= self.config.pitch_change_hz:
                direction = "increase" if delta_hz > 0 else "decrease"
                events.append({
                    "start_time": round(float(times[i - hop_step]), 4),
                    "end_time": round(float(times[i]), 4),
                    "duration": round(float(times[i] - times[i - hop_step]), 4),
                    "event_category": "acoustic_change",
                    "event_label": f"pitch_{direction}",
                    "detection_method": "f0_pYIN_semitone_derivative_100ms",
                    "confidence": round(min(1.0, abs(semitones) / 10.0), 4),
                    "timestamp_uncertainty_seconds": 0.05,
                    "measurements": {
                        "from_f0_hz": round(float(f1), 2),
                        "to_f0_hz": round(float(f2), 2),
                        "delta_hz": round(float(delta_hz), 2),
                        "semitone_shift": round(float(semitones), 2)
                    },
                    "verification_status": "unverified"
                })
        return events

    def _detect_loudness_jumps(self, times: np.ndarray, rms_db: np.ndarray, speech_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Detects sudden jumps in loudness (>8 dB) within a 80ms window, using median-smoothed RMS."""
        events = []
        hop_step = 8  # 80ms window

        rms_smooth = scipy.signal.medfilt(rms_db.copy(), kernel_size=9)

        for i in range(hop_step, len(rms_smooth)):
            if not (speech_mask[i] and speech_mask[i - hop_step]):
                continue
            delta_db = rms_smooth[i] - rms_smooth[i - hop_step]
            if abs(delta_db) >= self.config.loudness_change_db:
                direction = "increase" if delta_db > 0 else "decrease"
                events.append({
                    "start_time": round(float(times[i - hop_step]), 4),
                    "end_time": round(float(times[i]), 4),
                    "duration": round(float(times[i] - times[i - hop_step]), 4),
                    "event_category": "acoustic_change",
                    "event_label": f"loudness_{direction}",
                    "detection_method": "short_time_rms_db_derivative_80ms",
                    "confidence": round(min(1.0, abs(delta_db) / 16.0), 4),
                    "timestamp_uncertainty_seconds": 0.04,
                    "measurements": {
                        "from_dbfs": round(float(rms_smooth[i - hop_step]), 2),
                        "to_dbfs": round(float(rms_smooth[i]), 2),
                        "delta_db": round(float(delta_db), 2)
                    },
                    "verification_status": "unverified"
                })
        return events

    def _detect_tilt_shifts(self, times: np.ndarray, tilt: np.ndarray, voiced: np.ndarray, speech_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Detects sudden voice quality / spectral tilt shifts using median-smoothed tilt over 120ms."""
        events = []
        hop_step = 12  # 120ms window

        tilt_smooth = scipy.signal.medfilt(tilt.copy(), kernel_size=13)

        for i in range(hop_step, len(tilt_smooth)):
            if not (speech_mask[i] and speech_mask[i - hop_step] and voiced[i] and voiced[i - hop_step]):
                continue
            t1 = tilt_smooth[i - hop_step]
            t2 = tilt_smooth[i]
            if t1 <= 0:
                continue
            ratio_change = abs(t2 - t1) / t1
            if ratio_change >= self.config.spectral_tilt_change_ratio:
                direction = "brightening" if t2 > t1 else "darkening_or_breathiness"
                events.append({
                    "start_time": round(float(times[i - hop_step]), 4),
                    "end_time": round(float(times[i]), 4),
                    "duration": round(float(times[i] - times[i - hop_step]), 4),
                    "event_category": "acoustic_change",
                    "event_label": f"voice_quality_shift_{direction}",
                    "detection_method": "spectral_tilt_ratio_divergence_120ms",
                    "confidence": round(min(1.0, ratio_change), 4),
                    "timestamp_uncertainty_seconds": 0.06,
                    "measurements": {
                        "from_tilt_ratio": round(float(t1), 4),
                        "to_tilt_ratio": round(float(t2), 4),
                        "relative_change": round(float(ratio_change), 4)
                    },
                    "verification_status": "unverified"
                })
        return events

    def _consolidate_changes(
        self,
        events: List[Dict[str, Any]],
        merge_gap_sec: float,
        max_duration_sec: float,
    ) -> List[Dict[str, Any]]:
        """Merge repeated overlapping detections into compact evidence-backed events."""
        if not events:
            return []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for event in events:
            grouped.setdefault(event["event_label"], []).append(event)

        consolidated: List[Dict[str, Any]] = []
        for label, label_events in grouped.items():
            label_events.sort(key=lambda event: event["start_time"])
            current = None
            members: List[Dict[str, Any]] = []

            for event in label_events:
                if current is None:
                    current = dict(event)
                    members = [event]
                    continue

                gap = event["start_time"] - current["end_time"]
                span = event["end_time"] - current["start_time"]
                if gap <= merge_gap_sec and span <= max_duration_sec:
                    current["start_time"] = min(current["start_time"], event["start_time"])
                    current["end_time"] = max(current["end_time"], event["end_time"])
                    members.append(event)
                    if event["confidence"] > current["confidence"]:
                        current["confidence"] = event["confidence"]
                else:
                    consolidated.append(self._finalize_consolidated_event(current, members))
                    current = dict(event)
                    members = [event]

            if current is not None:
                consolidated.append(self._finalize_consolidated_event(current, members))

        consolidated.sort(key=lambda event: (event["start_time"], event["end_time"]))
        return consolidated

    @staticmethod
    def _finalize_consolidated_event(
        event: Dict[str, Any],
        members: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Attach provenance and normalize the interval after merging detections."""
        event["start_time"] = round(float(event["start_time"]), 4)
        event["end_time"] = round(float(event["end_time"]), 4)
        event["duration"] = round(event["end_time"] - event["start_time"], 4)
        event["confidence"] = round(float(event["confidence"]), 4)
        event["evidence_count"] = len(members)
        event["source_intervals"] = [
            {
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "confidence": item["confidence"],
            }
            for item in members
        ]
        return event

    @staticmethod
    def _group_delivery_changes(
        events: List[Dict[str, Any]],
        merge_gap_sec: float,
        max_duration_sec: float,
    ) -> List[Dict[str, Any]]:
        """Combine co-occurring feature changes into one local delivery event."""
        if not events:
            return []

        grouped: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_end = 0.0
        current_start = 0.0

        for event in sorted(events, key=lambda item: item["start_time"]):
            if not current:
                current = [event]
                current_start = event["start_time"]
                current_end = event["end_time"]
                continue

            gap = event["start_time"] - current_end
            span = event["end_time"] - current_start
            if gap <= merge_gap_sec and span <= max_duration_sec:
                current.append(event)
                current_end = max(current_end, event["end_time"])
            else:
                grouped.append(current)
                current = [event]
                current_start = event["start_time"]
                current_end = event["end_time"]

        if current:
            grouped.append(current)

        result: List[Dict[str, Any]] = []
        for members in grouped:
            if len(members) == 1:
                result.append(members[0])
                continue

            labels = [member["event_label"] for member in members]
            representative = max(members, key=lambda member: member["confidence"])
            result.append({
                "start_time": round(min(member["start_time"] for member in members), 4),
                "end_time": round(max(member["end_time"] for member in members), 4),
                "duration": round(
                    max(member["end_time"] for member in members)
                    - min(member["start_time"] for member in members),
                    4,
                ),
                "event_category": "acoustic_change",
                "event_label": "delivery_change",
                "detection_method": "consolidated_acoustic_features",
                "confidence": round(float(max(member["confidence"] for member in members)), 4),
                "timestamp_uncertainty_seconds": round(
                    max(member.get("timestamp_uncertainty_seconds", 0.05) for member in members),
                    4,
                ),
                "measurements": {
                    "components": labels,
                    "component_measurements": [
                        {
                            "label": member["event_label"],
                            "measurements": member.get("measurements", {}),
                            "confidence": member["confidence"],
                        }
                        for member in members
                    ],
                },
                "evidence_count": sum(member.get("evidence_count", 1) for member in members),
                "verification_status": "unverified",
                "representative_event": representative["event_label"],
            })

        return result
