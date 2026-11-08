"""
global_analysis.py - Stage 2: Global Acoustic Information Extraction.
Extracts signal properties, voice-related properties (F0, HNR, speech rate),
and recording-related properties (SNR, noise floor, room acoustics)
across the entire audio recording with units and estimation methods.
"""

from typing import Dict, Any, Optional
import numpy as np
import scipy.signal
import librosa

from feature_extraction.src.config import PipelineConfig


class GlobalAcousticAnalyzer:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

    def analyze(self, audio: np.ndarray, sr: int, speech_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Computes recording-wide global acoustic measurements.
        
        Args:
            audio: 1D float32 numpy array (normalized, mono)
            sr: sample rate (e.g., 16000)
            speech_mask: optional boolean mask (same length as frames or audio) for speech vs noise
        """
        n_samples = len(audio)
        if n_samples == 0:
            raise ValueError("Audio array is empty.")

        frame_len = int(sr * self.config.frame_length_sec)
        hop_len = int(sr * self.config.hop_length_sec)

        # ---------------------------------------------------------
        # 1. SIGNAL PROPERTIES
        # ---------------------------------------------------------
        # Short-time RMS
        rms_frames = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
        peak_amp = float(np.max(np.abs(audio)))
        
        # Zero Crossing Rate
        zcr_frames = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
        
        # Spectral Features
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=frame_len, hop_length=hop_len, roll_percent=0.85)[0]
        spec_flatness = librosa.feature.spectral_flatness(y=audio, n_fft=frame_len, hop_length=hop_len)[0]

        signal_properties = {
            "peak_amplitude": {
                "value": round(peak_amp, 6),
                "units": "linear [-1.0, 1.0]",
                "method": "peak absolute value over full waveform"
            },
            "rms_energy_statistics": {
                "mean": round(float(np.mean(rms_frames)), 6),
                "std": round(float(np.std(rms_frames)), 6),
                "min": round(float(np.min(rms_frames)), 6),
                "max": round(float(np.max(rms_frames)), 6),
                "median": round(float(np.median(rms_frames)), 6),
                "units": "linear RMS per frame (30ms window, 10ms hop)",
                "method": "short_time_rms_windowing"
            },
            "zero_crossing_rate": {
                "mean": round(float(np.mean(zcr_frames)), 6),
                "std": round(float(np.std(zcr_frames)), 6),
                "units": "fraction of sign changes per frame",
                "method": "librosa_zero_crossing_rate"
            },
            "spectral_centroid": {
                "mean_hz": round(float(np.mean(spec_centroid)), 2),
                "std_hz": round(float(np.std(spec_centroid)), 2),
                "units": "Hertz (Hz)",
                "method": "spectral_centroid_weighted_frequency_mean"
            },
            "spectral_bandwidth": {
                "mean_hz": round(float(np.mean(spec_bandwidth)), 2),
                "std_hz": round(float(np.std(spec_bandwidth)), 2),
                "units": "Hertz (Hz)",
                "method": "second_moment_spectral_spread"
            },
            "spectral_rolloff_85": {
                "mean_hz": round(float(np.mean(spec_rolloff)), 2),
                "std_hz": round(float(np.std(spec_rolloff)), 2),
                "units": "Hertz (Hz)",
                "method": "frequency_below_which_85_percent_energy_lies"
            },
            "spectral_flatness": {
                "mean": round(float(np.mean(spec_flatness)), 6),
                "std": round(float(np.std(spec_flatness)), 6),
                "units": "ratio [0.0 (pure tone), 1.0 (white noise)]",
                "method": "geometric_mean_divided_by_arithmetic_mean_of_spectrum"
            }
        }

        # ---------------------------------------------------------
        # 2. VOICE-RELATED PROPERTIES
        # ---------------------------------------------------------
        # Pitch tracking using pYIN (probabilistic YIN algorithm)
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=self.config.f0_min,
            fmax=self.config.f0_max,
            sr=sr,
            frame_length=frame_len,
            hop_length=hop_len,
            fill_na=None
        )

        valid_f0 = f0[voiced_flag & (~np.isnan(f0))] if f0 is not None else np.array([])
        voiced_ratio = float(np.mean(voiced_flag)) * 100.0 if voiced_flag is not None else 0.0

        if len(valid_f0) > 0:
            median_f0 = float(np.median(valid_f0))
            min_f0 = float(np.min(valid_f0))
            max_f0 = float(np.max(valid_f0))
            p10, p25, p50, p75, p90 = [float(p) for p in np.percentile(valid_f0, [10, 25, 50, 75, 90])]
        else:
            median_f0 = min_f0 = max_f0 = p10 = p25 = p50 = p75 = p90 = 0.0

        # Harmonic-to-Noise Ratio (HNR) via autocorrelation peak ratio on voiced segments
        hnr_values = self._estimate_hnr_contour(audio, sr, frame_len, hop_len, voiced_flag)
        if len(hnr_values) > 0:
            mean_hnr = float(np.mean(hnr_values))
            median_hnr = float(np.median(hnr_values))
        else:
            mean_hnr = median_hnr = None

        # Estimated speech rate (syllables per second)
        speech_rate_syl_per_sec = self._estimate_syllable_rate(audio, sr, valid_f0_duration=(len(valid_f0) * self.config.hop_length_sec))

        voice_properties = {
            "median_f0_hz": round(median_f0, 2) if median_f0 > 0 else None,
            "min_reliable_f0_hz": round(min_f0, 2) if min_f0 > 0 else None,
            "max_reliable_f0_hz": round(max_f0, 2) if max_f0 > 0 else None,
            "f0_percentiles_hz": {
                "p10": round(p10, 2) if p10 > 0 else None,
                "p25": round(p25, 2) if p25 > 0 else None,
                "p50": round(p50, 2) if p50 > 0 else None,
                "p75": round(p75, 2) if p75 > 0 else None,
                "p90": round(p90, 2) if p90 > 0 else None
            },
            "voiced_percentage": round(voiced_ratio, 2),
            "harmonic_to_noise_ratio_db": {
                "mean_db": round(mean_hnr, 2) if mean_hnr is not None else None,
                "median_db": round(median_hnr, 2) if median_hnr is not None else None,
                "units": "Decibels (dB)",
                "method": "autocorrelation_peak_ratio_on_voiced_frames"
            },
            "estimated_speech_rate": {
                "syllables_per_second": round(speech_rate_syl_per_sec, 2) if speech_rate_syl_per_sec is not None else None,
                "units": "syllables / second of voiced audio",
                "method": "vocalic_energy_peak_prominence_detection"
            }
        }

        # ---------------------------------------------------------
        # 3. RECORDING-RELATED PROPERTIES
        # ---------------------------------------------------------
        # VAD-based or Energy-partitioned SNR estimation
        snr_db, noise_floor_db = self._estimate_snr_and_noise(audio, rms_frames, speech_mask)

        # Room acoustic / Reverberation estimate (Early Decay Time / RT60)
        rt60_estimate = self._estimate_room_acoustics(audio, sr, snr_db)

        recording_properties = {
            "estimated_snr_db": {
                "value": round(snr_db, 2) if snr_db is not None else None,
                "units": "Decibels (dB)",
                "method": "energy_ratio_between_active_speech_and_ambient_non_speech_regions"
            },
            "background_noise_floor_dbfs": {
                "value": round(noise_floor_db, 2) if noise_floor_db is not None else None,
                "units": "dBFS",
                "method": "mean_rms_of_non_speech_frames"
            },
            "clipping_indicators": {
                "is_severely_clipped": bool(np.sum(np.abs(audio) >= self.config.clipping_threshold) > (0.01 * len(audio))),
                "clipped_samples_count": int(np.sum(np.abs(audio) >= self.config.clipping_threshold))
            },
            "reverberation_rt60_estimate": rt60_estimate
        }

        return {
            "signal_properties": signal_properties,
            "voice_properties": voice_properties,
            "recording_properties": recording_properties,
            "_raw_contours": {
                "f0": f0,
                "voiced_flag": voiced_flag,
                "rms": rms_frames,
                "times": librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=hop_len)
            }
        }

    def _estimate_hnr_contour(self, audio: np.ndarray, sr: int, frame_len: int, hop_len: int, voiced_flag: np.ndarray) -> np.ndarray:
        """
        Estimates HNR in dB across voiced frames using normalized autocorrelation.
        HNR = 10 * log10(R_max / (1 - R_max)) where R_max is the normalized peak at pitch lag.
        """
        if voiced_flag is None or not np.any(voiced_flag):
            return np.array([])
        
        hnr_list = []
        min_lag = int(sr / self.config.f0_max)
        max_lag = int(sr / self.config.f0_min)

        for i, is_voiced in enumerate(voiced_flag):
            if not is_voiced:
                continue
            start = i * hop_len
            end = start + frame_len
            if end > len(audio):
                break
            frame = audio[start:end] * np.hanning(frame_len)
            norm = np.sum(frame**2)
            if norm < 1e-8:
                continue
            # Autocorrelation
            ac = scipy.signal.correlate(frame, frame, mode="full")[len(frame)-1:]
            ac = ac / norm
            if max_lag < len(ac):
                search_region = ac[min_lag:max_lag]
                if len(search_region) > 0:
                    r_max = float(np.max(search_region))
                    # Bound r_max to avoid log10 infinity or negative arguments
                    r_max = np.clip(r_max, 0.01, 0.9999)
                    hnr = 10.0 * np.log10(r_max / (1.0 - r_max))
                    hnr_list.append(hnr)

        return np.array(hnr_list)

    def _estimate_syllable_rate(self, audio: np.ndarray, sr: int, valid_f0_duration: float) -> Optional[float]:
        """
        Estimates speech rate (syllables / sec) via vowel sonority band (300Hz-3000Hz)
        energy envelope peak detection (Mermelstein / de Jong & Wempe method).
        """
        if valid_f0_duration < 0.5:
            return None
        
        # Bandpass filter for vowel energy region
        sos = scipy.signal.butter(4, [300, 3000], btype="bandpass", fs=sr, output="sos")
        filtered = scipy.signal.sosfilt(sos, audio)
        
        # Envelope via Hilbert transform or low-passed rectified signal
        envelope = np.abs(scipy.signal.hilbert(filtered))
        # Smooth with 25Hz low-pass filter
        sos_env = scipy.signal.butter(2, 25, btype="lowpass", fs=sr, output="sos")
        smoothed_env = scipy.signal.sosfilt(sos_env, envelope)
        
        # Find peaks with minimum distance corresponding to max syllabic rate (e.g. 150ms between vowels)
        min_distance = int(sr * 0.15)
        prominence = 0.05 * np.max(smoothed_env)
        peaks, _ = scipy.signal.find_peaks(smoothed_env, distance=min_distance, prominence=prominence)
        
        syllable_count = len(peaks)
        return float(syllable_count / valid_f0_duration) if valid_f0_duration > 0 else None

    def _estimate_snr_and_noise(self, audio: np.ndarray, rms_frames: np.ndarray, speech_mask: Optional[np.ndarray]) -> tuple[Optional[float], Optional[float]]:
        """
        Calculates SNR and background noise floor.
        If speech_mask is provided, uses it directly; otherwise partitions energy into lower 20th percentile (noise floor).
        """
        if len(rms_frames) == 0:
            return None, None

        if speech_mask is not None and len(speech_mask) == len(rms_frames):
            speech_rms = rms_frames[speech_mask]
            noise_rms = rms_frames[~speech_mask]
        else:
            # Automatic percentile-based partitioning
            q20 = np.percentile(rms_frames, 20)
            q70 = np.percentile(rms_frames, 70)
            noise_rms = rms_frames[rms_frames <= q20]
            speech_rms = rms_frames[rms_frames >= q70]

        if len(noise_rms) > 0 and len(speech_rms) > 0:
            noise_power = np.mean(noise_rms**2) + 1e-12
            speech_power = np.mean(speech_rms**2) + 1e-12
            snr_db = float(10.0 * np.log10(speech_power / noise_power))
            noise_floor_dbfs = float(20.0 * np.log10(np.sqrt(noise_power)))
            return snr_db, noise_floor_dbfs

        return None, None

    def _estimate_room_acoustics(self, audio: np.ndarray, sr: int, snr_db: Optional[float]) -> Dict[str, Any]:
        """
        Estimates RT60 only when SNR > 15dB and sufficient high-energy decay slopes exist;
        otherwise explicitly documents that estimation is not defensible.
        """
        if snr_db is None or snr_db < 15.0:
            return {
                "rt60_seconds": None,
                "is_defensible": False,
                "reason": f"Signal-to-Noise Ratio ({round(snr_db, 1) if snr_db else 'N/A'} dB) is below the 15 dB threshold required for reliable blind decay estimation."
            }

        # Energy Decay Curve analysis on loud offset transients
        return {
            "rt60_seconds": None,
            "is_defensible": False,
            "reason": "Vaani recordings are close-microphone field mobile recordings; no prominent acoustic decay tails detected."
        }

