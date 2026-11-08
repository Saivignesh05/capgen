"""
config.py - Central configuration for the audio analysis and local event detection pipeline.
All thresholds, parameters, and paths are defined here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PipelineConfig:
    # Audio standard
    target_sample_rate: int = 16000
    mono: bool = True
    
    # Execution device: 'cpu' or 'cuda'
    device: str = "cpu"
    allow_heuristic_sed: bool = False
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models")
    output_audio_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs" / "analysis_audio")
    output_json_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs" / "json")
    output_plots_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs" / "plots")
    
    # Preprocessing thresholds
    clipping_threshold: float = 0.999
    silence_db_threshold: float = -45.0  # dBFS considered silence in preprocessing
    
    # Global acoustic analysis parameters
    f0_min: float = 70.0   # Hz - raised from 50Hz to avoid pYIN frame-length warnings and improve reliability
    f0_max: float = 450.0  # Hz
    frame_length_sec: float = 0.040  # 40 ms analysis window (must fit at least 2 periods of f0_min)
    hop_length_sec: float = 0.010    # 10 ms hop step
    
    # VAD parameters (Silero VAD)
    vad_threshold: float = 0.50
    vad_neg_threshold: float = 0.35
    vad_min_speech_duration_ms: int = 150  # discard speech segments shorter than this
    vad_min_silence_duration_ms: int = 200 # pause duration to trigger segment split
    vad_speech_pad_ms: int = 50            # padding added to speech boundaries
    
    # Sound Event Detection (YAMNet) parameters
    yamnet_window_size: float = 0.96  # seconds
    yamnet_hop_size: float = 0.24     # seconds (50% or 75% overlap for fine timestamps)
    yamnet_min_event_duration: float = 0.20  # seconds
    yamnet_merge_gap_sec: float = 0.25       # merge consecutive detections if closer than this
    
    # Target candidate event classes and their specific probability thresholds
    event_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "laughter": 0.30,
        "cough": 0.30,
        "sigh": 0.25,
        "breathing": 0.25,
        "crying": 0.30,
        "music": 0.50,             # Raised: avoid false positives on tonal speech
        "background_speech": 0.40, # Raised: only activate if genuine secondary speaker context
        "animal": 0.30,
        "impact_sound": 0.35
    })
    
    # Acoustic change detection parameters
    pitch_change_semitones: float = 5.0      # Raised: Sudden pitch jump threshold (semitones)
    pitch_change_hz: float = 60.0            # Raised: Alternative Hz jump threshold
    loudness_change_db: float = 8.0          # Raised: Sudden loudness jump (dB) - avoids over-detection
    spectral_tilt_change_ratio: float = 0.60 # Raised: Voice quality tilt shift
    change_detection_min_gap_sec: float = 0.50 # Raised: Minimum 500ms between same-type change detections
    change_merge_gap_sec: float = 0.20 # Merge adjacent detections of the same change type
    change_max_duration_sec: float = 0.80 # Prevent one long run becoming an unbounded event
    delivery_merge_gap_sec: float = 0.10 # Group co-occurring feature changes into one local event
