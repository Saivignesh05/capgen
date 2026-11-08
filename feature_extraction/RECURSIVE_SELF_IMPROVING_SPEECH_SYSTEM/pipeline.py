"""
pipeline.py - Master Pipeline Orchestrator.
Integrates all 9 stages of the audio analysis and local event detection system:
1. Audio preprocessing and metadata
2. Global acoustic information
3. Speech and silence segmentation (VAD)
4. Local sound-event detection (SED)
5. Local vocal-delivery and acoustic changes
6. Unified chronological timeline
7. Standardized JSON output
8. Visualizations
9. Verification and ground-truth generation.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
import logging
import soundfile as sf
import torch

from src.config import PipelineConfig
from src.preprocessing import AudioPreprocessor
from src.global_analysis import GlobalAcousticAnalyzer
from src.vad_segmentation import VoiceActivityDetector
from src.sound_event_detection import SoundEventDetector
from src.acoustic_changes import AcousticChangeDetector
from src.timeline import LocalTimelineGenerator
from src.export_json import JsonExporter
from src.visualizer import AudioVisualizer
from src.description import build_description
from src.voice_profile import estimate_voice_profile
from src.asr_and_alignment import WordAligner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AudioAnalysisPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Check GPU availability and configure execution device
        if self.config.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU execution.")
            self.config.device = "cpu"
            
        logger.info(f"Initializing Audio Analysis Pipeline on device: {self.config.device.upper()}")
        
        # Initialize modular stages
        self.preprocessor = AudioPreprocessor(self.config)
        self.vad = VoiceActivityDetector(self.config)
        self.global_analyzer = GlobalAcousticAnalyzer(self.config)
        self.sed = SoundEventDetector(self.config)
        self.change_detector = AcousticChangeDetector(self.config)
        self.timeline_gen = LocalTimelineGenerator()
        self.exporter = JsonExporter(self.config.output_json_dir)
        self.visualizer = AudioVisualizer(self.config.output_plots_dir)
        self.word_aligner = WordAligner(self.config)

    def process_file(self, audio_path: str | Path, generate_plots: bool = True) -> Dict[str, Any]:
        """
        Processes a single audio file through the complete pipeline.
        """
        start_time = time.time()
        audio_path = Path(audio_path).resolve()
        logger.info(f"===> Processing: {audio_path.name}")

        # Stage 1: Preprocessing & Metadata
        logger.info("Stage 1: Preprocessing audio and extracting metadata...")
        prep_res = self.preprocessor.process(audio_path)
        metadata = prep_res["metadata"]
        analysis_audio = prep_res["analysis_audio"]
        sr = prep_res["sample_rate"]

        # Stage 3: Voice Activity Detection (run early so speech_mask informs global analysis)
        logger.info("Stage 3: Running speech/silence segmentation (VAD)...")
        vad_res = self.vad.process(analysis_audio, sr)
        speech_segments = vad_res["speech_segments"]
        silence_intervals = vad_res["silence_intervals"]

        # Stage 1B: Forced Alignment & Word Timestamps (Rime Stage 1)
        logger.info("Stage 1B: Running Forced Alignment & Word Timestamping...")
        alignment_res = self.word_aligner.process(
            audio_path=metadata["analysis_path"],
            speech_segments=speech_segments,
            known_transcript=metadata.get("transcript")
        )
        word_alignment = alignment_res["word_alignment"]
        caption_units = alignment_res["caption_units"]
        inter_word_pauses = alignment_res["pauses"]

        # Stage 2: Global Acoustic Analysis
        logger.info("Stage 2: Computing global acoustic properties...")
        global_res = self.global_analyzer.analyze(analysis_audio, sr)
        global_res["voice_properties"]["voice_profile"] = estimate_voice_profile(
            global_res["signal_properties"],
            global_res["voice_properties"],
        )
        raw_contours = global_res.pop("_raw_contours", {})

        # Stage 4: Local Sound-Event Detection
        logger.info("Stage 4: Running local sound-event detection (SED)...")
        sed_res = self.sed.detect(analysis_audio, sr)
        sound_events = sed_res["events"]
        sound_event_candidates = sed_res.get("candidate_events", [])
        sed_times = sed_res["time_stamps"]
        sed_probs = sed_res["class_probabilities"]

        # Stage 5: Local Vocal-Delivery & Acoustic Changes
        logger.info("Stage 5: Detecting local vocal-delivery and acoustic transitions...")
        change_res = self.change_detector.detect_changes(analysis_audio, sr, speech_segments)
        acoustic_changes = change_res["acoustic_changes"]

        # Stage 6: Unified Chronological Local Timeline
        logger.info("Stage 6: Assembling chronological multi-track timeline...")
        timeline = self.timeline_gen.build_timeline(speech_segments, sound_events, acoustic_changes)

        # Processing Info
        proc_time = round(time.time() - start_time, 3)
        processing_info = {
            "execution_device": self.config.device,
            "execution_time_seconds": proc_time,
            "sample_rate": sr,
            "raw_acoustic_candidate_count": change_res.get("raw_candidate_count", len(acoustic_changes)),
            "consolidated_acoustic_event_count": change_res.get(
                "consolidated_event_count", len(acoustic_changes)
            ),
            "sound_event_candidates_count": len(sound_event_candidates),
            "models_used": [
                {"stage": "VAD", "model": vad_res.get("model")},
                {"stage": "SED", "model": sed_res.get("model")}
            ],
            "libraries": ["librosa", "scipy", "soundfile", "numpy", "torch"],
            "pipeline_version": "1.0.0"
        }
        description = build_description(
            metadata,
            global_res,
            speech_segments,
            sound_events,
            acoustic_changes,
            sound_event_candidates,
        )

        # Stage 7: Standardized JSON Output
        logger.info("Stage 7: Exporting standardized JSON analysis file...")
        json_path = self.exporter.export(
            clip_id=metadata["clip_id"],
            metadata=metadata,
            global_info=global_res,
            speech_segments=speech_segments,
            local_events=sound_events,
            sound_event_candidates=sound_event_candidates,
            acoustic_changes=acoustic_changes,
            timeline=timeline,
            processing_info=processing_info,
            description=description,
            word_alignment=word_alignment,
            caption_units=caption_units,
            inter_word_pauses=inter_word_pauses,
        )

        # Stage 8: Diagnostic Visualizations
        plot_path = None
        if generate_plots:
            logger.info("Stage 8: Generating diagnostic multi-panel visualization...")
            plot_path = self.visualizer.generate_plots(
                clip_id=metadata["clip_id"],
                audio=analysis_audio,
                sr=sr,
                speech_segments=speech_segments,
                silence_intervals=silence_intervals,
                raw_contours=raw_contours,
                acoustic_changes=acoustic_changes,
                sed_times=sed_times,
                sed_probs=sed_probs,
                timeline=timeline
            )

        logger.info(f"Finished {audio_path.name} in {proc_time}s. JSON: {json_path.name}")

        return {
            "clip_id": metadata["clip_id"],
            "json_path": str(json_path),
            "plot_path": str(plot_path) if plot_path else None,
            "alignment_report_path": alignment_res.get("report_path"),
            "metadata": metadata,
            "global_info": global_res,
            "speech_segments": speech_segments,
            "local_events": sound_events,
            "sound_event_candidates": sound_event_candidates,
            "acoustic_changes": acoustic_changes,
            "timeline": timeline,
            "word_alignment": word_alignment,
            "caption_units": caption_units,
            "inter_word_pauses": inter_word_pauses,
            "processing_info": processing_info,
            "description": description,
        }
