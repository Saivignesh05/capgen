"""
export_json.py - Stage 7: Standardized JSON Output Generator.
Serializes audio analysis, predictions, and measurements into a structured JSON schema.
Strictly distinguishes raw signal measurements, model predictions, and human ground truth.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Encodes numpy data types to standard python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class JsonExporter:
    def __init__(self, output_dir: str | Path = None):
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/json")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        clip_id: str,
        metadata: Dict[str, Any],
        global_info: Dict[str, Any],
        speech_segments: List[Dict[str, Any]],
        local_events: List[Dict[str, Any]],
        sound_event_candidates: List[Dict[str, Any]],
        acoustic_changes: List[Dict[str, Any]],
        timeline: List[Dict[str, Any]],
        processing_info: Dict[str, Any],
        description: Dict[str, Any] | None = None,
        word_alignment: List[Dict[str, Any]] | None = None,
        caption_units: List[Dict[str, Any]] | None = None,
        inter_word_pauses: List[Dict[str, Any]] | None = None,
    ) -> Path:
        """
        Creates and writes a standardized JSON file for the audio clip.
        """
        # Ensure deep clean of any numpy objects in global_info
        clean_global = {
            "signal_properties": global_info.get("signal_properties", {}),
            "voice_properties": global_info.get("voice_properties", {}),
            "recording_properties": global_info.get("recording_properties", {})
        }

        output_data = {
            "clip_id": clip_id,
            "metadata": metadata,
            "global_info": clean_global,
            "speech_segments": speech_segments,
            "local_events": local_events,
            "sound_event_candidates": sound_event_candidates,
            "acoustic_changes": acoustic_changes,
            "timeline": timeline,
            "processing_info": processing_info
        }
        if word_alignment is not None:
            output_data["word_alignment"] = word_alignment
        if caption_units is not None:
            output_data["caption_units"] = caption_units
        if inter_word_pauses is not None:
            output_data["inter_word_pauses"] = inter_word_pauses
        if description is not None:
            output_data["description"] = description

        json_path = self.output_dir / f"{clip_id}_analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)

        return json_path
