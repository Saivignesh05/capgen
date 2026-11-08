"""
timeline.py - Stage 6: Unified Chronological Local Timeline.
Fuses VAD speech intervals, local sound-event detections (vocal & environmental),
and acoustic change points into an ordered, multi-track timeline without losing overlapping events.
"""

from typing import Dict, Any, List


class LocalTimelineGenerator:
    def __init__(self):
        pass

    def build_timeline(
        self,
        speech_segments: List[Dict[str, Any]],
        sound_events: List[Dict[str, Any]],
        acoustic_changes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merges heterogeneous events into a single sorted chronological list.
        Each event retains its distinct category and properties.
        """
        timeline = []
        uid = 1

        # 1. Add Speech Segments
        for seg in speech_segments:
            timeline.append({
                "timeline_id": f"TL_{uid:04d}",
                "event_id": seg.get("segment_id", f"speech_{uid}"),
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "duration": seg["duration"],
                "event_category": "speech",
                "event_label": "speech_segment",
                "detection_method": seg.get("detection_method", "vad"),
                "confidence": seg.get("confidence", 1.0),
                "timestamp_uncertainty_seconds": 0.03,
                "relevant_acoustic_measurements": {},
                "verification_status": "unverified"
            })
            uid += 1

        # 2. Add Local Sound Events (vocal events, environmental sounds, impacts)
        for ev in sound_events:
            timeline.append({
                "timeline_id": f"TL_{uid:04d}",
                "event_id": ev.get("event_id", f"sed_{uid}"),
                "start_time": ev["start_time"],
                "end_time": ev["end_time"],
                "duration": ev["duration"],
                "event_category": ev.get("event_category", "sound_event"),
                "event_label": ev.get("event_label", "unknown"),
                "detection_method": ev.get("detection_method", "yamnet"),
                "confidence": ev.get("confidence", 0.5),
                "timestamp_uncertainty_seconds": ev.get("timestamp_uncertainty_seconds", 0.12),
                "relevant_acoustic_measurements": {
                    "max_confidence": ev.get("max_confidence", ev.get("confidence"))
                },
                "verification_status": ev.get("verification_status", "unverified")
            })
            uid += 1

        # 3. Add Acoustic Change Points
        for ch in acoustic_changes:
            timeline.append({
                "timeline_id": f"TL_{uid:04d}",
                "event_id": ch.get("change_id", f"change_{uid}"),
                "start_time": ch["start_time"],
                "end_time": ch["end_time"],
                "duration": ch["duration"],
                "event_category": ch.get("event_category", "acoustic_change"),
                "event_label": ch.get("event_label", "acoustic_transition"),
                "detection_method": ch.get("detection_method", "frame_derivative"),
                "confidence": ch.get("confidence", 0.5),
                "timestamp_uncertainty_seconds": ch.get("timestamp_uncertainty_seconds", 0.02),
                "relevant_acoustic_measurements": ch.get("measurements", {}),
                "verification_status": ch.get("verification_status", "unverified")
            })
            uid += 1

        # Sort strictly by start_time, then end_time
        timeline.sort(key=lambda x: (x["start_time"], x["end_time"]))

        # Renumber timeline IDs sequentially
        for idx, item in enumerate(timeline):
            item["timeline_id"] = f"TL_{idx+1:04d}"

        return timeline

