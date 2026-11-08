"""Build conservative, evidence-grounded Track 2 descriptions."""

from typing import Any, Dict, List


RUBRIC_TYPES = (
    "speaker",
    "source",
    "morphology",
    "pitch",
    "timing",
    "event",
    "channel",
    "space",
    "other_sounds",
    "coupling",
    "addressee",
    "subtext",
    "change",
)


def build_description(
    metadata: Dict[str, Any],
    global_info: Dict[str, Any],
    speech_segments: List[Dict[str, Any]],
    local_events: List[Dict[str, Any]],
    acoustic_changes: List[Dict[str, Any]],
    sound_event_candidates: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Create a compact description and a rubric-aligned evidence inventory."""
    duration = float(metadata.get("duration_seconds", 0.0))
    language = metadata.get("language") or "unknown language"
    voice = global_info.get("voice_properties", {})
    recording = global_info.get("recording_properties", {})
    signal = global_info.get("signal_properties", {})
    vaani = metadata.get("vaani", {})

    median_f0 = voice.get("median_f0_hz")
    speech_rate = voice.get("estimated_speech_rate", {}).get("syllables_per_second")
    snr = recording.get("estimated_snr_db", {}).get("value")
    clipping = recording.get("clipping_indicators", {}).get("is_severely_clipped")
    peak = signal.get("peak_amplitude", {}).get("value")
    rms = signal.get("rms_energy_statistics", {}).get("median")
    centroid = signal.get("spectral_centroid", {}).get("mean_hz")
    bandwidth = signal.get("spectral_bandwidth", {}).get("mean_hz")
    f0_range = voice.get("min_reliable_f0_hz"), voice.get("max_reliable_f0_hz")
    voiced = voice.get("voiced_percentage")
    hnr = voice.get("harmonic_to_noise_ratio_db", {}).get("median_db")
    noise_floor = recording.get("background_noise_floor_dbfs", {}).get("value")
    reverb = recording.get("reverberation_rt60_estimate", {})
    voice_profile = voice.get("voice_profile", {})

    global_sentences = [
        f"This is a {duration:.1f}-second {language} speech recording with "
        f"{len(speech_segments)} detected speech segment(s)."
    ]
    if vaani:
        source_parts = [
            f"Vaani identifies speaker {vaani.get('speakerID', 'as unavailable')}",
            f"recorded in {vaani.get('district', 'an unspecified district')}, {vaani.get('state', 'an unspecified state')}",
        ]
        global_sentences.append("The dataset metadata says " + " and ".join(source_parts) + ".")
        if vaani.get("transcript"):
            global_sentences.append(f"The supplied transcript is: {vaani['transcript']}.")
        else:
            global_sentences.append("The Vaani record has no transcript text for this clip.")
    global_sentences.append("One speaking source is detected; age, gender, dialect, and identity are not inferred from audio.")
    if median_f0 is not None:
        global_sentences.append(f"The median fundamental frequency is about {median_f0:.0f} Hz.")
    if speech_rate is not None:
        global_sentences.append(f"The estimated speech rate is {speech_rate:.1f} syllables per second.")
    if snr is not None:
        global_sentences.append(f"The estimated signal-to-noise ratio is {snr:.1f} dB.")
    if clipping is not None:
        global_sentences.append(
            "No severe clipping is detected." if not clipping else "Severe clipping is detected."
        )
    if f0_range[0] is not None and f0_range[1] is not None:
        global_sentences.append(f"Reliable pitch ranges from about {f0_range[0]:.0f} to {f0_range[1]:.0f} Hz, with {voiced:.1f}% voiced frames.")
    if hnr is not None:
        global_sentences.append(f"The median harmonic-to-noise estimate is {hnr:.1f} dB.")
    if centroid is not None and bandwidth is not None:
        global_sentences.append(f"The mean spectral centroid is {centroid:.0f} Hz with {bandwidth:.0f} Hz bandwidth.")
    if rms is not None and noise_floor is not None:
        global_sentences.append(f"Median frame RMS is {rms:.3f}; the estimated noise floor is {noise_floor:.1f} dBFS.")
    if reverb.get("is_defensible") is False:
        global_sentences.append("Room reverberation is not reported because the available SNR is insufficient for a defensible estimate.")
    if voice_profile.get("voice_quality", {}).get("label") not in (None, "unknown"):
        profile = voice_profile["voice_quality"]
        global_sentences.append(
            f"Acoustic evidence is most consistent with a {profile['label'].replace('_', ' ')} "
            f"voice profile (likelihood {profile['probability']:.2f}); "
            "this is not a perceptual or clinical diagnosis."
        )

    local_sentences: List[str] = []
    local_facts: List[Dict[str, Any]] = []
    for event in local_events:
        confidence = float(event.get("confidence", 0.0))
        certainty = "possible" if confidence < 0.60 else "candidate"
        local_facts.append({
            "kind": "sound_event",
            "label": event.get("event_label", "sound event"),
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "confidence": confidence,
            "status": "observed_candidate",
            "source": event.get("detection_method", "sound_event_detector"),
        })
        local_sentences.append(
            f"Observed candidate: {event.get('event_label', 'sound event')} from "
            f"{event['start_time']:.2f} to {event['end_time']:.2f} seconds "
            f"(probability {confidence:.2f}; source {event.get('detection_method', 'detector')})."
        )

    for event in sound_event_candidates or []:
        confidence = float(event.get("confidence", 0.0))
        local_facts.append({
            "kind": "sound_event",
            "label": event.get("event_label", "sound event"),
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "confidence": confidence,
            "status": "observed_candidate",
            "source": event.get("detection_method", "sound_event_detector"),
        })
        local_sentences.append(
            f"Observed YAMNet candidate: {event.get('event_label', 'sound event')} "
            f"from {event['start_time']:.2f} to {event['end_time']:.2f} seconds "
            f"(probability {confidence:.2f})."
        )

    for change in acoustic_changes:
        local_fact = _change_fact(change)
        local_facts.append(local_fact)
        local_sentences.append(_change_sentence(local_fact))

    speaker_id = metadata.get("speakerID")
    rubric_items = {
        "speaker": _item("needs_human_review", {
            "speaking_sources": len(speech_segments),
            "speaker_id": speaker_id or vaani.get("speakerID"),
            "dataset_metadata": {
                "state": vaani.get("state"),
                "district": vaani.get("district"),
            },
            "identity_inferred": False,
        }),
        "source": _item("observed_measurement", {
            "voice_profile": voice_profile,
            "available_evidence": "HNR, voiced ratio, spectral flatness, centroid, and bandwidth",
        }, "acoustic_heuristic_profile"),
        "morphology": _item("needs_human_review", {}),
        "pitch": _item("supported", {"median_hz": median_f0}, "global_info.voice_properties"),
        "timing": _item("supported", {
            "speech_segments": speech_segments,
            "speech_rate_syllables_per_second": speech_rate,
        }, "VAD and global_info.voice_properties"),
        "event": _item(
            "supported" if local_events else ("observed_candidate" if sound_event_candidates else "unavailable"),
            local_events or sound_event_candidates or [],
            "sound_event_detector",
        ),
        "channel": _item("supported", {
            "sample_rate_hz": metadata.get("analysis_sample_rate"),
            "original_sample_rate_hz": metadata.get("original_sample_rate"),
            "channels": metadata.get("num_channels"),
            "peak_amplitude": peak,
            "rms_amplitude_dbfs": metadata.get("rms_amplitude", {}).get("dbfs"),
            "silence_percentage": metadata.get("silence_percentage"),
            "severe_clipping": clipping,
        }, "metadata and global_info"),
        "space": _item("needs_human_review", {}),
        "other_sounds": _item(
            "supported" if local_events else ("observed_candidate" if sound_event_candidates else "unavailable"),
            local_events or sound_event_candidates or [],
            "sound_event_detector",
        ),
        "coupling": _item("observed_measurement", acoustic_changes, "consolidated_acoustic_features"),
        "addressee": _item("needs_human_review", {}),
        "subtext": _item("needs_human_review", {}),
        "change": _item("observed_measurement", acoustic_changes, "consolidated_acoustic_features"),
    }

    text = " ".join(global_sentences + local_sentences)
    return {
        "description": text,
        "global_description": " ".join(global_sentences),
        "local_description": (
            " ".join(local_sentences)
            if local_sentences
            else "No verified local sound events are available. "
            "Unverified detector candidates are retained separately for audit."
        ),
        "local_facts": local_facts,
        "rubric_items": rubric_items,
        "note": "This report lists observed hypotheses with probabilities. They are useful evidence but are not human-verified facts.",
        "detector_reliability": {
            "sound_events": "observed_with_probability",
            "sound_event_candidates": len(sound_event_candidates or []),
            "acoustic_changes": "measurement_candidates_only",
            "requires_human_audio_audit": True,
        },
    }


def _item(status: str, value: Any, source: str = "") -> Dict[str, Any]:
    return {"status": status, "value": value, "source": source}


def _human_component(component: str) -> str:
    return component.replace("voice_quality_shift_", "voice quality ").replace("_", " ")


def _change_fact(change: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one consolidated detector interval for local reporting."""
    measurements = change.get("measurements", {}).get("component_measurements", [])
    loudness = [
        item["measurements"].get("delta_db")
        for item in measurements
        if item["label"].startswith("loudness_")
        and item["measurements"].get("delta_db") is not None
    ]
    tilt = [
        item["measurements"].get("relative_change")
        for item in measurements
        if item["label"].startswith("voice_quality_shift_")
        and item["measurements"].get("relative_change") is not None
    ]
    labels = [item["label"] for item in measurements]
    return {
        "kind": "acoustic_transition",
        "start_time": change["start_time"],
        "end_time": change["end_time"],
        "duration": change["duration"],
        "components": labels,
        "loudness_delta_db": loudness,
        "spectral_tilt_relative_change": tilt,
        "evidence_count": change.get("evidence_count", 0),
        "confidence": change.get("confidence", 0.0),
        "status": "observed_measurement",
        "source": change.get("detection_method", "acoustic_features"),
    }


def _change_sentence(fact: Dict[str, Any]) -> str:
    """Render measured local evidence without overclaiming a semantic event."""
    labels = fact["components"]
    loudness = fact["loudness_delta_db"]
    tilt = fact["spectral_tilt_relative_change"]
    loudness_text = ""
    if loudness:
        if any(value > 0 for value in loudness) and any(value < 0 for value in loudness):
            loudness_text = "a loudness fluctuation"
        elif loudness[0] > 0:
            loudness_text = f"a loudness increase of about {max(loudness):.1f} dB"
        else:
            loudness_text = f"a loudness decrease of about {abs(min(loudness)):.1f} dB"
    tilt_text = "a spectral-tilt change" if tilt else ""
    pitch_text = ""
    if any(label.startswith("pitch_") for label in labels):
        pitch_directions = sorted({
            label.removeprefix("pitch_")
            for label in labels
            if label.startswith("pitch_")
        })
        pitch_text = "a pitch " + " and ".join(pitch_directions) + " trend"
    evidence = " and ".join(part for part in (loudness_text, pitch_text, tilt_text) if part)
    if not evidence:
        evidence = (
            ", ".join(_human_component(label) for label in labels)
            if labels
            else "an acoustic change with no single dominant feature"
        )
    return (
        f"From {fact['start_time']:.2f} to {fact['end_time']:.2f} seconds, "
        f"the measured voice delivery shows {evidence} "
        f"(probability {fact['confidence']:.2f}, supported by "
        f"{fact['evidence_count']} frames). This describes an observed acoustic "
        "pattern, not a verified emotion or intention."
    )
