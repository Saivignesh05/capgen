"""CPU-only voice profile estimates from existing acoustic measurements."""

from typing import Any, Dict


def estimate_voice_profile(
    signal_properties: Dict[str, Any],
    voice_properties: Dict[str, Any],
) -> Dict[str, Any]:
    """Estimate voice-quality likelihoods without inferring identity or gender."""
    hnr = voice_properties.get("harmonic_to_noise_ratio_db", {}).get("median_db")
    voiced = voice_properties.get("voiced_percentage")
    flatness = signal_properties.get("spectral_flatness", {}).get("mean")
    centroid = signal_properties.get("spectral_centroid", {}).get("mean_hz")
    bandwidth = signal_properties.get("spectral_bandwidth", {}).get("mean_hz")

    evidence = {
        "median_hnr_db": hnr,
        "voiced_percentage": voiced,
        "spectral_flatness": flatness,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
    }

    if hnr is None or voiced is None or flatness is None:
        return {
            "gender": {"label": "unknown", "probability": 0.0, "reason": "insufficient evidence"},
            "voice_quality": {"label": "unknown", "probability": 0.0},
            "likelihoods": {},
            "evidence": evidence,
            "method": "acoustic_heuristic_profile",
            "status": "insufficient_evidence",
        }

    breathy = _clamp((8.0 - hnr) / 8.0 * 0.55 + flatness * 0.45)
    hoarse = _clamp((10.0 - hnr) / 10.0 * 0.65 + (1.0 - voiced / 100.0) * 0.35)
    strained = _clamp(
        (hnr - 5.0) / 12.0 * 0.45
        + (1.0 - min(centroid or 0.0, 4000.0) / 4000.0) * 0.25
        + min(max(bandwidth or 0.0, 0.0) / 3000.0, 1.0) * 0.30
    )
    modal = _clamp(
        (1.0 - abs(hnr - 12.0) / 12.0) * 0.45
        + (voiced / 100.0) * 0.35
        + (1.0 - min(flatness, 1.0)) * 0.20
    )
    likelihoods = {
        "modal_like": round(modal, 3),
        "breathy_like": round(breathy, 3),
        "hoarse_or_rough_like": round(hoarse, 3),
        "strained_like": round(strained, 3),
    }
    label, probability = max(likelihoods.items(), key=lambda item: item[1])
    return {
        "gender": {
            "label": "unknown",
            "probability": 0.0,
            "reason": "gender is not inferred from pitch or voice quality",
        },
        "voice_quality": {
            "label": label,
            "probability": probability,
        },
        "likelihoods": likelihoods,
        "evidence": evidence,
        "method": "acoustic_heuristic_profile",
        "status": "observed_likelihood",
        "warning": "These are acoustic likelihoods, not clinical or perceptual labels.",
    }


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)
