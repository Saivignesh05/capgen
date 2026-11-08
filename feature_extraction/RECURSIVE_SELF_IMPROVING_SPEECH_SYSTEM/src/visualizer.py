"""
visualizer.py - Stage 8: Diagnostic Visualizations.
Renders publication-ready 6-panel visualizations for each processed audio clip:
1. Waveform with speech and silence regions
2. Log-frequency Spectrogram
3. F0 (pitch) contour over time
4. RMS energy over time with marked acoustic change points
5. Sound-event detection class probability curves
6. Final combined multi-track event timeline.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all platforms and threads
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import librosa.display


class AudioVisualizer:
    def __init__(self, output_dir: str | Path = None):
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_plots(
        self,
        clip_id: str,
        audio: np.ndarray,
        sr: int,
        speech_segments: List[Dict[str, Any]],
        silence_intervals: List[Dict[str, Any]],
        raw_contours: Dict[str, Any],
        acoustic_changes: List[Dict[str, Any]],
        sed_times: np.ndarray,
        sed_probs: Dict[str, np.ndarray],
        timeline: List[Dict[str, Any]]
    ) -> Path:
        """
        Creates a synchronized 6-panel visualization figure for the audio clip.
        """
        duration = len(audio) / sr
        fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
        plt.subplots_adjust(hspace=0.35)

        time_axis = np.linspace(0, duration, len(audio))

        # ------------------------------------------------------------------
        # Panel 1: Waveform with Speech vs Silence Spans
        # ------------------------------------------------------------------
        ax1 = axes[0]
        ax1.plot(time_axis, audio, color="#2b5c8f", alpha=0.7, lw=0.6, label="Waveform")
        for seg in speech_segments:
            ax1.axvspan(seg["start_time"], seg["end_time"], color="#4caf50", alpha=0.25, label="Speech" if "Speech" not in [l.get_label() for l in ax1.lines] else "")
        for sil in silence_intervals:
            ax1.axvspan(sil["start_time"], sil["end_time"], color="#f44336", alpha=0.15, label="Silence/Non-Speech" if "Silence/Non-Speech" not in [l.get_label() for l in ax1.lines] else "")
        ax1.set_ylabel("Amplitude", fontsize=9, fontweight="bold")
        ax1.set_title(f"Audio Analysis & Local Event Timeline: {clip_id}", fontsize=13, fontweight="bold", pad=8)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.set_ylim(-1.05, 1.05)

        # ------------------------------------------------------------------
        # Panel 2: Log-frequency Spectrogram
        # ------------------------------------------------------------------
        ax2 = axes[1]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=512, hop_length=160)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, hop_length=160, x_axis="time", y_axis="linear", ax=ax2, cmap="magma")
        ax2.set_ylabel("Freq (Hz)", fontsize=9, fontweight="bold")
        ax2.set_ylim(0, 8000)

        # ------------------------------------------------------------------
        # Panel 3: F0 (Pitch) Contour over Time
        # ------------------------------------------------------------------
        ax3 = axes[2]
        f0 = raw_contours.get("f0")
        f0_times = raw_contours.get("times")
        voiced_flag = raw_contours.get("voiced_flag")
        if f0 is not None and len(f0) > 0:
            valid_mask = (f0 > 0) & (~np.isnan(f0))
            if voiced_flag is not None and len(voiced_flag) == len(f0):
                valid_mask &= voiced_flag
            ax3.scatter(f0_times[valid_mask], f0[valid_mask], color="#9c27b0", s=6, label="Voiced F0 (Hz)")
        ax3.set_ylabel("F0 (Hz)", fontsize=9, fontweight="bold")
        ax3.set_ylim(40, 450)
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend(loc="upper right", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 4: RMS Energy over Time & Acoustic Change Points
        # ------------------------------------------------------------------
        ax4 = axes[3]
        rms = raw_contours.get("rms")
        if rms is not None and len(rms) > 0:
            rms_times = raw_contours.get("times")
            rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6))
            ax4.plot(rms_times, rms_db, color="#ff9800", lw=1.2, label="Short-time RMS (dBFS)")
        
        # Mark acoustic change points
        for ch in acoustic_changes:
            ax4.axvline(ch["start_time"], color="#e91e63", ls=":", lw=1.5)
            ax4.text(ch["start_time"], ax4.get_ylim()[1] if ax4.get_ylim()[1] else -10,
                     f" {ch['event_label']}", rotation=90, fontsize=7, color="#c2185b", verticalalignment="top")
        ax4.set_ylabel("RMS (dBFS)", fontsize=9, fontweight="bold")
        ax4.grid(True, linestyle="--", alpha=0.4)
        ax4.legend(loc="upper right", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 5: Sound-Event Detection Class Probability Curves
        # ------------------------------------------------------------------
        ax5 = axes[4]
        colors = {
            "laughter": "#e91e63",
            "cough": "#d32f2f",
            "breathing": "#009688",
            "sigh": "#795548",
            "music": "#3f51b5",
            "impact_sound": "#ff5722",
            "background_speech": "#2196f3"
        }
        for label, probs in sed_probs.items():
            if np.max(probs) > 0.15:  # Only display curves that show some activity
                c = colors.get(label, "#607d8b")
                ax5.plot(sed_times[:len(probs)], probs, label=label, color=c, lw=1.2)
        ax5.set_ylabel("Probability", fontsize=9, fontweight="bold")
        ax5.set_ylim(-0.05, 1.05)
        ax5.grid(True, linestyle="--", alpha=0.4)
        ax5.legend(loc="upper right", fontsize=7, ncol=3)

        # ------------------------------------------------------------------
        # Panel 6: Final Combined Multi-Track Event Timeline
        # ------------------------------------------------------------------
        ax6 = axes[5]
        # Map event categories to vertical tracks
        track_map = {
            "speech": 3,
            "vocal_event": 2,
            "acoustic_change": 1,
            "environmental_sound": 0
        }
        cat_colors = {
            "speech": "#4caf50",
            "vocal_event": "#e91e63",
            "acoustic_change": "#ff9800",
            "environmental_sound": "#2196f3"
        }

        for item in timeline:
            cat = item.get("event_category", "vocal_event")
            y_track = track_map.get(cat, 1)
            s_t = item["start_time"]
            dur = max(0.08, item["duration"])
            col = cat_colors.get(cat, "#757575")

            rect = patches.Rectangle((s_t, y_track - 0.35), dur, 0.7, linewidth=1, edgecolor="black", facecolor=col, alpha=0.75)
            ax6.add_patch(rect)
            # Label
            label_text = f"{item['event_label']} ({item.get('confidence', 0.5):.2f})"
            ax6.text(s_t + 0.02, y_track, label_text, fontsize=7, va="center", color="white", weight="bold")

        ax6.set_yticks([0, 1, 2, 3])
        ax6.set_yticklabels(["Env Sound", "Acoustic Chg", "Vocal Event", "Speech"], fontsize=8, fontweight="bold")
        ax6.set_ylim(-0.8, 3.8)
        ax6.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
        ax6.set_xlim(0, duration)
        ax6.grid(True, linestyle="--", alpha=0.4)

        plot_path = self.output_dir / f"{clip_id}_analysis.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return plot_path

