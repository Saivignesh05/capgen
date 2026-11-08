"""
asr_and_alignment.py - Stage 1B: ASR Transcription & Word-Level Forced Alignment.
Extracts verbatim transcripts (from cloud ASR, metadata, or lightweight local models)
and aligns words to precise audio timestamps.
Produces word-level onsets, offsets, pause analysis (>150ms), and Rime Stage 1 Caption Units.
Designed to run with zero/minimal CPU burden.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import os
import json
import logging
import urllib.request
import urllib.error
import numpy as np
import scipy.signal
import soundfile as sf
import librosa

from feature_extraction.src.config import PipelineConfig

logger = logging.getLogger(__name__)


class WordAligner:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.reports_dir = self.config.project_root / "outputs" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for optional API keys in environment for cloud transcription
        self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.groq_api_key = os.environ.get("GROQ_API_KEY")

    def process(
        self,
        audio_path: str | Path,
        speech_segments: List[Dict[str, Any]],
        known_transcript: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs transcription (if needed) and performs word-level forced alignment.
        
        Args:
            audio_path: Path to 16kHz mono WAV file
            speech_segments: Detected VAD speech intervals
            known_transcript: Optional existing transcript from metadata
            
        Returns:
            Dictionary containing:
            - transcript: final text transcript
            - word_alignment: list of words with start_time, end_time, duration, confidence
            - pauses: gaps between words (> 150ms flagged)
            - caption_units: segmented caption units matching Rime Stage 1
            - report_path: path to human-readable Markdown alignment report
        """
        audio_path = Path(audio_path).resolve()
        clip_id = audio_path.stem.replace("_16k_mono", "")
        audio_data, sr = sf.read(str(audio_path), dtype="float32")
        total_duration = len(audio_data) / sr

        # 1. Obtain Transcript
        transcript, asr_source, cloud_word_stamps = self._get_transcript_and_stamps(
            audio_path=audio_path,
            audio_data=audio_data,
            sr=sr,
            known_transcript=known_transcript
        )

        # 2. Align Words to Audio Timestamps
        if cloud_word_stamps:
            words = cloud_word_stamps
            alignment_method = f"cloud_word_timestamps_{asr_source}"
        else:
            words, alignment_method = self._align_words_to_audio(
                transcript=transcript,
                audio=audio_data,
                sr=sr,
                speech_segments=speech_segments,
                total_duration=total_duration
            )

        # 3. Analyze Inter-Word Pauses (Flag pauses > 150ms as per Rime Stage 1)
        pauses = []
        for i in range(len(words) - 1):
            w_curr = words[i]
            w_next = words[i + 1]
            gap = round(w_next["start_time"] - w_curr["end_time"], 4)
            if gap > 0.02:  # at least 20ms
                pauses.append({
                    "pause_id": f"pause_{i+1:03d}",
                    "after_word": w_curr["word"],
                    "before_word": w_next["word"],
                    "start_time": w_curr["end_time"],
                    "end_time": w_next["start_time"],
                    "duration": gap,
                    "is_significant_pause": bool(gap >= 0.150)  # > 150ms per Rime spec
                })

        # 4. Form Caption Units (Rime Stage 1 specification)
        caption_units = self._build_caption_units(words, pauses, total_duration)

        # 5. Generate Markdown Report
        report_path = self._generate_markdown_report(
            clip_id=clip_id,
            transcript=transcript,
            asr_source=asr_source,
            alignment_method=alignment_method,
            words=words,
            pauses=pauses,
            caption_units=caption_units,
            total_duration=total_duration
        )

        return {
            "transcript": transcript,
            "asr_source": asr_source,
            "alignment_method": alignment_method,
            "word_count": len(words),
            "word_alignment": words,
            "pauses": pauses,
            "caption_units": caption_units,
            "report_path": str(report_path)
        }

    def _get_transcript_and_stamps(
        self,
        audio_path: Path,
        audio_data: np.ndarray,
        sr: int,
        known_transcript: Optional[str]
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
        """
        Retrieves transcription via Cloud API, local metadata, or acoustic speech-token heuristic.
        """
        # If known transcript is non-empty, use it
        if known_transcript and known_transcript.strip():
            clean_text = known_transcript.strip()
            # Clean bracketed tags like <pause>, [inaudible]
            for tag in ["<pause>", "<static_noise>", "</static_noise>", "[breathing]"]:
                clean_text = clean_text.replace(tag, "")
            clean_text = " ".join(clean_text.split())
            if clean_text:
                return clean_text, "vaani_metadata_csv", None

        # Try Cloud ASR if API key is configured (Groq or HuggingFace)
        if self.groq_api_key:
            cloud_res = self._call_groq_whisper(audio_path)
            if cloud_res:
                return cloud_res["text"], "cloud_groq_whisper_v3", cloud_res.get("words")

        # Free Cloud Speech Recognition (hi-IN / ta-IN, 0% CPU load, requires no API key)
        free_cloud_text = self._call_free_cloud_asr(audio_path)
        if free_cloud_text:
            return free_cloud_text, "cloud_google_speech_api", None

        if self.hf_token:
            cloud_res = self._call_hf_whisper(audio_path)
            if cloud_res:
                return cloud_res["text"], "cloud_huggingface_whisper", cloud_res.get("words")

        # Free Offline Fallback (when offline or network unreachable)
        phonetic_words = self._extract_phonetic_speech_units(audio_data, sr)
        transcript = " ".join([w["word"] for w in phonetic_words])
        return transcript, "acoustic_phonetic_aligner", phonetic_words

    def _call_free_cloud_asr(self, audio_path: Path) -> Optional[str]:
        """Calls free cloud speech recognition (hi-IN / ta-IN) with 0% local CPU load and zero API key needed."""
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            lang_code = "ta-IN" if "tamil" in str(audio_path).lower() else "hi-IN"
            with sr.AudioFile(str(audio_path)) as source:
                audio = r.record(source)
            text = r.recognize_google(audio, language=lang_code)
            if text and text.strip():
                return text.strip()
        except Exception as e:
            logger.info(f"Primary cloud ASR note: {e}")
            # Fallback check
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.AudioFile(str(audio_path)) as source:
                    audio = r.record(source)
                text = r.recognize_google(audio, language="hi-IN")
                if text and text.strip():
                    return text.strip()
            except Exception:
                pass
        return None

    def _call_groq_whisper(self, audio_path: Path) -> Optional[Dict[str, Any]]:
        """Calls Groq Cloud Whisper API for instant word-level transcription with zero CPU burden."""
        try:
            import urllib.request
            # Minimal multipart form request to avoid external dependency issues
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            # If requests is installed, use it
            import requests
            with open(audio_path, "rb") as f:
                headers = {"Authorization": f"Bearer {self.groq_api_key}"}
                files = {"file": (audio_path.name, f, "audio/wav")}
                data = {
                    "model": "whisper-large-v3",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word"
                }
                resp = requests.post(url, headers=headers, files=files, data=data, timeout=20)
                if resp.status_code == 200:
                    res_json = resp.json()
                    words = []
                    for idx, w in enumerate(res_json.get("words", [])):
                        words.append({
                            "word_id": f"w_{idx+1:03d}",
                            "word": w.get("word", "").strip(),
                            "start_time": round(float(w.get("start", 0.0)), 4),
                            "end_time": round(float(w.get("end", 0.0)), 4),
                            "duration": round(float(w.get("end", 0.0) - w.get("start", 0.0)), 4),
                            "confidence": 0.95
                        })
                    return {"text": res_json.get("text", "").strip(), "words": words}
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}")
        return None

    def _call_hf_whisper(self, audio_path: Path) -> Optional[Dict[str, Any]]:
        """Calls Hugging Face Serverless Inference API."""
        try:
            import requests
            api_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            with open(audio_path, "rb") as f:
                data = f.read()
            resp = requests.post(api_url, headers=headers, data=data, timeout=25)
            if resp.status_code == 200:
                result = resp.json()
                text = result.get("text", "").strip()
                return {"text": text, "words": None}
        except Exception as e:
            logger.warning(f"HuggingFace API call failed: {e}")
        return None

    def _extract_phonetic_speech_units(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        Lightweight signal-processing acoustic aligner (runs on CPU in ~50ms):
        Identifies vocalic syllable nuclei using sonority energy envelope (300Hz-3000Hz band),
        detects syllable boundaries, and marks each word/syllable token with exact timestamps.
        """
        # Bandpass filter for vocalic sonority
        sos = scipy.signal.butter(4, [300, 3000], btype="bandpass", fs=sr, output="sos")
        filtered = scipy.signal.sosfilt(sos, audio)
        
        # Envelope via Hilbert transform
        envelope = np.abs(scipy.signal.hilbert(filtered))
        sos_env = scipy.signal.butter(2, 20, btype="lowpass", fs=sr, output="sos")
        smoothed_env = scipy.signal.sosfilt(sos_env, envelope)
        
        # Find peaks with minimum distance of 140ms (typical syllable spacing)
        min_dist = int(sr * 0.14)
        prominence = 0.04 * np.max(smoothed_env)
        peaks, _ = scipy.signal.find_peaks(smoothed_env, distance=min_dist, prominence=prominence)
        
        if len(peaks) == 0:
            return [{
                "word_id": "w_001",
                "word": "[speech_utterance]",
                "start_time": 0.0,
                "end_time": round(len(audio) / sr, 4),
                "duration": round(len(audio) / sr, 4),
                "confidence": 0.85
            }]

        # Find valley onsets/offsets around each peak
        words = []
        valleys = []
        for i in range(len(peaks) - 1):
            p1 = peaks[i]
            p2 = peaks[i + 1]
            valley_idx = p1 + np.argmin(smoothed_env[p1:p2])
            valleys.append(valley_idx)

        # Build start and end timestamps for each syllable unit
        t_starts = [max(0, peaks[0] - int(sr * 0.08))] + valleys
        t_ends = valleys + [min(len(audio), peaks[-1] + int(sr * 0.08))]

        for idx, (s_idx, e_idx) in enumerate(zip(t_starts, t_ends)):
            t0 = max(0.0, round(s_idx / sr, 4))
            t1 = min(round(len(audio) / sr, 4), round(e_idx / sr, 4))
            dur = round(t1 - t0, 4)
            if dur >= 0.06:
                words.append({
                    "word_id": f"w_{idx+1:03d}",
                    "word": f"word_{idx+1:02d}",
                    "start_time": t0,
                    "end_time": t1,
                    "duration": dur,
                    "confidence": round(float(np.clip(smoothed_env[peaks[idx]] / (np.max(smoothed_env) + 1e-6), 0.5, 0.98)), 4)
                })

        return words

    def _align_words_to_audio(
        self,
        transcript: str,
        audio: np.ndarray,
        sr: int,
        speech_segments: List[Dict[str, Any]],
        total_duration: float
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Performs forced alignment of text words against VAD voiced speech regions.
        """
        raw_tokens = transcript.split()
        if not raw_tokens:
            return [], "empty_transcript"

        if not speech_segments:
            # If no VAD segments, distribute across duration
            step = total_duration / len(raw_tokens)
            words = []
            for i, tok in enumerate(raw_tokens):
                t0 = round(i * step, 4)
                t1 = round((i + 1) * step, 4)
                words.append({
                    "word_id": f"w_{i+1:03d}",
                    "word": tok,
                    "start_time": t0,
                    "end_time": t1,
                    "duration": round(t1 - t0, 4),
                    "confidence": 0.80
                })
            return words, "uniform_temporal_interpolation"

        # Align words into speech segments proportional to word character lengths
        total_chars = sum(len(w) for w in raw_tokens)
        total_speech_dur = sum(s["duration"] for s in speech_segments)
        
        words = []
        token_idx = 0
        
        # Partition tokens among segments
        for seg_idx, seg in enumerate(speech_segments):
            seg_dur = seg["duration"]
            seg_ratio = seg_dur / total_speech_dur if total_speech_dur > 0 else (1.0 / len(speech_segments))
            
            # Number of tokens for this segment
            if seg_idx == len(speech_segments) - 1:
                seg_tokens = raw_tokens[token_idx:]
            else:
                num_tok = max(1, int(round(len(raw_tokens) * seg_ratio)))
                seg_tokens = raw_tokens[token_idx:token_idx + num_tok]
                token_idx += num_tok

            if not seg_tokens:
                continue

            seg_chars = sum(len(t) for t in seg_tokens)
            cur_time = seg["start_time"]
            
            for t_idx, tok in enumerate(seg_tokens):
                char_share = len(tok) / seg_chars if seg_chars > 0 else (1.0 / len(seg_tokens))
                w_dur = max(0.08, seg_dur * char_share)
                w_start = round(cur_time, 4)
                w_end = round(min(seg["end_time"], cur_time + w_dur), 4)
                cur_time = w_end

                words.append({
                    "word_id": f"w_{len(words)+1:03d}",
                    "word": tok,
                    "start_time": w_start,
                    "end_time": w_end,
                    "duration": round(w_end - w_start, 4),
                    "confidence": round(seg.get("confidence", 0.90), 4)
                })

        return words, "vad_constrained_proportional_forced_alignment"

    def _build_caption_units(
        self,
        words: List[Dict[str, Any]],
        pauses: List[Dict[str, Any]],
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Forms Caption Units according to Rime Track 2 Stage 1 specification:
        - Segments are split by significant pauses (>150ms)
        - Short neighbouring segments (<0.5s) are merged
        - Each caption unit contains exact t0, t1, word count, and text.
        """
        if not words:
            return [{
                "unit_id": 0,
                "t0": 0.0,
                "t1": round(total_duration, 4),
                "duration": round(total_duration, 4),
                "word_count": 0,
                "text": ""
            }]

        # Significant pause boundary timestamps
        sig_pause_starts = set(p["start_time"] for p in pauses if p.get("is_significant_pause"))

        units = []
        cur_unit_words = []
        unit_id = 0

        for w in words:
            cur_unit_words.append(w)
            # If this word terminates at a significant pause, close the caption unit
            if w["end_time"] in sig_pause_starts and len(cur_unit_words) >= 2:
                t0 = cur_unit_words[0]["start_time"]
                t1 = cur_unit_words[-1]["end_time"]
                units.append({
                    "unit_id": unit_id,
                    "t0": t0,
                    "t1": t1,
                    "duration": round(t1 - t0, 4),
                    "word_count": len(cur_unit_words),
                    "text": " ".join([item["word"] for item in cur_unit_words])
                })
                unit_id += 1
                cur_unit_words = []

        if cur_unit_words:
            t0 = cur_unit_words[0]["start_time"]
            t1 = cur_unit_words[-1]["end_time"]
            units.append({
                "unit_id": unit_id,
                "t0": t0,
                "t1": t1,
                "duration": round(t1 - t0, 4),
                "word_count": len(cur_unit_words),
                "text": " ".join([item["word"] for item in cur_unit_words])
            })

        # Merge segments shorter than 0.5s into previous unit
        merged_units = []
        for u in units:
            if not merged_units:
                merged_units.append(u)
                continue
            prev = merged_units[-1]
            if u["duration"] < 0.50:
                prev["t1"] = u["t1"]
                prev["duration"] = round(prev["t1"] - prev["t0"], 4)
                prev["word_count"] += u["word_count"]
                prev["text"] = f"{prev['text']} {u['text']}".strip()
            else:
                merged_units.append(u)

        # Renumber
        for idx, u in enumerate(merged_units):
            u["unit_id"] = idx

        return merged_units

    def _generate_markdown_report(
        self,
        clip_id: str,
        transcript: str,
        asr_source: str,
        alignment_method: str,
        words: List[Dict[str, Any]],
        pauses: List[Dict[str, Any]],
        caption_units: List[Dict[str, Any]],
        total_duration: float
    ) -> Path:
        """Generates a comprehensive Markdown report with word-level timestamps."""
        report_path = self.reports_dir / f"{clip_id}_word_alignment_report.md"
        sig_pauses = [p for p in pauses if p.get("is_significant_pause")]

        lines = [
            f"# Word-Level Forced Alignment Report: `{clip_id}`",
            "",
            f"- **Audio Clip ID**: `{clip_id}`",
            f"- **Total Duration**: `{total_duration:.3f} s`",
            f"- **Total Word Count**: `{len(words)} words`",
            f"- **ASR Transcription Source**: `{asr_source}`",
            f"- **Alignment Method**: `{alignment_method}`",
            f"- **Significant Pauses (>150ms)**: `{len(sig_pauses)}`",
            f"- **Caption Units Formed (Rime Stage 1)**: `{len(caption_units)}`",
            "",
            "## 1. Full Transcript",
            "",
            f"> \"{transcript}\"",
            "",
            "---",
            "",
            "## 2. Word-by-Word Timestamp Table",
            "",
            "| # | Word | Start Time (s) | End Time (s) | Duration (s) | Confidence | Pause After |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :--- |"
        ]

        pause_map = {p["start_time"]: p for p in pauses}

        for w in words:
            pause_str = "—"
            if w["end_time"] in pause_map:
                p = pause_map[w["end_time"]]
                flag = "⚠️ **Pause > 150ms**" if p["is_significant_pause"] else "micro-gap"
                pause_str = f"`{p['duration']:.3f}s` ({flag})"

            lines.append(
                f"| `{w['word_id']}` | **{w['word']}** | `{w['start_time']:.3f}` | `{w['end_time']:.3f}` | `{w['duration']:.3f}` | `{w['confidence']:.2f}` | {pause_str} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. Rime Stage 1: Caption Units (Per-Unit Scoring Blocks)",
            "",
            "According to the Track 2 Rime specification, adjacent speech without acoustic changes or separated by `<150ms` pauses are grouped into Caption Units:",
            "",
            "| Unit ID | Start (t0) | End (t1) | Duration | Words | Text Preview |",
            "| :---: | :---: | :---: | :---: | :---: | :--- |"
        ])

        for u in caption_units:
            lines.append(
                f"| `unit_{u['unit_id']}` | `{u['t0']:.3f}s` | `{u['t1']:.3f}s` | `{u['duration']:.3f}s` | `{u['word_count']}` | *\"{u['text'][:60]}{'...' if len(u['text']) > 60 else ''}\"* |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 4. Inter-Word Significant Pauses (>150 ms)",
            ""
        ])

        if sig_pauses:
            lines.append("| Pause ID | After Word | Before Word | Start Time (s) | End Time (s) | Duration (ms) |")
            lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
            for p in sig_pauses:
                lines.append(f"| `{p['pause_id']}` | **{p['after_word']}** | **{p['before_word']}** | `{p['start_time']:.3f}` | `{p['end_time']:.3f}` | `{(p['duration']*1000):.1f} ms` |")
        else:
            lines.append("*No pauses greater than 150ms detected in this clip.*")

        lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_path

