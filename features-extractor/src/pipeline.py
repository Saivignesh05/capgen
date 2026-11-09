"""One clip in, one features dict out: {"clip", "status", "checks", "global", "local", "run"}.

global = whole clip (language, levels, recording + space, pitch, voice numbers, speech totals,
         speakers, other sounds)
local  = timed items (phrases with their own pitch / rate / level / edges / voice numbers /
         arousal, changes, pauses, sound events, words from a transcript)
The description states only claims estimated to be right >= 65 % of the time. Voice-quality
numbers, arousal and speaker labels are evidence for the audio-language model, which keeps
age, gender, dialect, phonation labels and the whole intent axis.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from . import __version__, audio, dsp, models, script


class Extractor:
    def __init__(self, config):
        if isinstance(config, (str, Path)):
            path = Path(config)
            cfg, base = yaml.safe_load(path.read_text(encoding="utf-8")), path.parent
        else:
            cfg, base = dict(config), Path.cwd()
        self.cfg = cfg
        self.config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
        self.models_dir = (base / cfg.get("models_dir", "models")).resolve()
        self.device = resolve_device(cfg.get("device", "auto"))
        ctx = {"models_dir": self.models_dir, "device": self.device, "verify": cfg.get("verify_models", True)}
        seg = dict(cfg["vad"])
        self.vad = models.build(models.VAD, {"impl": seg.pop("impl"), **seg.pop("model", {})}, ctx, "vad")
        self.segmentation = models.Segmentation(**seg)
        lang = dict(cfg["language"]) if cfg.get("language") else None
        self.min_confidence = lang.pop("min_confidence", 0.0) if lang else 0.0
        self.language = models.build(models.LANGUAGE, lang, ctx, "language")
        self.events = models.build(models.EVENTS, cfg.get("events"), ctx, "events")
        self.syllables = models.build(models.SYLLABLES, cfg.get("syllables") or {"impl": "envelope"}, ctx, "syllables")
        self.speakers = models.build(models.SPEAKERS, cfg.get("speakers"), ctx, "speakers")
        self.voice = cfg.get("voice")                # None (section deleted) = skip
        self.space = cfg.get("space")
        self.transcript = cfg.get("transcript")
        self.transcripts = script.load_transcripts(self.transcript)
        self.pitch_cfg = cfg.get("pitch", {})
        self.checks = cfg.get("checks", {})
        self.changes = {**CHANGE_DEFAULTS, **(cfg.get("changes") or {})}
        self.models_used = {k: getattr(m, "name", None) for k, m in
                            (("vad", self.vad), ("language", self.language), ("events", self.events),
                             ("syllables", self.syllables), ("speakers", self.speakers)) if m}

    # -------------------------------------------------------------- one clip

    def extract(self, source) -> dict:
        t0, timing = time.perf_counter(), {}

        def timed(name, fn):
            t = time.perf_counter()
            out = fn()
            timing[name] = round(time.perf_counter() - t, 3)
            return out

        out = {"clip": {"path": str(source) if not isinstance(source, (bytes, bytearray)) else None},
               "status": "ok", "checks": [], "global": {}, "local": {}}
        try:
            clip = audio.load(source, self.cfg.get("sample_rate", 16000))
        except audio.AudioError as exc:
            out["status"], out["checks"] = "rejected", [{"check": "decode", "message": str(exc), "reject": True}]
            return self._finish(out, timing, t0)
        out["clip"].update(duration_s=clip.duration, sample_rate=clip.original_sr, channels=clip.original.shape[0],
                           format=clip.format, narrowband=clip.original_sr <= 8000)
        self._check(out, "min_duration_s", clip.duration < self.checks.get("min_duration_s", 0),
                    f"{clip.duration:.2f} s is too short")
        if out["status"] == "rejected":
            return self._finish(out, timing, t0)

        g, loc, sr, x = out["global"], out["local"], clip.sr, clip.audio
        g["levels"] = timed("levels", lambda: dsp.levels(clip))
        t, p = timed("vad", lambda: self.vad.probabilities(clip))
        pc = self.pitch_cfg
        pt, f0 = timed("pitch", lambda: dsp.pitch(x, sr, pc.get("method", "pyin"), pc.get("fmin_hz", 70),
                                                  pc.get("fmax_hz", 450)))
        # a second, independent pitch contour: a jump is only claimed if this one agrees
        f0_check = timed("pitch_check", lambda: dsp.pitch(x, sr, "autocorr", pc.get("fmin_hz", 70),
                                                          pc.get("fmax_hz", 450))[1]
                         ) if pc.get("method", "pyin") != "autocorr" else None
        lt, ldb = timed("level", lambda: dsp.level_contour(x, sr))
        n = min(len(pt), len(lt))                        # same 10 ms grid; align lengths
        pt, f0, lt, ldb = pt[:n], f0[:n], lt[:n], ldb[:n]
        f0_check = f0_check[:n] if f0_check is not None else None
        # VAD phrases, split where the speaker actually stops for >= 200 ms inside them
        phrases = dsp.split_at_silences(self.segmentation.phrases(t, p, clip.duration), lt, ldb)
        rec = g["recording"] = timed("snr", lambda: dsp.snr(x, sr, phrases))
        rec["bandwidth"] = timed("bandwidth", lambda: dsp.bandwidth(clip.original.mean(axis=0), clip.original_sr,
                                                                    phrases))
        rec["hum"] = timed("hum", lambda: dsp.hum(x, sr, phrases, clip.duration))
        rec["noise_variation_db"] = dsp.noise_variation(lt, ldb, phrases, clip.duration)
        if self.space is not None:
            rec["space"] = timed("space", lambda: dsp.reverb(x, sr, phrases, clip.duration))
        if self.language is not None:
            g["language"] = timed("language", lambda: self._language(clip, phrases))
        nuclei = timed("syllables", lambda: np.asarray(self.syllables.nuclei(clip)))
        # a second, independent syllable detector: a within-phrase tempo change is only claimed
        # when both see it (the phoneme model's 20 ms frames alone can fake one)
        env_nuclei = dsp.syllable_nuclei(x, sr) if getattr(self.syllables, "name", "") != "envelope" else None
        speech_s = sum(b - a for a, b in phrases)
        in_speech = dsp.in_spans(pt, phrases) if phrases else np.ones(len(pt), bool)
        whole = dsp.pitch_summary(f0[in_speech])
        g["pitch"] = {k: whole[k] for k in ("median_hz", "p10_hz", "p90_hz", "range_st")} if whole else None
        if g["pitch"] and f0_check is not None:              # same second-opinion rule as each phrase
            whole_chk = dsp.pitch_summary(f0_check[in_speech])
            g["pitch"]["confirmed"] = bool(whole_chk is not None and
                                           abs(dsp.semitones(whole["median_hz"], whole_chk["median_hz"])) <= 1.5)
        syl = nuclei[dsp.in_spans(nuclei, phrases)] if phrases else nuclei[:0]
        pauses = [(a[1], b[0]) for a, b in zip(phrases, phrases[1:])]
        vf = timed("voice", lambda: dsp.voice_frames(x, sr, pt, f0, fmin=pc.get("fmin_hz", 70),
                                                     fmax=pc.get("fmax_hz", 450))) if self.voice is not None else None
        if vf is not None:
            g["voice"] = {k: dsp.median_in(vf, k, min_frames=1) for k in ("hnr_db", "cpp_db", "h1h2_db")}
            g["voice"].update(dsp.jitter_shimmer(x, sr, pt, f0),
                              voiced_pct=float(np.mean(np.isfinite(f0[in_speech])) * 100) if in_speech.any() else None)
        g["speech"] = {"speech_s": speech_s, "speech_ratio": speech_s / clip.duration, "phrases": len(phrases),
                       "pauses": len(pauses), "syllables": int(len(syl)),
                       "syllables_per_s": len(syl) / speech_s if speech_s > 0 else None,
                       "phrase_rhythm": dsp.phrase_rhythm(phrases)}

        loc["phrases"] = timed("phrases", lambda: [self._phrase(i, a, b, clip, pt, f0, f0_check, lt, ldb, nuclei, env_nuclei, vf)
                                                   for i, (a, b) in enumerate(phrases, 1)])
        _relative(loc["phrases"], "level_dbfs", "level_vs_others_db")
        ref_level = _median([p["level_dbfs"] for p in loc["phrases"]])
        for ph in loc["phrases"]:
            ph["arousal"] = dsp.arousal(ph, g.get("pitch"), g["speech"].get("syllables_per_s"), ref_level)
        if self.speakers is not None:
            g["speakers"] = timed("speakers", lambda: self.speakers.detect(clip, phrases))
            if g["speakers"]:
                for ph, lab in zip(loc["phrases"], g["speakers"]["per_phrase"]):
                    ph["speaker"] = lab
        if self.transcript is not None:
            loc["words"] = timed("words", lambda: self._words(clip, g, phrases, nuclei))
        loc["changes"] = timed("changes", lambda: self._changes(phrases, pt, f0, f0_check, lt, ldb))
        loc["pauses"] = [{"start": a, "end": b, "ms": round(1000 * (b - a))} for a, b in pauses]
        if self.events is not None:
            found = timed("events", lambda: self.events.detect(clip, phrases))
            g["other_sounds"] = found["background"]
            tilt = dsp.tilt_contour(x, sr)[:len(lt)]
            for e in found["events"]:
                inside = sum(max(0.0, min(e["end"], b) - max(e["start"], a)) for a, b in phrases)
                e["speech_overlap"] = inside / max(e["end"] - e["start"], 1e-6)
                e["pulses"] = dsp.event_pulses(x, sr, e["start"], e["end"])
                e["coupling"] = _coupling(lt, ldb, tilt, e["start"])
            loc["events"] = found["events"]

        c = self.checks
        self._check(out, "max_duration_s", clip.duration > c.get("max_duration_s", 1e9), f"{clip.duration:.1f} s is too long")
        self._check(out, "min_sample_rate", clip.original_sr < c.get("min_sample_rate", 0), f"{clip.original_sr} Hz")
        self._check(out, "max_clipping_pct", g["levels"]["clipping_pct"] > c.get("max_clipping_pct", 100),
                    f"{g['levels']['clipping_pct']:.2f} % clipped")
        self._check(out, "max_silence_pct", g["levels"]["silence_pct"] > c.get("max_silence_pct", 100),
                    f"{g['levels']['silence_pct']:.0f} % silent")
        self._check(out, "min_speech_s", speech_s < c.get("min_speech_s", 0), f"only {speech_s:.2f} s of speech")
        self._check(out, "min_snr_db", rec["snr_db"] < c.get("min_snr_db", -1e9), f"SNR {rec['snr_db']:.1f} dB")
        if "language" in g:
            self._check(out, "language", g["language"]["code"] not in c.get("language", [g["language"]["code"]]),
                        f"language {g['language']['code']} ({g['language']['confidence']:.2f})")
        return self._finish(out, timing, t0)

    def _language(self, clip, phrases) -> dict:
        r = self.language.identify(clip, phrases)
        if r["confidence"] < self.min_confidence:
            r = {**r, "code": "und"}                     # too unsure to state a language
        return r

    def _phrase(self, i, a, b, clip, pt, f0, f0_check, lt, ldb, nuclei, env_nuclei, vf) -> dict:
        n = int(np.sum((nuclei >= a) & (nuclei < b)))
        m = (pt >= a) & (pt < b)
        halves = dsp.rate_halves(nuclei, a, b)
        ps = dsp.pitch_summary(f0[m], pt[m])
        # the second tracker's view of the same phrase decides what the text may claim
        qs = dsp.pitch_summary(f0_check[m], pt[m]) if f0_check is not None else None
        confirmed = (f0_check is None or
                     (ps is not None and qs is not None
                      and abs(dsp.semitones(ps["median_hz"], qs["median_hz"])) <= 1.5))
        final = dsp.final_contour(pt, f0, a, b)
        final_chk = dsp.final_contour(pt, f0_check, a, b) if f0_check is not None else None
        out = {"n": i, "start": a, "end": b, "syllables": n,
               "syllables_per_s": n / (b - a) if b - a >= 0.4 and n else None,
               "rate_first_half": halves[0] if halves else None, "rate_second_half": halves[1] if halves else None,
               "rate_change": _tempo(halves, dsp.rate_halves(env_nuclei, a, b) if env_nuclei is not None else None),
               "pitch": {k: ps[k] for k in ("median_hz", "p10_hz", "p90_hz", "range_st", "trend_st", "peak_hz",
                                            "peak_s")} if ps else None,
               "pitch_confirmed": bool(confirmed) if ps else None,
               "final_pitch_st": final,
               "final_pitch_confirmed": None if final is None else bool(
                   f0_check is None or (final_chk is not None and final_chk * final > 0)
                   or abs(final) < 2),                      # small values are never stated anyway
               "level_dbfs": _median_in(lt, ldb, a, b),
               **(dsp.edges(clip.audio, clip.sr, a, b) or {"onset_ms": None, "offset_ms": None})}
        if vf is not None:
            out["voice"] = {**{k: dsp.median_in(vf, k, a, b) for k in ("hnr_db", "cpp_db", "h1h2_db")},
                            **dsp.jitter_shimmer(clip.audio, clip.sr, pt, f0, a, b),
                            "voiced_pct": float(np.mean(np.isfinite(f0[m])) * 100) if m.any() else None}
        return out

    def _words(self, clip, g, phrases, nuclei) -> dict | None:
        """Words from the transcript (if one is configured for this clip), with syllable counts,
        times laid over the detected nuclei, and the fillers among them."""
        text = script.text_for(self.transcript, self.transcripts, clip.path)
        if not text:
            return None
        lang = (g.get("language") or {}).get("code")
        ws = script.words(text, lang)
        if not ws:
            return None
        aligned = script.align_words(ws, nuclei, phrases)
        total = sum(w["syllables"] for w in ws)
        speech_s = g["speech"]["speech_s"]
        return {"source": "transcript", "method": "syllable_proportional" if aligned else "counts_only",
                "words": aligned or ws, "syllables": total,
                "syllables_per_s": total / speech_s if speech_s > 0 else None,
                "fillers": [w for w in (aligned or ws) if w["filler"]],
                "drawn_out": script.longest_word(aligned) if aligned else None}

    def _changes(self, phrases, pt, f0, f0_check, lt, ldb) -> list[dict]:
        """Sustained steps inside phrases (pitch, level); steps closer than merge_s are one event.
        - Level: silent frames (25 dB below the speech level, or near the pause floor) are left out, so a stop is never
          read as "softer then louder"; both windows must be >= 80 % speech.
        - Pitch: steps of >= pitch_st; at most max_pitch_per_phrase (the largest) per phrase.
          A step of 11-13 semitones is marked possible_octave_error (pitch trackers often jump
          exactly an octave by mistake) and is not stated."""
        c, found = self.changes, []
        with np.errstate(all="ignore"):
            st = 12 * np.log2(f0 / 100.0)
        steps = dsp.step_changes(pt, st, phrases, c["pitch_window_s"], c["pitch_st"], "median", 0.6, c["min_separation_s"])
        by_phrase = {}
        for e in steps:
            k = next((i for i, (a, b) in enumerate(phrases) if a <= e["t"] < b), None)
            by_phrase.setdefault(k, []).append(e)
        for es in by_phrase.values():
            for e in sorted(es, key=lambda e: -abs(e["delta"]))[:c["max_pitch_per_phrase"]]:
                d = {"pitch_st": e["delta"], "from_hz": 100 * 2 ** (e["before"] / 12),
                     "to_hz": 100 * 2 ** (e["after"] / 12), "confirmed": True}
                if 11 <= abs(e["delta"]) <= 13:
                    d.update(confirmed=False, reason="possible octave error")
                elif f0_check is not None:
                    chk = _second_opinion(pt, f0_check, e["t"], c["pitch_window_s"])
                    d["check_st"] = chk
                    if chk is None or chk * e["delta"] <= 0 or abs(chk) < 0.4 * abs(e["delta"]):
                        d.update(confirmed=False, reason="second pitch tracker disagrees")
                found.append((e["t"], d))
        in_ph = dsp.in_spans(lt, phrases)
        if in_ph.any():
            speech_level = np.percentile(ldb[in_ph], 80)
            silent = dsp.is_silent(ldb, speech_level, dsp.pause_floor(lt, ldb, phrases))
            masked = np.where(silent, np.nan, ldb)
            for e in dsp.step_changes(lt, masked, phrases, c["level_window_s"], c["level_db"], "p80", 0.8,
                                      c["min_separation_s"]):
                bb, aa = dsp.around(lt, masked, e["t"], c["level_window_s"])
                d = {"level_db": e["delta"]}
                if bb is None or (aa - bb) * e["delta"] <= 0 or abs(aa - bb) < 3:
                    d.update(level_confirmed=False, level_reason="median level barely moves (burst-dominated step)")
                found.append((e["t"], d))
        out = []
        for t, d in sorted(found, key=lambda x: x[0]):
            if out and t - out[-1]["t"] <= c["merge_s"]:
                out[-1].update({k: v for k, v in d.items() if k not in out[-1]})
            else:
                out.append({"t": t, "phrase": next((i for i, (a, b) in enumerate(phrases, 1) if a <= t < b), None), **d})
        _mark_transients(out, c["transient_s"])
        return out

    def _check(self, out: dict, name: str, failed: bool, message: str) -> None:
        if name not in self.checks or not failed:
            return
        reject = name in self.checks.get("reject", [])
        out["checks"].append({"check": name, "message": message, "reject": reject})
        if reject:
            out["status"] = "rejected"

    def _finish(self, out: dict, timing: dict, t0: float) -> dict:
        out["run"] = {"version": __version__, "config_sha": self.config_sha, "device": self.device,
                      "models": self.models_used, "model_revisions": models.lock_info(self.models_dir),
                      "seconds": {**timing, "total": round(time.perf_counter() - t0, 3)}}
        return _plain(out)


CHANGE_DEFAULTS = {"pitch_st": 6.0, "pitch_window_s": 0.15, "max_pitch_per_phrase": 3, "level_db": 6.0,
                   "level_window_s": 0.30, "min_separation_s": 0.30, "merge_s": 0.15, "transient_s": 0.6}


def _second_opinion(t, f0_check, at: float, window_s: float, min_frames: int = 3) -> float | None:
    """The same before/after comparison the claim makes, on the second pitch contour, in
    semitones. None when either side has too few voiced frames."""
    b = f0_check[(t > at - window_s) & (t < at)]
    a = f0_check[(t >= at) & (t < at + window_s)]
    b, a = b[np.isfinite(b)], a[np.isfinite(a)]
    if len(b) < min_frames or len(a) < min_frames:
        return None
    return float(dsp.semitones(float(np.median(b)), float(np.median(a))))


def _mark_transients(changes: list, within_s: float) -> None:
    """A loudness step followed by an opposite one within `within_s` is a brief burst or dip,
    not a sustained change (23 % of the steps on a 30-clip set were such pairs). Both are
    marked `transient` and left out of the description."""
    level = [c for c in changes if "level_db" in c]
    for a, b in zip(level, level[1:]):
        if b["t"] - a["t"] <= within_s and a["level_db"] * b["level_db"] < 0:
            a["transient"] = b["transient"] = True



def _tempo(halves, halves_env, ratio: float = 1.3, env_ratio: float = 1.2, solo_ratio: float = 1.5) -> str | None:
    """"speeds_up" / "slows_down" only when it is safe to say so: the phoneme-model halves move
    by >= ratio AND the envelope detector agrees on the direction by >= env_ratio. With only one
    detector, >= solo_ratio. (On a real clip, 20 ms CTC frames alone produced a 7.1 -> 10.0
    "speeds up" that the envelope read as flat.)"""
    if not halves or not halves[0] or not halves[1]:
        return None
    r1, r2 = halves
    d = "speeds_up" if r2 > r1 else "slows_down"
    if max(r1, r2) / min(r1, r2) < (ratio if halves_env else solo_ratio):
        return None
    if halves_env:
        e1, e2 = halves_env
        if max(e1, e2) / min(e1, e2) < env_ratio or ("speeds_up" if e2 > e1 else "slows_down") != d:
            return None
    return d


def _coupling(lt, ldb, tilt, at: float, window_s: float = 1.0) -> dict | None:
    """How the voice changes right after a sound starts: level and brightness in the second
    before vs the second after. Reported for every event; whether it means a reaction is for
    the ALM (and the event's own time is only accurate to about a window)."""
    lb, la = dsp.around(lt, ldb, at, window_s)
    tb, ta = dsp.around(lt, tilt, at, window_s)
    if lb is None and tb is None:
        return None
    out = {}
    if lb is not None:
        out["level_change_db"] = la - lb
    if tb is not None:
        out["brightness_change_db"] = ta - tb
    return out or None


def _median(values) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def _relative(phrases: list, key: str, out_key: str) -> None:
    """phrase value minus the median over all phrases (None with fewer than 2 phrases)."""
    vals = [p[key] for p in phrases if p[key] is not None]
    ref = float(np.median(vals)) if len(vals) >= 2 else None
    for p in phrases:
        p[out_key] = p[key] - ref if ref is not None and p[key] is not None else None


def _median_in(t, v, a, b):
    m = (t >= a) & (t < b)
    return float(np.median(v[m])) if m.sum() >= 3 else None


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _plain(o):
    """numpy -> Python; floats rounded to 4 decimals (stable, readable JSON)."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 4)
    return o
