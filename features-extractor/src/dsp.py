"""Acoustic measurements (numpy / scipy / librosa). Plain functions, no models.
Only measures behind claims estimated to be right >= 65 % of the time.
Everything returns JSON-ready values; None means "could not be measured"."""

from __future__ import annotations

import math

import numpy as np

# ------------------------------------------------------------------ framing helpers


def frame(x: np.ndarray, n: int, hop: int, center: bool = True) -> np.ndarray:
    """(frames, n) view; center=True: frame i is centred at i*hop (librosa convention)."""
    x = np.asarray(x, np.float64)
    if center:
        x = np.pad(x, n // 2, mode="reflect" if len(x) > n // 2 else "constant")
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    count = 1 + (len(x) - n) // hop
    return np.lib.stride_tricks.as_strided(x, (count, n), (x.strides[0] * hop, x.strides[0]), writeable=False)


def db(x, floor: float = 1e-9):
    return 20 * np.log10(np.maximum(x, floor))


def runs(mask) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of True runs."""
    d = np.diff(np.concatenate([[0], np.asarray(mask, int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def in_spans(t: np.ndarray, spans) -> np.ndarray:
    m = np.zeros(len(t), bool)
    for a, b in spans:
        m |= (t >= a) & (t < b)
    return m


def semitones(a: float, b: float) -> float:
    return 12 * math.log2(b / a)


# ------------------------------------------------------------------ level / clipping


def levels(clip, clipping_threshold: float = 0.999, severe_fraction: float = 0.01,
           silence_dbfs: float = -45.0) -> dict:
    """On the audio as decoded (all channels, native rate)."""
    x, sr = clip.original.astype(np.float64), clip.original_sr
    peak, rms = float(np.max(np.abs(x))), float(np.sqrt(np.mean(x ** 2)))
    clipped = int(np.sum(np.abs(x) >= clipping_threshold))
    fr_db = db(np.sqrt(np.mean(frame(x.mean(axis=0), int(sr * 0.025), int(sr * 0.010), False) ** 2, axis=1)))
    return {"peak_dbfs": float(db(peak)), "rms_dbfs": float(db(rms)),
            "clipping_pct": 100 * clipped / x.size, "severe_clipping": clipped > severe_fraction * x.size,
            "silence_pct": float(np.mean(fr_db < silence_dbfs) * 100)}


# ------------------------------------------------------------------ pitch


def pitch(audio, sr, method: str = "pyin", fmin: float = 70, fmax: float = 450,
          frame_s: float = 0.040, hop_s: float = 0.010) -> tuple[np.ndarray, np.ndarray]:
    """-> (frame times, f0 Hz with NaN where unvoiced). pyin = librosa.pyin (default);
    autocorr = numpy normalised autocorrelation (no librosa, less robust)."""
    n, hop = int(sr * frame_s), int(sr * hop_s)
    if method == "pyin":
        import librosa
        f0, _, _ = librosa.pyin(audio, fmin=fmin, fmax=fmax, sr=sr, frame_length=n, hop_length=hop, fill_na=np.nan)
        return np.arange(len(f0)) * hop / sr, np.asarray(f0, float)
    if method != "autocorr":
        raise ValueError(f"pitch method must be pyin or autocorr, got {method!r}")
    from scipy.ndimage import median_filter
    fr = frame(audio, n, hop)
    fr = (fr - fr.mean(axis=1, keepdims=True)) * np.hanning(n)
    lo, hi = int(sr / fmax), min(n - 2, int(sr / fmin))
    spec = np.fft.rfft(fr, 1 << (2 * n - 1).bit_length(), axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    energy = ac[:, 0] + 1e-12
    region = (ac / energy[:, None])[:, lo:hi + 1]
    k = np.argmax(region, axis=1)
    peak = region[np.arange(len(k)), k]
    lag = (lo + k).astype(float)
    i = np.where((k > 0) & (k < region.shape[1] - 1))[0]          # parabolic refinement
    a, b, c = region[i, k[i] - 1], region[i, k[i]], region[i, k[i] + 1]
    d = a - 2 * b + c
    lag[i] += np.where(np.abs(d) > 1e-12, 0.5 * (a - c) / d, 0.0)
    voiced = (peak >= 0.45) & (10 * np.log10(energy / n) > -50)
    f0 = np.where(voiced, sr / lag, np.nan)
    sm = median_filter(np.nan_to_num(f0), size=3, mode="nearest")
    return np.arange(len(f0)) * hop / sr, np.where(voiced, np.where(sm > 0, sm, f0), np.nan)


def pitch_summary(f0: np.ndarray, t: np.ndarray | None = None, min_frames: int = 8) -> dict | None:
    """Median, 10th-90th percentile range (semitones), start-to-end movement (median of the
    first third vs the last third, semitones), 95th-percentile peak and when it happens."""
    ok = np.isfinite(f0)
    v = f0[ok]
    if len(v) < min_frames:
        return None
    p10, p50, p90, p95 = np.percentile(v, [10, 50, 90, 95])
    k = max(3, len(v) // 3)
    out = {"median_hz": float(p50), "p10_hz": float(p10), "p90_hz": float(p90),
           "range_st": semitones(p10, p90), "trend_st": semitones(float(np.median(v[:k])), float(np.median(v[-k:]))),
           "peak_hz": float(p95)}
    if t is not None:
        out["peak_s"] = float(t[ok][int(np.argmax(np.where(v <= p95, v, -1)))])
    return out


# ------------------------------------------------------------------ loudness contour


def level_contour(audio, sr, frame_s: float = 0.040, hop_s: float = 0.010) -> tuple[np.ndarray, np.ndarray]:
    """-> (frame times, frame level in dBFS), on the same 10 ms grid as the pitch."""
    n, hop = int(sr * frame_s), int(sr * hop_s)
    rms = np.sqrt(np.mean(frame(audio, n, hop) ** 2, axis=1))
    return np.arange(len(rms)) * hop / sr, db(rms)


def bandwidth(audio, sr, phrases, drop_db: float = 45.0) -> dict:
    """Where the speech energy actually is, on the audio at its ORIGINAL sample rate: long-term
    spectrum over the speech frames in 125 Hz bands; the edges are the lowest / highest band
    within drop_db of the strongest. band_limited = nothing above ~4.5 kHz although the file
    could hold more (phone / codec)."""
    n_fft = int(2 ** round(math.log2(0.032 * sr)))
    hop = n_fft // 2
    fr = frame(audio, n_fft, hop)
    t = np.arange(len(fr)) * hop / sr
    m = in_spans(t, phrases) if phrases else np.ones(len(fr), bool)
    if m.sum() < 5:
        m = np.ones(len(fr), bool)
    p = np.mean(np.abs(np.fft.rfft(fr[m] * np.hanning(n_fft), axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(n_fft) * sr
    edges = np.arange(0, sr / 2, 125.0)
    bands = 10 * np.log10(np.array([p[(f >= a) & (f < a + 125)].sum() for a in edges]) + 1e-20)
    on = np.where(bands >= bands.max() - drop_db)[0]
    low, high = float(edges[on[0]]), float(min(edges[on[-1]] + 125, sr / 2))
    return {"low_hz": low, "high_hz": high, "band_limited": high <= 4500 < sr / 2 * 0.95}


# ------------------------------------------------------------------ recording


def snr(audio, sr, phrases, frame_s: float = 0.040, hop_s: float = 0.010) -> dict:
    """Speech power over noise power. Speech/noise frames from the VAD phrases, using only
    frames whose whole window is on one side (so speech does not leak into noise);
    without phrases or noise frames: 70th / 20th energy percentiles."""
    n, hop = int(sr * frame_s), int(sr * hop_s)
    rms = np.sqrt(np.mean(frame(audio, n, hop) ** 2, axis=1))
    t, h = np.arange(len(rms)) * hop / sr, frame_s / 2
    s = n_ = np.zeros(0)
    method = "vad"
    if phrases:
        s = rms[in_spans(t, [(a + h, max(a + h, b - h)) for a, b in phrases])]
        n_ = rms[~in_spans(t, [(max(0, a - h), b + h) for a, b in phrases])]
    if not s.size or not n_.size:
        method = "percentile"
        n_, s = rms[rms <= np.percentile(rms, 20)], rms[rms >= np.percentile(rms, 70)]
    noise, speech = float(np.mean(n_ ** 2)) + 1e-12, float(np.mean(s ** 2)) + 1e-12
    return {"snr_db": 10 * math.log10(speech / noise), "noise_floor_dbfs": 10 * math.log10(noise),
            "snr_method": method}


def step_changes(t, v, spans, window_s: float, threshold: float, stat: str = "median",
                 min_valid: float = 0.6, min_separation_s: float = 0.3) -> list[dict]:
    """Sustained steps inside each span: compare a statistic of the window_s before a
    point with the window_s after it (median, or "p80" = 80th percentile, which follows
    syllable peaks and ignores the dips between them). A step counts only if both windows
    lie inside the span, have >= min_valid finite values, and differ by >= threshold.
    One event per step, placed at the middle of where the difference is largest."""
    import warnings
    from numpy.lib.stride_tricks import sliding_window_view
    if len(t) < 2:
        return []
    dt = t[1] - t[0]
    w = max(2, int(round(window_s / dt)))
    out = []
    for a, b in spans:
        idx = np.where((t >= a) & (t < b))[0]
        if len(idx) < 2 * w + 1:
            continue
        win = sliding_window_view(np.asarray(v, float)[idx], w)
        valid = np.isfinite(win).mean(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = np.nanmedian(win, axis=1) if stat == "median" else np.nanpercentile(win, 80, axis=1)
        k = np.arange(w, len(idx) - w + 1)
        d = st[k] - st[k - w]
        ok = (valid[k] >= min_valid) & (valid[k - w] >= min_valid) & np.isfinite(d) & (np.abs(d) >= threshold)
        found = []
        for r0, r1 in runs(ok):
            seg = np.abs(d[r0:r1 + 1])
            near = np.where(seg >= 0.95 * seg.max())[0]
            j = r0 + near[len(near) // 2]
            found.append((abs(d[j]), j))
        kept = []
        for size, j in sorted(found, reverse=True):
            if all(abs(j - q) * dt >= min_separation_s for _, q in kept):
                kept.append((size, j))
        for _, j in sorted(kept, key=lambda x: x[1]):
            out.append({"t": float(t[idx[k[j]]]), "before": float(st[k[j] - w]), "after": float(st[k[j]]),
                        "delta": float(d[j])})
    return out


def syllable_nuclei(audio, sr, band=(300, 3000), lowpass_hz: float = 40, min_gap_s: float = 0.08,
                    prominence_db: float = 1.0, above_floor_db: float = 10.0) -> np.ndarray:
    """Times of vowel-band envelope peaks (no model; the fallback when the phoneme model is off).
    Peaks on the dB envelope >= 80 ms apart, >= prominence_db above their surroundings and
    >= above_floor_db over the 30th-percentile level. Still undercounts fast speech by ~10-20 %."""
    from scipy import signal
    from scipy.fft import next_fast_len
    sos = signal.butter(4, band, btype="bandpass", fs=sr, output="sos")
    env = np.abs(signal.hilbert(signal.sosfilt(sos, audio), next_fast_len(len(audio)))[:len(audio)])
    env = signal.sosfiltfilt(signal.butter(2, lowpass_hz, btype="lowpass", fs=sr, output="sos"), env)
    if not np.max(env) > 0:
        return np.zeros(0)
    e = 20 * np.log10(np.maximum(env, 1e-6))
    peaks, _ = signal.find_peaks(e, distance=max(1, int(sr * min_gap_s)), prominence=prominence_db,
                                 height=np.percentile(e, 30) + above_floor_db)
    return peaks / sr


def pause_floor(t, level_db, phrases) -> float | None:
    """Typical level when nobody speaks: median frame level outside the phrases (>= 0.3 s of it)."""
    m = ~in_spans(t, phrases) if phrases else np.zeros(len(t), bool)
    return float(np.median(level_db[m])) if m.sum() >= 30 else None


def is_silent(level_db, speech_db: float, floor_db: float | None, drop_db: float = 25.0, near_floor_db: float = 6.0):
    """Silent = drop_db below the speech level, or (in noisy recordings, where a pause is less
    than drop_db down) within near_floor_db of the pause floor."""
    quiet = level_db < speech_db - drop_db
    if floor_db is not None and floor_db + near_floor_db < speech_db - 8:
        quiet |= level_db < floor_db + near_floor_db
    return quiet


def split_at_silences(phrases, t, level_db, min_gap_s: float = 0.20, drop_db: float = 25.0,
                      min_len_s: float = 0.15) -> list[tuple[float, float]]:
    """The VAD sometimes runs through short stops. Inside each phrase, a silent stretch (see
    is_silent; relative to the phrase's speech level, its 80th percentile) of at least min_gap_s
    is a pause: the phrase is split there. Shorter dips (consonant closures, ~50-180 ms) stay inside."""
    dt = t[1] - t[0] if len(t) > 1 else 0.01
    floor = pause_floor(t, level_db, phrases)
    out = []
    for a, b in phrases:
        m = np.where((t >= a) & (t < b))[0]
        if len(m) < 5:
            out.append((a, b))
            continue
        ref = np.percentile(level_db[m], 80)
        quiet = is_silent(level_db[m], ref, floor, drop_db)
        cuts = [(t[m[i]], t[m[j]] + dt) for i, j in runs(quiet)
                if (j - i + 1) * dt >= min_gap_s and i > 0 and j < len(m) - 1]
        start = a
        for c0, c1 in cuts:
            if c0 - start >= min_len_s:
                out.append((float(start), float(c0)))
            start = c1
        if b - start >= min_len_s:
            out.append((float(start), float(b)))
    return out


# ------------------------------------------------------------------ phrase shape and timing


def final_contour(t, f0, a: float, b: float, window_s: float = 0.3) -> float | None:
    """Where the phrase ends in pitch: median of the last window_s of voiced frames minus the
    median of the rest of the phrase, in semitones (negative = final fall, positive = final rise)."""
    m = (t >= a) & (t < b) & np.isfinite(f0)
    tv, fv = t[m], f0[m]
    if len(tv) < 10:
        return None
    last, rest = fv[tv > tv[-1] - window_s], fv[tv <= tv[-1] - window_s]
    if len(last) < 5 or len(rest) < 5:
        return None
    return semitones(float(np.median(rest)), float(np.median(last)))


def rate_halves(nuclei, a: float, b: float, min_intervals: int = 3) -> tuple[float, float] | None:
    """Syllable rate in the first and second half of a phrase, each from the median spacing of
    its syllables (counting syllables per half is off by one at the boundary). Needs
    min_intervals spacings per half."""
    v = nuclei[(nuclei >= a) & (nuclei < b)]
    if len(v) < 2 * min_intervals + 1:
        return None
    gaps_, mids = np.diff(v), (v[:-1] + v[1:]) / 2
    mid = (a + b) / 2
    g1, g2 = gaps_[mids < mid], gaps_[mids >= mid]
    if len(g1) < min_intervals or len(g2) < min_intervals:
        return None
    return float(1 / np.median(g1)), float(1 / np.median(g2))


def phrase_rhythm(phrases) -> dict | None:
    """Spacing of phrase starts (>= 3 phrases): mean period and its variation (CV)."""
    if len(phrases) < 3:
        return None
    d = np.diff([a for a, _ in phrases])
    return {"period_s": float(np.mean(d)), "cv": float(np.std(d) / np.mean(d))}


def edges(audio, sr, a: float, b: float, pad: float = 0.1) -> dict | None:
    """How a phrase starts and ends, from a fine level envelope (10 ms windows, 2.5 ms hop):
    onset_ms = time to rise from (peak - 20 dB, or 3 dB above the noise) to peak - 6 dB;
    offset_ms = time to fall from peak - 6 dB to that low level. Short = abrupt / cut off."""
    n, hop = int(sr * 0.010), int(sr * 0.0025)
    i0, i1 = max(0, int((a - pad) * sr)), min(len(audio), int((b + pad) * sr))
    seg = audio[i0:i1]
    if len(seg) < 4 * n:
        return None
    env = db(np.sqrt(np.mean(frame(seg, n, hop) ** 2, axis=1)), 1e-9)
    t = i0 / sr + np.arange(len(env)) * hop / sr
    inside = (t >= a) & (t < b)
    if inside.sum() < 8:
        return None
    peak = np.percentile(env[inside], 95)
    outside = env[~inside]
    floor = np.percentile(outside, 20) if len(outside) >= 4 else -120.0
    low = max(peak - 20, floor + 3)
    if low >= peak - 8:
        return None                                         # too noisy to see the edges
    above_low, above_hi = np.where(env >= low)[0], np.where(env >= peak - 6)[0]
    if not len(above_low) or not len(above_hi):
        return None
    onset = (above_hi[0] - above_low[0]) * hop / sr * 1000
    offset = (above_low[-1] - above_hi[-1]) * hop / sr * 1000
    return {"onset_ms": float(max(onset, 0)), "offset_ms": float(max(offset, 0))}


# ------------------------------------------------------------------ non-speech: hum, noise


def gaps(phrases, duration: float, min_s: float = 0.1) -> list[tuple[float, float]]:
    out, last = [], 0.0
    for a, b in phrases:
        if a - last >= min_s:
            out.append((last, a))
        last = b
    if duration - last >= min_s:
        out.append((last, duration))
    return out


def hum(audio, sr, phrases, duration, min_prominence_db: float = 10.0) -> dict | None:
    """Mains hum / rumble: narrow peaks at 50 or 60 Hz and their harmonics (1-4) standing
    >= min_prominence_db above the spectrum around them. Uses the non-speech audio when there
    is >= 1.5 s of it (speech harmonics can mimic peaks), else the whole clip with 3 harmonics required."""
    from scipy.signal import welch
    ns = [audio[int(a * sr):int(b * sr)] for a, b in gaps(phrases, duration, 0.2)]
    use_pauses = bool(ns) and sum(map(len, ns)) >= 1.5 * sr
    x = np.concatenate(ns) if use_pauses else np.asarray(audio)
    need = 2 if use_pauses else 3
    if len(x) < sr:
        return None
    f, p = welch(x, sr, nperseg=min(len(x), 16384))
    pdb = 10 * np.log10(p + 1e-20)
    best = None
    for base in (50, 60):
        prom = []
        for h in range(1, 5):
            fc = base * h
            peak = pdb[(f >= fc - 1.5) & (f <= fc + 1.5)]
            ring = pdb[((f >= fc - 15) & (f <= fc - 4)) | ((f >= fc + 4) & (f <= fc + 15))]
            if len(peak) and len(ring):
                prom.append(float(peak.max() - np.median(ring)))
        strong = [i + 1 for i, v in enumerate(prom) if v >= min_prominence_db]
        if len(strong) >= need or (prom and prom[0] >= min_prominence_db + 5):
            cand = {"hz": base, "harmonics": strong, "prominence_db": max(prom)}
            if best is None or cand["prominence_db"] > best["prominence_db"]:
                best = cand
    return best


def noise_variation(t, level_db, phrases, duration) -> float | None:
    """Spread (90th - 10th percentile, dB) of the frame level outside speech: small = steady
    noise (fan, hiss), large = fluctuating (traffic, other sounds). None with < 0.3 s outside speech."""
    m = in_spans(t, gaps(phrases, duration, 0.1)) if phrases else np.ones(len(t), bool)
    if m.sum() < 30:
        return None
    p10, p90 = np.percentile(level_db[m], [10, 90])
    return float(p90 - p10)


# ------------------------------------------------------------------ voice quality (numbers for the ALM)
# Praat (parselmouth) is the usual tool for these, but it is GPL-3.0, so they are implemented
# here in numpy. They are evidence for the ALM, not labels: noise, the vowel and the microphone
# move them as much as the voice does, so the description states only the extremes.


def voice_frames(audio, sr, t, f0, frame_s: float = 0.040, fmin: float = 70, fmax: float = 450) -> dict:
    """Per voiced frame: time, HNR, CPP and H1-H2 (dB).
    HNR: normalised autocorrelation peak r -> 10 log10(r / (1 - r)), divided by the window's own
    autocorrelation (Boersma 1993) so a clean voice is not biased low.
    CPP: height of the cepstral peak at the pitch period above the regression line through the
    cepstrum (lower = breathier / rougher).
    H1-H2: level of the 1st harmonic minus the 2nd, no formant correction (higher = breathier,
    lower or negative = more pressed)."""
    x, n = np.asarray(audio, np.float64), int(sr * frame_s)
    lo, hi = int(sr / fmax), int(sr / fmin)
    empty = {k: np.zeros(0) for k in ("t", "hnr_db", "cpp_db", "h1h2_db")}
    if hi >= n:
        return empty
    ok = np.isfinite(f0)
    st = (t[ok] * sr).astype(int) - n // 2
    keep = (st >= 0) & (st + n <= len(x))
    st, tv, fv = st[keep], t[ok][keep], f0[ok][keep]
    if not len(st):
        return empty
    win = np.hanning(n)
    fr = x[st[:, None] + np.arange(n)] * win
    norm = np.sum(fr ** 2, axis=1)
    good = norm >= 1e-8
    fr, norm, tv, fv = fr[good], norm[good], tv[good], fv[good]
    if not len(fr):
        return empty
    rw = np.correlate(win, win, "full")[n - 1:]
    rw = np.maximum(rw / rw[0], 1e-6)
    nfft = 1 << (2 * n - 1).bit_length()
    sp = np.fft.rfft(fr, nfft, axis=1)
    ac = np.fft.irfft(sp * np.conj(sp), nfft, axis=1)[:, :n] / norm[:, None] / rw
    r = np.clip(np.max(ac[:, lo:hi], axis=1), 0.01, 0.9999)
    hnr = 10 * np.log10(r / (1 - r))
    logp = 10 * np.log10(np.abs(sp) ** 2 + 1e-12)
    ceps = 20 * np.log10(np.abs(np.fft.irfft(logp, nfft, axis=1)[:, :hi + 1]) + 1e-12)
    q = np.arange(lo, hi + 1)
    seg = ceps[:, lo:hi + 1]
    slope, icpt = np.polyfit(q, seg.T, 1)
    k = np.argmax(seg, axis=1)
    cpp = seg[np.arange(len(k)), k] - (slope * q[k] + icpt)
    nf = 8192
    mag = 20 * np.log10(np.abs(np.fft.rfft(fr, nf, axis=1)) + 1e-12)
    fbin = sr / nf
    h1h2 = np.array([mag[i, int(f * 0.9 / fbin):int(f * 1.1 / fbin) + 1].max()
                     - mag[i, int(2 * f * 0.9 / fbin):int(2 * f * 1.1 / fbin) + 1].max() for i, f in enumerate(fv)])
    return {"t": tv, "hnr_db": hnr, "cpp_db": cpp, "h1h2_db": h1h2}


def median_in(vf: dict, key: str, a: float = -1e9, b: float = 1e9, min_frames: int = 5) -> float | None:
    m = (vf["t"] >= a) & (vf["t"] < b)
    return float(np.median(vf[key][m])) if m.sum() >= min_frames else None


PHONATION = {
    # creak: the voice drops low and the pulses get irregular or noisy
    "creak_pitch_st": -3.0, "creak_jitter_pct": 0.3, "creak_hnr_db": -2.0, "creak_cpp_db": -1.5,
    # breath: more energy in the first harmonic, less harmonic structure
    "breathy_h1h2_db": 3.0, "breathy_hnr_db": -2.0, "breathy_cpp_db": -1.5,
    # press/strain: pitch up, first harmonic squeezed, harmonics strong
    "strained_pitch_st": 3.0, "strained_h1h2_db": -3.0, "strained_hnr_db": 1.0,
    # modal: everything close to the clip's own middle
    "modal_pitch_st": 1.5, "modal_h1h2_db": 2.0, "modal_hnr_db": 2.0,
    "min_support": 2,
}


def phonation(voice: dict | None, clip_voice: dict | None, pitch_st: float | None = None,
              cfg: dict | None = None) -> dict | None:
    """How one phrase's voice is produced — clear/modal, creaky, breathy or strained — judged
    against *this clip's own* median, never against published norms.

    Absolute thresholds for phonation do not transfer across languages, microphones or vowels,
    which is why the raw numbers alone were never labelled. A within-clip contrast does
    transfer: the same speaker on the same microphone, one phrase against the rest, which is
    also the form the rubric asks for ("creaky voice on the second phrase").

    Needs `min_support` independent measures pointing the same way, so a single noisy frame
    cannot produce a label. `pitch_st` (the phrase's pitch above the clip median) is only
    passed in when the second pitch tracker confirmed it — creak itself breaks pitch tracking,
    so an unconfirmed contour must not be used as evidence for creak.

    Returns {label, confirmed, support, deltas} or None when too little is measurable.
    """
    c = {**PHONATION, **(cfg or {})}
    if not voice or not clip_voice:
        return None

    def d(key):
        x, y = voice.get(key), clip_voice.get(key)
        return None if x is None or y is None else x - y

    deltas = {"pitch_st": pitch_st, "hnr_db": d("hnr_db"), "cpp_db": d("cpp_db"),
              "h1h2_db": d("h1h2_db"), "jitter_pct": d("jitter_pct")}
    if sum(v is not None for v in deltas.values()) < c["min_support"]:
        return None

    def le(key, lim):                                  # at or below a (negative) limit
        v = deltas[key]
        return v is not None and v <= lim

    def ge(key, lim):
        v = deltas[key]
        return v is not None and v >= lim

    def near(key, lim):
        v = deltas[key]
        return v is not None and abs(v) <= lim

    def phrase(key, lim, word):
        v = deltas[key]
        unit = {"pitch_st": "semitones", "hnr_db": "dB", "cpp_db": "dB", "h1h2_db": "dB",
                "jitter_pct": "points"}[key]
        return f"{key.rsplit('_', 1)[0].replace('_', ' ')} {abs(v):.1f} {unit} {word} the clip"

    # Low HNR and low CPP mean "less clean harmonics", which creak and breath share, so
    # counting votes alone confuses the two. Each label needs a DISCRIMINATIVE primary signal
    # -- creak: low pitch or irregular pulses; breath: a strong first harmonic; press: a
    # squeezed one -- plus at least one corroborating measure.
    low_pitch = lambda: le("pitch_st", c["creak_pitch_st"])                        # noqa: E731
    high_jit = lambda: ge("jitter_pct", c["creak_jitter_pct"])                      # noqa: E731
    tests = {
        "creaky": (
            [(low_pitch(), lambda: phrase("pitch_st", 0, "below")),
             (high_jit(), lambda: phrase("jitter_pct", 0, "above"))],
            [(le("hnr_db", c["creak_hnr_db"]), lambda: phrase("hnr_db", 0, "below")),
             (le("cpp_db", c["creak_cpp_db"]), lambda: phrase("cpp_db", 0, "below")),
             (low_pitch(), lambda: phrase("pitch_st", 0, "below")),
             (high_jit(), lambda: phrase("jitter_pct", 0, "above"))],
            not ge("h1h2_db", c["breathy_h1h2_db"]),          # a strong H1 means breath, not creak
        ),
        "breathy": (
            [(ge("h1h2_db", c["breathy_h1h2_db"]), lambda: phrase("h1h2_db", 0, "above"))],
            [(le("hnr_db", c["breathy_hnr_db"]), lambda: phrase("hnr_db", 0, "below")),
             (le("cpp_db", c["breathy_cpp_db"]), lambda: phrase("cpp_db", 0, "below"))],
            not (low_pitch() and high_jit()),                  # low and irregular is creak
        ),
        "strained": (
            [(le("h1h2_db", c["strained_h1h2_db"]), lambda: phrase("h1h2_db", 0, "below"))],
            [(ge("pitch_st", c["strained_pitch_st"]), lambda: phrase("pitch_st", 0, "above")),
             (ge("hnr_db", c["strained_hnr_db"]), lambda: phrase("hnr_db", 0, "above"))],
            True,
        ),
    }
    for label, (primary, corroborate, allowed) in tests.items():
        if not allowed or not any(ok for ok, _ in primary):
            continue
        support, seen = [], set()
        for ok, why in primary + corroborate:
            if ok and (t := why()) not in seen:
                seen.add(t)
                support.append(t)
        if len(support) >= c["min_support"]:
            return {"label": label, "confirmed": True, "support": support, "deltas": deltas}

    # nothing stands out: clear/modal, but only when the measures really are close to the middle
    if (near("h1h2_db", c["modal_h1h2_db"]) and near("hnr_db", c["modal_hnr_db"])
            and (deltas["pitch_st"] is None or near("pitch_st", c["modal_pitch_st"]))):
        return {"label": "modal", "confirmed": True,
                "support": ["every voice measure within the clip's middle"], "deltas": deltas}
    return {"label": None, "confirmed": False,
            "support": [], "reason": "voice measures mixed", "deltas": deltas}


def jitter_shimmer(audio, sr, t, f0, a: float = -1e9, b: float = 1e9, min_periods: int = 12) -> dict:
    """Cycle-to-cycle irregularity over one span, from the pitch contour:
    jitter_pct = mean |T(i) - T(i-1)| / mean T, on runs of consecutive voiced frames;
    shimmer_db = mean |20 log10(A(i)/A(i-1))| of the peak amplitude of one period each.
    Praat's local jitter/shimmer use its own period marks, so values differ a little; use
    them for comparison between phrases of the same clip, not against published norms."""
    x = np.asarray(audio, np.float64)
    m = (t >= a) & (t < b) & np.isfinite(f0)
    if m.sum() < min_periods:
        return {"jitter_pct": None, "shimmer_db": None}
    per, amp = [], []
    for i0, i1 in runs(m):
        for i in range(i0, i1 + 1):
            T = 1.0 / f0[i]
            s = int(t[i] * sr)
            if s + int(T * sr) >= len(x) or T <= 0:
                continue
            per.append(T)
            amp.append(float(np.max(np.abs(x[s:s + int(T * sr)])) + 1e-12))
    if len(per) < min_periods:
        return {"jitter_pct": None, "shimmer_db": None}
    per, amp = np.array(per), np.array(amp)
    jit = float(np.mean(np.abs(np.diff(per))) / np.mean(per) * 100)
    shim = float(np.mean(np.abs(20 * np.log10(amp[1:] / amp[:-1]))))
    return {"jitter_pct": jit, "shimmer_db": shim}


# ------------------------------------------------------------------ brightness, coupling, arousal


def tilt_contour(audio, sr, frame_s: float = 0.040, hop_s: float = 0.010) -> np.ndarray:
    """Brightness per frame: energy above 2 kHz over energy below 1 kHz, in dB."""
    n, hop = int(sr * frame_s), int(sr * hop_s)
    mag = np.abs(np.fft.rfft(frame(audio, n, hop) * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n) * sr
    return 10 * np.log10((mag[:, f > 2000].sum(axis=1) + 1e-9) / (mag[:, f < 1000].sum(axis=1) + 1e-9))


def event_pulses(audio, sr, a: float, b: float, min_gap_s: float = 0.08, prominence_db: float = 4.0) -> int | None:
    """Pulses inside a sound event: peaks of its own envelope ("three laugh pulses").
    None for events shorter than 0.2 s, where one pulse cannot be told from the onset."""
    if b - a < 0.2:
        return None
    seg = np.asarray(audio)[int(a * sr):int(b * sr)]
    if len(seg) < int(0.05 * sr):
        return None
    from scipy import signal
    k = max(1, int(sr * 0.020))                 # 20 ms: one pulse per burst, not per pitch period
    env = db(np.sqrt(np.convolve(seg.astype(np.float64) ** 2, np.ones(k) / k, "same")), 1e-9)
    peaks, _ = signal.find_peaks(env, distance=max(1, int(sr * min_gap_s)), prominence=prominence_db)
    return int(len(peaks)) or None


def around(t, v, at: float, window_s: float, min_frames: int = 8):
    """Medians of v in [at - window_s, at) and [at, at + window_s), ignoring NaN."""
    import warnings
    before, after = (t >= at - window_s) & (t < at), (t >= at) & (t < at + window_s)
    if min(before.sum(), after.sum()) < min_frames:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.nanmedian(v[before])), float(np.nanmedian(v[after]))


def arousal(phrase: dict, clip_pitch: dict | None, clip_rate: float | None, ref_level: float | None) -> dict | None:
    """How animated a phrase is relative to the rest of the clip: pitch (semitones above the
    clip median), range, loudness and rate. Evidence for the ALM's intent lines, not a label."""
    p = phrase.get("pitch")
    out = {}
    if p and clip_pitch and clip_pitch.get("median_hz"):
        out["pitch_above_clip_st"] = semitones(clip_pitch["median_hz"], p["median_hz"])
        out["range_st"] = p["range_st"]
    if phrase.get("level_dbfs") is not None and ref_level is not None:
        out["level_above_clip_db"] = phrase["level_dbfs"] - ref_level
    if phrase.get("syllables_per_s") and clip_rate:
        out["rate_ratio"] = phrase["syllables_per_s"] / clip_rate
    return out or None


# ------------------------------------------------------------------ space (reverberation)


def reverb(audio, sr, phrases, duration: float, min_tail_s: float = 0.25, min_drop_db: float = 20.0) -> dict | None:
    """How live the room sounds, from how fast the level falls after the speaker stops.

    After each phrase end, the level is followed until it is within 3 dB of the pause floor;
    the straight-line slope of that fall gives a decay time (seconds for 60 dB). The median
    over the phrase ends becomes one of three buckets, and nothing finer: a blind estimate
    from speech cannot support more. None when no tail is long or deep enough, which is the
    usual case for close-microphone recordings in quiet rooms."""
    t, l = level_contour(audio, sr)
    floor = pause_floor(t, l, phrases)
    if floor is None or not phrases:
        return None
    times = []
    for a, b in phrases:
        m = (t >= b) & (t < b + 1.0)
        if m.sum() < 10:
            continue
        seg, ts = l[m], t[m]
        start = seg[:3].max()
        if start - floor < min_drop_db:
            continue
        done = np.where(seg <= floor + 3)[0]
        if not len(done) or ts[done[0]] - b < 0.05:
            continue
        k = done[0] + 1
        slope = np.polyfit(ts[:k] - b, seg[:k], 1)[0]            # dB per second (negative)
        if slope < -5:
            times.append(60.0 / -slope)
    if len(times) < 2:
        return None
    rt = float(np.median(times))
    bucket = "dry" if rt < 0.3 else "some reverberation" if rt < 0.6 else "reverberant"
    return {"decay_s": rt, "bucket": bucket, "tails": len(times)}
