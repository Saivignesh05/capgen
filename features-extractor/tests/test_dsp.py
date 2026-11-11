"""Acoustic measures against signals whose true values are known."""

import numpy as np
import pytest
from conftest import SR, harmonic, noise, place, rms, sine, syllables, t_axis, wav_bytes

from src import audio, dsp


def clip_of(x, sr=SR):
    return audio.load(wav_bytes(x, sr, np.float32), sr)


# ------------------------------------------------------------------ levels


def test_levels_peak_rms_and_silence():
    x = np.concatenate([sine(300, 1.0, 0.5), np.zeros(SR, np.float32)])
    lv = dsp.levels(clip_of(x))
    assert lv["peak_dbfs"] == pytest.approx(20 * np.log10(0.5), abs=0.05)
    assert lv["rms_dbfs"] == pytest.approx(20 * np.log10(rms(x)), abs=0.05)
    assert lv["silence_pct"] == pytest.approx(50, abs=3) and lv["clipping_pct"] == 0 and not lv["severe_clipping"]
    assert set(lv) == {"peak_dbfs", "rms_dbfs", "clipping_pct", "severe_clipping", "silence_pct"}


def test_clipping_is_counted():
    x = np.clip(sine(300, 1.0, 2.0), -1, 1)
    lv = dsp.levels(clip_of(x))
    assert lv["clipping_pct"] > 30 and lv["severe_clipping"]


# ------------------------------------------------------------------ pitch


@pytest.mark.parametrize("f0", [90, 140, 220, 330])
def test_autocorr_pitch_is_accurate(f0):
    t, f = dsp.pitch(harmonic(f0, 1.5), SR, "autocorr")
    v = f[np.isfinite(f)]
    assert len(v) > 0.8 * len(f) and np.median(v) == pytest.approx(f0, rel=0.01)
    assert t[1] - t[0] == pytest.approx(0.01)


def test_pyin_pitch_is_accurate():
    pytest.importorskip("librosa")
    _, f = dsp.pitch(harmonic(180, 1.5), SR, "pyin")
    assert np.nanmedian(f) == pytest.approx(180, rel=0.02)


def test_unvoiced_noise_has_little_pitch():
    _, f = dsp.pitch(noise(1.5, -20), SR, "autocorr")
    assert np.mean(np.isfinite(f)) < 0.2


def test_pitch_summary_range_and_trend():
    s = dsp.pitch_summary(np.linspace(100, 200, 300))           # one octave upward glide
    assert s["median_hz"] == pytest.approx(150, rel=0.01)
    assert s["range_st"] == pytest.approx(12 * np.log2(190 / 110), abs=0.2)
    assert s["trend_st"] == pytest.approx(12 * np.log2(183.3 / 116.7), abs=0.5) and s["trend_st"] > 6
    flat = dsp.pitch_summary(np.full(50, 150.0))
    assert flat["range_st"] == 0 and flat["trend_st"] == 0
    assert dsp.pitch_summary(np.array([150.0, np.nan] * 3)) is None   # too few voiced frames


# ------------------------------------------------------------------ loudness and bandwidth


def test_level_contour_tracks_level_on_the_pitch_grid():
    t, lvl = dsp.level_contour(noise(1.0, -20), SR)
    assert np.median(lvl) == pytest.approx(-20, abs=1) and t[1] - t[0] == pytest.approx(0.01)
    tp, _ = dsp.pitch(noise(1.0, -20), SR, "autocorr")
    assert abs(len(tp) - len(t)) <= 1


def lowpass(x, hz):
    from scipy import signal
    return signal.sosfiltfilt(signal.butter(10, hz, fs=SR, output="sos"), x).astype(np.float32)


def test_bandwidth_full_band_and_phone_band():
    full = dsp.bandwidth(noise(3.0, -20), SR, [(0.0, 3.0)])
    assert full["high_hz"] == pytest.approx(8000, abs=200) and not full["band_limited"]
    phone = dsp.bandwidth(lowpass(noise(3.0, -20), 3400), SR, [(0.0, 3.0)])
    assert 3400 <= phone["high_hz"] <= 4300 and phone["band_limited"]      # edge where the filter is 45 dB down


# ------------------------------------------------------------------ changes inside phrases


def test_pitch_step_is_found_once_at_the_right_time():
    x = harmonic(np.where(t_axis(3.0) < 1.7, 120.0, 180.0), 3.0)
    t, f = dsp.pitch(x, SR, "autocorr")
    st = 12 * np.log2(f / 100)
    (e,) = dsp.step_changes(t, st, [(0.0, 3.0)], 0.15, 4.0)
    assert e["t"] == pytest.approx(1.7, abs=0.1) and e["delta"] == pytest.approx(12 * np.log2(1.5), abs=0.3)


def test_syllable_pulsing_is_not_a_loudness_change_but_a_real_step_is():
    x = syllables(4, 4.0)                                      # full-depth 4 Hz pulsing, steady overall
    t, lvl = dsp.level_contour(x, SR)
    assert dsp.step_changes(t, lvl, [(0.0, 4.0)], 0.3, 6.0, "p80") == []
    x[2 * SR:] *= 10 ** (10 / 20)                              # +10 dB from 2.0 s
    t, lvl = dsp.level_contour(x, SR)
    (e,) = dsp.step_changes(t, lvl, [(0.0, 4.0)], 0.3, 6.0, "p80")
    assert e["t"] == pytest.approx(2.0, abs=0.15) and e["delta"] == pytest.approx(10, abs=1.5)


def test_steps_need_both_windows_inside_the_span():
    x = harmonic(np.where(t_axis(3.0) < 1.5, 120.0, 180.0), 3.0)
    t, f = dsp.pitch(x, SR, "autocorr")
    assert dsp.step_changes(t, 12 * np.log2(f / 100), [(0.0, 1.55), (1.6, 3.0)], 0.15, 4.0) == []


# ------------------------------------------------------------------ SNR


@pytest.mark.parametrize("snr", [5, 15, 30])
def test_snr_with_known_speech_spans(snr):
    speech = place([(1.0, 3.0)], 4.0, lambda d: harmonic(150, d, 0.3))
    floor = noise(4.0, 20 * np.log10(rms(speech[SR:3 * SR])) - snr, seed=1)
    r = dsp.snr(speech + floor, SR, [(1.0, 3.0)])
    assert r["snr_method"] == "vad"
    expected = 10 * np.log10(10 ** (snr / 10) + 1)                 # speech frames hold speech + noise
    assert r["snr_db"] == pytest.approx(expected, abs=0.5)


def test_snr_without_phrases_falls_back_to_percentiles():
    assert dsp.snr(noise(2.0, -30), SR, [])["snr_method"] == "percentile"


@pytest.mark.parametrize("rate", [3, 4, 5])
def test_syllable_nuclei_count(rate):
    x = syllables(rate, 4.0)
    assert len(dsp.syllable_nuclei(x, SR)) == pytest.approx(rate * 4, abs=1)


def test_split_at_silences_only_for_real_stops():
    x = noise(4.0, -70)
    x[int(0.5 * SR):int(1.5 * SR)] += harmonic(150, 1.0, 0.3)
    x[int(1.8 * SR):int(2.6 * SR)] += harmonic(150, 0.8, 0.3)                   # 300 ms stop at 1.5-1.8
    x[int(2.72 * SR):int(3.5 * SR)] += harmonic(150, 0.78, 0.3)                 # 120 ms closure at 2.6-2.72
    t, lvl = dsp.level_contour(x, SR)
    out = dsp.split_at_silences([(0.45, 3.55)], t, lvl)
    assert len(out) == 2
    assert out[0][1] == pytest.approx(1.5, abs=0.04) and out[1][0] == pytest.approx(1.8, abs=0.04)
    assert out[1][1] == pytest.approx(3.55)                                     # the short closure stays inside


def test_bandwidth_at_the_native_rate():
    x48 = noise(3.0, -20, sr=48000)
    bw = dsp.bandwidth(x48, 48000, [(0.0, 3.0)])
    assert bw["high_hz"] > 20000 and not bw["band_limited"]


def test_helpers():
    assert dsp.runs([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]
    assert dsp.in_spans(np.array([0.5, 1.5, 2.5]), [(1, 2)]).tolist() == [False, True, False]
    assert dsp.semitones(100, 200) == pytest.approx(12)
    assert dsp.frame(t_axis(1.0), 400, 160).shape == (101, 400)


# ------------------------------------------------------------------ phrase shape and timing


def test_final_contour_sees_a_final_fall_only_at_the_end():
    x = t_axis(2.0)
    f0 = np.where(x < 1.5, 150.0, 150.0 * 2 ** (-4 / 12))       # last 0.5 s four semitones lower
    t, f = dsp.pitch(harmonic(f0, 2.0), SR, "autocorr")
    assert dsp.final_contour(t, f, 0.0, 2.0) == pytest.approx(-4, abs=0.5)
    t, f = dsp.pitch(harmonic(150, 2.0), SR, "autocorr")
    assert abs(dsp.final_contour(t, f, 0.0, 2.0)) < 0.3


def test_rate_halves():
    nuc = np.array([0.1, 0.35, 0.6, 0.85, 1.1, 1.25, 1.4, 1.55, 1.7, 1.85, 2.0])   # 4/s then ~6.7/s
    r1, r2 = dsp.rate_halves(nuc, 0.0, 2.1)
    assert r1 == pytest.approx(4, rel=0.01) and r2 == pytest.approx(1 / 0.15, rel=0.01)
    steady = np.arange(0.1, 2.0, 0.25)                                           # 4/s throughout
    assert dsp.rate_halves(steady, 0.0, 2.0) == pytest.approx((4, 4))
    assert dsp.rate_halves(nuc[:5], 0.0, 1.2) is None                          # too few syllables


def test_phrase_rhythm_regular_and_irregular():
    reg = dsp.phrase_rhythm([(0.0, 0.5), (0.72, 1.2), (1.44, 1.9), (2.16, 2.6), (2.88, 3.3)])
    assert reg["period_s"] == pytest.approx(0.72) and reg["cv"] < 0.01
    assert dsp.phrase_rhythm([(0.0, 0.5), (0.6, 1.0), (2.5, 3.0), (2.9, 3.2)])["cv"] > 0.5
    assert dsp.phrase_rhythm([(0, 1), (2, 3)]) is None


def test_edges_abrupt_versus_gradual():
    x = noise(3.0, -70)
    abrupt = harmonic(150, 1.0, 0.3)
    ramp = np.linspace(0, 1, int(0.3 * SR)) ** 3                              # ~300 ms swell / fade
    gradual = harmonic(150, 1.0, 0.3) * np.concatenate([ramp, np.ones(SR - 2 * len(ramp)), ramp[::-1]])
    y = x.copy()
    y[SR:2 * SR] += abrupt
    e = dsp.edges(y, SR, 1.0, 2.0)
    assert e["onset_ms"] <= 15 and e["offset_ms"] <= 15
    y = x.copy()
    y[SR:2 * SR] += gradual.astype(np.float32)
    g = dsp.edges(y, SR, 1.0, 2.0)
    assert g["onset_ms"] >= 100 and g["offset_ms"] >= 100


# ------------------------------------------------------------------ non-speech


@pytest.mark.parametrize("mains", [50, 60])
def test_mains_hum_with_harmonics(mains):
    y = noise(4.0, -50) + sum(0.01 / h * sine(mains * h, 4.0, 1.0) for h in (1, 2, 3))
    h = dsp.hum(y.astype(np.float32), SR, [], 4.0)
    assert h["hz"] == mains and {1, 2, 3} <= set(h["harmonics"]) and h["prominence_db"] > 20
    assert dsp.hum(noise(4.0, -50), SR, [], 4.0) is None


def test_noise_variation_steady_versus_fluctuating():
    t, lvl = dsp.level_contour(noise(3.0, -50), SR)
    assert dsp.noise_variation(t, lvl, [], 3.0) < 3
    y = noise(3.0, -50) * np.repeat([1.0, 10.0, 1.0, 10.0, 1.0, 10.0], SR // 2).astype(np.float32)
    t, lvl = dsp.level_contour(y, SR)
    assert dsp.noise_variation(t, lvl, [], 3.0) > 15


# ------------------------------------------------------------------ voice numbers (for the ALM)


def test_cpp_and_hnr_fall_with_breath_noise_and_h1h2_follows_the_spectrum():
    clean = harmonic(150, 2.0, 0.4)
    t, f = dsp.pitch(clean, SR, "autocorr")
    vc, vn = dsp.voice_frames(clean, SR, t, f), dsp.voice_frames(clean + noise(2.0, -18), SR, t, f)
    assert np.median(vc["cpp_db"]) > np.median(vn["cpp_db"]) + 5
    assert np.median(vc["hnr_db"]) > np.median(vn["hnr_db"]) + 8
    assert np.median(vc["h1h2_db"]) == pytest.approx(6.0, abs=1.0)        # 1/k harmonics: 20 log10(2)
    x = t_axis(2.0)
    weak_h2 = (0.4 * np.sin(2 * np.pi * 150 * x) + 0.04 * np.sin(2 * np.pi * 300 * x)).astype(np.float32)
    assert np.median(dsp.voice_frames(weak_h2, SR, t, f)["h1h2_db"]) == pytest.approx(20.0, abs=1.0)
    assert dsp.median_in(vc, "cpp_db", 0.5, 1.0) is not None and dsp.median_in(vc, "cpp_db", 5, 6) is None


def test_jitter_and_shimmer_rise_with_irregularity():
    steady = harmonic(150, 2.0, 0.4)
    t, f = dsp.pitch(steady, SR, "autocorr")
    a = dsp.jitter_shimmer(steady, SR, t, f)
    wobble = harmonic(150 + 12 * np.sin(2 * np.pi * 37 * t_axis(2.0)), 2.0, 0.4)   # fast F0 wobble
    tw, fw = dsp.pitch(wobble, SR, "autocorr")
    b = dsp.jitter_shimmer(wobble, SR, tw, fw)
    assert a["jitter_pct"] < b["jitter_pct"] and a["jitter_pct"] < 1.0
    am = (steady * (1 + 0.5 * np.sin(2 * np.pi * 70 * t_axis(2.0)))).astype(np.float32)
    assert dsp.jitter_shimmer(am, SR, t, f)["shimmer_db"] > a["shimmer_db"] + 1
    assert dsp.jitter_shimmer(steady, SR, t, f, 5.0, 6.0) == {"jitter_pct": None, "shimmer_db": None}


# ------------------------------------------------------------------ pulses, coupling, arousal


def test_event_pulses_counts_bursts():
    x = place([(0.1, 0.25), (0.4, 0.55), (0.7, 0.85)], 1.2, lambda d: harmonic(300, d, 0.4),
              floor=noise(1.2, -70))
    assert dsp.event_pulses(x, SR, 0.0, 1.2) == 3
    assert dsp.event_pulses(x, SR, 0.0, 0.15) is None                     # too short to count


def test_around_and_coupling_window():
    t = np.arange(0, 4, 0.01)
    v = np.where(t < 2.0, -30.0, -22.0)
    before, after = dsp.around(t, v, 2.0, 1.0)
    assert (before, after) == (-30.0, -22.0)
    assert dsp.around(t, v, 0.02, 1.0) == (None, None)                    # not enough frames before


def test_arousal_is_relative_to_the_clip():
    ph = {"pitch": {"median_hz": 200.0, "range_st": 8.0}, "level_dbfs": -18.0, "syllables_per_s": 5.0}
    out = dsp.arousal(ph, {"median_hz": 150.0}, 4.0, -22.0)
    assert out["pitch_above_clip_st"] == pytest.approx(12 * np.log2(200 / 150))
    assert out["level_above_clip_db"] == pytest.approx(4.0) and out["rate_ratio"] == pytest.approx(1.25)
    assert dsp.arousal({}, None, None, None) is None


# ------------------------------------------------------------------ space


def reverberate(x, sr, rt60, seed=0):
    """Convolve with an exponentially decaying noise tail of the given RT60."""
    rng = np.random.default_rng(seed)
    n = int(0.8 * sr)
    ir = rng.standard_normal(n) * 10 ** (-3 * np.arange(n) / sr / rt60)
    ir[0] += 6.0
    y = np.convolve(x, ir / np.abs(ir).sum() * 6, "full")[:len(x)]
    return (y / (np.abs(y).max() + 1e-9) * 0.4).astype(np.float32)


def test_reverb_buckets_a_live_room_and_skips_a_dry_one():
    phrases = [(0.5, 1.5), (2.2, 3.2), (4.0, 5.0)]
    speech = place([(a, b) for a, b in phrases], 6.0, lambda d: syllables(4, d, 150, 0.5))
    live = reverberate(speech, SR, 0.8) + noise(6.0, -60)
    out = dsp.reverb(live.astype(np.float32), SR, phrases, 6.0)
    assert out["bucket"] in ("some reverberation", "reverberant") and out["tails"] >= 2
    dry = (speech + noise(6.0, -60)).astype(np.float32)
    d = dsp.reverb(dry, SR, phrases, 6.0)
    assert d is None or d["bucket"] == "dry"
    assert dsp.reverb(dry, SR, [], 6.0) is None
