"""Features dict -> tool description: a global paragraph, then one timed line per phrase,
pause, change and sound event (in time order).

Only claims estimated to be right >= 65 % of the time: timing, pitch, loudness, phrase edges,
recording quality, room bucket, speaker count/changes, the sound-event model's labels (with
their probabilities) and, when a transcript is configured, syllable counts and fillers.
Voice-quality numbers and arousal go in as measurements, not labels; age, gender, dialect,
phonation labels and intent are left to the audio-language model. Works on saved features
JSON, so the wording can change without re-running the models."""

from __future__ import annotations

LANGUAGES = {"hi": "Hindi", "ta": "Tamil"}
SOUNDS = {"impact": "knock or bang", "breathing": "audible breathing"}


def span(a: float, b: float) -> str:
    return f"{a:.2f}–{b:.2f} s"


def sound(label: str) -> str:
    return SOUNDS.get(label, label.replace("_", " "))


def kept_events(f: dict) -> list:
    """Sound events stated in the text ("other voices" never: speakers are for the ALM)."""
    return [e for e in f.get("local", {}).get("events", []) if e["label"] != "background_speech"]


def describe(f: dict) -> str:
    return "\n".join(x for x in (global_text(f), local_text(f)) if x)


def global_text(f: dict) -> str:
    c, g = f["clip"], f.get("global", {})
    out = []
    lang = (g.get("language") or {}).get("code")
    out.append(f"{LANGUAGES[lang] + ', ' if lang in LANGUAGES else ''}{c['duration_s']:.1f} s.")

    rec = [f"{c['sample_rate'] / 1000:g} kHz {'mono' if c['channels'] == 1 else str(c['channels']) + '-channel'} file"
           + (" (narrowband: nothing above 4 kHz)" if c.get("narrowband") else "")]
    r = g.get("recording") or {}
    if r.get("snr_db") is not None:
        s = r["snr_db"]
        word = "very clean" if s >= 30 else "clean" if s >= 20 else "noticeable background noise" if s >= 10 else "noisy"
        rec += [f"SNR about {s:.0f} dB ({word})", f"noise floor {r['noise_floor_dbfs']:.0f} dBFS"]
    if (bw := r.get("bandwidth")) and not c.get("narrowband"):
        rec.append(f"speech energy only up to about {bw['high_hz'] / 1000:.1f} kHz (band-limited, phone-like)"
                   if bw["band_limited"] else f"speech energy up to about {bw['high_hz'] / 1000:.1f} kHz")
    if h := r.get("hum"):
        rec.append(f"mains hum at {h['hz']} Hz (harmonics {', '.join(map(str, h['harmonics'])) or '1'}; "
                   f"up to {h['prominence_db']:.0f} dB above the surrounding noise)")
    nv = r.get("noise_variation_db")
    if nv is not None and nv <= 6 and r.get("snr_db", 99) < 40:
        rec.append("steady background noise")         # "fluctuating" is not stated: pauses hold breaths etc.
    sp = (r.get("space") or {})
    if sp.get("bucket") and sp["bucket"] != "dry":
        rec.append(f"the room adds {sp['bucket']} (level falls at about {60 / sp['decay_s']:.0f} dB/s "
                   f"after {sp['tails']} phrase ends)")
    if lv := g.get("levels"):
        rec.append(f"peak {lv['peak_dbfs']:.1f} dBFS")
        rec.append("severe clipping" if lv["severe_clipping"] else
                   f"light clipping ({lv['clipping_pct']:.2f}% of samples)" if lv["clipping_pct"] > 0.01 else "no clipping")
    out.append("Recording: " + ", ".join(rec) + ".")

    ev = kept_events(f)
    if ev:
        out.append(f"Other sounds (timed below): {', '.join(dict.fromkeys(sound(e['label']) for e in ev))}.")
    elif f.get("run", {}).get("models", {}).get("events"):
        out.append("No other sounds detected by the sound-event model.")
    if bg := g.get("other_sounds"):
        out.append("Outside speech the sound-event model also hears: "
                   + ", ".join(f"{b['class']} ({b['p']:.2f})" for b in bg) + ".")

    if (p := g.get("pitch")) and p.get("confirmed", True):   # second pitch tracker must agree
        out.append(f"Voice: median pitch {p['median_hz']:.0f} Hz, usual range {p['p10_hz']:.0f}–{p['p90_hz']:.0f} Hz "
                   f"({p['range_st']:.0f} semitones, 10th–90th percentile).")
    if v := g.get("voice"):
        bits = [f"{lab} {v[k]:.1f} {u}" for k, lab, u in
                (("hnr_db", "harmonic-to-noise ratio", "dB"), ("cpp_db", "cepstral peak prominence", "dB"),
                 ("h1h2_db", "H1–H2", "dB"), ("jitter_pct", "jitter", "%"), ("shimmer_db", "shimmer", "dB"))
                if v.get(k) is not None]
        if bits:
            out.append("Voice measurements (numbers only, not a judgement of voice quality): "
                       + ", ".join(bits) + ".")
    if (s := g.get("speakers")) and s.get("confident"):      # anything in between: say nothing
        out.append("One voice throughout." if s["confident"] == "single" else
                   f"At least {s['count']} clearly different voices; the voice changes at "
                   + ", ".join(f"{t:.2f} s" for t in s["change_times"]) + ".")

    s = g.get("speech")
    if s is not None:
        if not s["phrases"]:
            out.append("No speech detected.")
        else:
            pauses = [x["ms"] for x in f["local"].get("pauses", [])]
            t = f"Speech: {s['speech_s']:.1f} s in {s['phrases']} phrase{'s' if s['phrases'] > 1 else ''}"
            if len(pauses) == 1:
                t += f" with one pause of {pauses[0]} ms"
            elif pauses:
                t += f" separated by {len(pauses)} pauses ({min(pauses)}–{max(pauses)} ms)"
            w = f.get("local", {}).get("words") or {}
            if w.get("syllables_per_s"):
                t += (f"; {w['syllables']} syllables in the transcript, "
                      f"about {w['syllables_per_s']:.1f} per second overall")
            elif s.get("syllables_per_s"):
                t += f"; about {s['syllables_per_s']:.1f} syllables per second overall"
            rh = s.get("phrase_rhythm")
            if rh and rh["cv"] <= 0.15:
                t += f"; phrases start at a nearly regular {rh['period_s']:.2f} s period"
            out.append(t + ".")
    return " ".join(out)


def change_bits(ch: dict) -> list[str]:
    """Only the parts that survived their checks: a pitch jump the second tracker confirmed,
    and a loudness step that is not half of a brief burst."""
    out = []
    if "pitch_st" in ch and ch.get("confirmed", True):
        out.append(f"pitch {'jumps up' if ch['pitch_st'] > 0 else 'drops'} {abs(ch['pitch_st']):.0f} semitones "
                   f"({ch['from_hz']:.0f} → {ch['to_hz']:.0f} Hz)")
    if "level_db" in ch and not ch.get("transient") and ch.get("level_confirmed", True):
        out.append(f"{abs(ch['level_db']):.0f} dB {'louder' if ch['level_db'] > 0 else 'softer'}")
    return out


def phrase_bits(ph: dict, several: bool) -> list[str]:
    bits = []
    if ph.get("syllables_per_s"):
        n = ph["syllables"]
        bits.append(f"about {n} syllable{'s' if n > 1 else ''} ({ph['syllables_per_s']:.1f} per second)")
    r1, r2 = ph.get("rate_first_half"), ph.get("rate_second_half")
    if ph.get("rate_change") and r1 and r2:
        bits.append(f"{'speeds up' if ph['rate_change'] == 'speeds_up' else 'slows down'} "
                    f"from about {r1:.1f} to {r2:.1f} syllables per second")
    # pitch numbers are stated only when a second, independent tracker agrees with them
    if (p := ph.get("pitch")) and ph.get("pitch_confirmed") is not False:
        tr = p["trend_st"]
        move = (f"rises about {tr:.0f} semitones" if tr >= 1.5 else
                f"falls about {-tr:.0f} semitones" if tr <= -1.5 else "stays level")
        rng = f"range {p['range_st']:.0f} semitones" if p["range_st"] >= 1 else "flat (within a semitone)"
        bits.append(f"pitch median {p['median_hz']:.0f} Hz, {rng}, {move} start to end")
        if p["peak_hz"] / p["median_hz"] >= 2 ** (4 / 12) and p.get("peak_s") is not None:
            bits.append(f"pitch peak about {p['peak_hz']:.0f} Hz near {p['peak_s']:.2f} s")
    fp = ph.get("final_pitch_st")
    if fp is not None and abs(fp) >= 2 and ph.get("final_pitch_confirmed") is not False:
        bits.append(f"ends {abs(fp):.0f} semitones {'lower' if fp < 0 else 'higher'} than the rest of the phrase")
    d = ph.get("level_vs_others_db")
    if d is not None and several:
        if abs(d) >= 3:
            bits.append(f"{abs(d):.0f} dB {'louder' if d > 0 else 'softer'} than the other phrases")
    on, off = ph.get("onset_ms"), ph.get("offset_ms")
    if on is not None:
        if on <= 20:
            bits.append(f"abrupt start ({on:.0f} ms rise)")
        elif on >= 120:
            bits.append(f"gradual start ({on:.0f} ms rise)")
    if off is not None:
        if off <= 20:
            bits.append(f"cut off abruptly ({off:.0f} ms)")
        elif off >= 150:
            bits.append(f"fades out over {off:.0f} ms")
    return bits


def local_text(f: dict) -> str:
    loc = f.get("local", {})
    phrases = loc.get("phrases", [])
    if not phrases:
        return ""
    # phrase edges are only stated when the recording is clean enough to see them
    clean = ((f.get("global", {}).get("recording") or {}).get("snr_db") or 0) >= 20
    items = []
    for ph in phrases:
        if not clean:
            ph = {**ph, "onset_ms": None, "offset_ms": None}
        bits = phrase_bits(ph, len(phrases) > 1)
        kind = "short phrase" if ph["end"] - ph["start"] < 0.5 else "phrase"
        items.append((ph["start"], f"{span(ph['start'], ph['end'])}, {kind} {ph['n']}"
                                   + (": " + "; ".join(bits) if bits else "") + "."))
    for ch in loc.get("changes", []):
        bits = change_bits(ch)
        if bits:
            items.append((ch["t"], f"{ch['t']:.2f} s{', within phrase ' + str(ch['phrase']) if ch.get('phrase') else ''}: "
                                   + "; ".join(bits) + "."))
    for pa in loc.get("pauses", []):
        items.append((pa["start"], f"{span(pa['start'], pa['end'])}, pause of {pa['ms']} ms."))
    for e in kept_events(f):
        where = "during speech" if e.get("speech_overlap", 0) > 0 else "outside speech"
        extra = f", {e['pulses']} pulses" if e.get("pulses") and e["pulses"] >= 2 else ""
        line = f"{span(e['start'], e['end'])}, {sound(e['label'])} ({where}, p {e['p']:.2f}{extra})"
        cp = e.get("coupling") or {}
        d = cp.get("level_change_db")
        if d is not None and abs(d) >= 5:
            line += f"; the voice is {abs(d):.0f} dB {'louder' if d > 0 else 'softer'} in the second after it starts"
        items.append((e["start"], line + "."))
    w = loc.get("words") or {}
    for fl in w.get("fillers", []):
        if fl.get("start") is not None:
            items.append((fl["start"], f'{span(fl["start"], fl["end"])}, hesitation sound "{fl["text"]}" '
                                       f'(from the transcript).'))
    if (dw := w.get("drawn_out")) and dw.get("start") is not None:
        items.append((dw["start"], f'{span(dw["start"], dw["end"])}, the word "{dw["text"]}" is drawn out '
                                   f'({dw["ratio"]:.1f}x the usual time per syllable).'))
    return "\n".join(t for _, t in sorted(items, key=lambda x: x[0]))
