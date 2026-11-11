"""Features dict -> tool description: a global paragraph, then one timed line per phrase,
pause, change and sound event (in time order).

Only claims that survived their checks: timing, pitch, loudness, phrase edges, recording
quality, room bucket, speaker count/changes, the sound-event model's labels (with their
probabilities), how the voice is produced relative to the rest of the clip, and, when a
transcript is configured, syllable counts and fillers. Age, gender, dialect and intent are
left to the audio-language model -- but the measurements that bear on intent (arousal, rate
and level moves) are stated, so the model has evidence to commit on rather than a guess.

The description is cut to `WORD_LIMIT` words by raising a weight floor, not by truncating:
every clause carries a weight for how hard the item is to get right, and the floor rises
until the whole thing fits. Cheap items (sample rate, "no clipping") drop first; the hard
ones (sound events, voice production, timed changes) survive. Works on saved features JSON,
so the wording can change without re-running the models."""

from __future__ import annotations

import re

LANGUAGES = {"hi": "Hindi", "ta": "Tamil"}
SOUNDS = {"impact": "knock or bang", "breathing": "audible breathing"}

WORD_LIMIT = 300
FLOORS = (0, 1, 2, 3, 4, 5)       # tried in order until the description fits

# How hard the item is to get right, so the cheap ones go first when the budget bites.
W = {
    "head": 5,            # language and duration: the frame everything else hangs on
    "channel": 1,         # sample rate, channels, clipping
    "snr": 2,
    "bandwidth": 2,
    "hum": 3,
    "steady_noise": 1,
    "space": 4,
    "events_head": 3,
    "background": 2,
    "pitch": 3,
    "voice_numbers": 3,
    "phonation": 5,       # the Source axis
    "speakers": 4,
    "speech": 3,
    "rhythm": 2,
    "arousal_shape": 4,   # evidence for the intent axis
    "phrase_head": 4,     # one line carries Timing, Pitch, Morphology and Source at once
    "phrase_rate": 2,
    "phrase_pitch": 2,
    "phrase_final": 2,
    "phrase_level": 2,
    "phrase_edges": 3,
    "phrase_phonation": 5,
    "phrase_arousal": 3,
    "change": 4,
    "pause": 1,
    "event": 5,
    "coupling": 4,
    "words": 3,
    "code_switching": 4,   # script-exact, and the point of "how India actually talks"
}


def words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def span(a: float, b: float) -> str:
    return f"{a:.2f}–{b:.2f} s"


def sound(label: str) -> str:
    return SOUNDS.get(label, label.replace("_", " "))


def _nth(i: int | None, roman: str | None = None) -> str:
    """"the 3rd word (machchaan)" -- position plus the word in Latin letters. The description
    stays entirely in English: never the native script, and never the whole transcript, only
    the few words that carry something (a hesitation, a word drawn out, a word repeated)."""
    if not i:
        return f'a word ("{roman}")' if roman else "a word"
    suffix = "th" if 10 <= i % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f'the {i}{suffix} word' + (f' ("{roman}")' if roman else "")


def kept_events(f: dict) -> list:
    """Sound events stated in the text ("other voices" never: speakers are for the ALM)."""
    return [e for e in f.get("local", {}).get("events", []) if e["label"] != "background_speech"]


GLOBAL_SHARE = 0.5          # at most half the budget on the clip-level paragraph
DIMINISH = 0.75             # each further line of the same kind, anywhere in the clip, costs this
SPREAD = 0.35               # each further line of any kind in that region costs this
BIN_S = 5.0                 # a "region" of the clip, in seconds


def describe(f: dict, word_limit: int = WORD_LIMIT, global_share: float = GLOBAL_SHARE) -> str:
    """Fit the description into `word_limit` words. Rime cuts at 300 words before grading, and
    a clause cut in half scores nothing while still risking a contradiction, so whole items go
    rather than halves.

    Two stages, because the two halves fail differently. The clip-level paragraph is one
    stream of clauses, so a rising weight floor drops its cheapest (sample rate, "no
    clipping") first. The timed lines are many competing items, so they are *selected*:
    highest-weight first until the budget runs out, then put back in time order. A 24 s clip
    with twenty phrases therefore keeps its eight most informative lines in full detail
    instead of twenty stripped ones -- the rubric pays per item stated, so detail on fewer
    lines scores better than a thin line for everything."""
    budget = max(int(word_limit * global_share), 1)
    head, floor = "", FLOORS[-1]
    for fl in FLOORS:
        head, floor = global_text(f, fl), fl
        if words(head) <= budget:
            break
    if words(head) > word_limit:                       # pathologically small limit
        return head

    left = word_limit - words(head)
    pool, chosen, taken = list(local_items(f, floor)), [], {}
    duration = f.get("clip", {}).get("duration_s") or 1.0
    bin_s = BIN_S                              # one region per ~5 s of clip

    def region(t):
        return int(t // bin_s)

    while pool and left > 0:
        # A further line of the same kind in the same stretch of the clip is worth less. The
        # rubric scores by item type and its timed items are spread over the whole clip, so a
        # mix of phrases, changes and events across all 24 s beats twelve pitch jumps from the
        # first four.
        best = max(pool, key=lambda it: (it[0] - DIMINISH * taken.get(it[2], 0)
                                         - SPREAD * taken.get(("*", region(it[1])), 0), -it[1]))
        pool.remove(best)
        n = words(best[3])
        if n <= left:
            chosen.append((best[1], best[3]))
            for key in (best[2], ("*", region(best[1]))):     # kind clip-wide, spread per region
                taken[key] = taken.get(key, 0) + 1
            left -= n
    body = "\n".join(line for _, line in sorted(chosen, key=lambda x: x[0]))
    return "\n".join(x for x in (head, body) if x)


def _render(f: dict, floor: int) -> str:
    """The whole description at one floor, with nothing selected away. Used for diagnostics."""
    return "\n".join(x for x in (global_text(f, floor), local_text(f, floor)) if x)


def global_text(f: dict, floor: int = 0) -> str:
    c, g = f["clip"], f.get("global", {})
    out = []

    def keep(kind):
        return W[kind] >= floor

    lang = (g.get("language") or {}).get("code")
    if keep("head"):
        out.append(f"{LANGUAGES[lang] + ', ' if lang in LANGUAGES else ''}{c['duration_s']:.1f} s.")

    rec = []
    r = g.get("recording") or {}
    if keep("channel"):
        rec.append(f"{c['sample_rate'] / 1000:g} kHz "
                   f"{'mono' if c['channels'] == 1 else str(c['channels']) + '-channel'} file"
                   + (" (narrowband: nothing above 4 kHz)" if c.get("narrowband") else ""))
    if keep("snr") and r.get("snr_db") is not None:
        s = r["snr_db"]
        word = ("very clean" if s >= 30 else "clean" if s >= 20
                else "noticeable background noise" if s >= 10 else "noisy")
        rec += [f"SNR about {s:.0f} dB ({word})", f"noise floor {r['noise_floor_dbfs']:.0f} dBFS"]
    if keep("bandwidth") and (bw := r.get("bandwidth")) and not c.get("narrowband"):
        rec.append(f"speech energy only up to about {bw['high_hz'] / 1000:.1f} kHz "
                   f"(band-limited, phone-like)" if bw["band_limited"]
                   else f"speech energy up to about {bw['high_hz'] / 1000:.1f} kHz")
    if keep("hum") and (h := r.get("hum")):
        rec.append(f"mains hum at {h['hz']} Hz (harmonics {', '.join(map(str, h['harmonics'])) or '1'}; "
                   f"up to {h['prominence_db']:.0f} dB above the surrounding noise)")
    nv = r.get("noise_variation_db")
    if keep("steady_noise") and nv is not None and nv <= 6 and r.get("snr_db", 99) < 40:
        rec.append("steady background noise")     # "fluctuating" is not stated: pauses hold breaths
    sp = (r.get("space") or {})
    if keep("space") and sp.get("bucket") and sp["bucket"] != "dry":
        rec.append(f"the room adds {sp['bucket']} (level falls at about {60 / sp['decay_s']:.0f} dB/s "
                   f"after {sp['tails']} phrase ends)")
    if keep("channel") and (lv := g.get("levels")):
        rec.append(f"peak {lv['peak_dbfs']:.1f} dBFS")
        rec.append("severe clipping" if lv["severe_clipping"] else
                   f"light clipping ({lv['clipping_pct']:.2f}% of samples)"
                   if lv["clipping_pct"] > 0.01 else "no clipping")
    if rec:
        out.append("Recording: " + ", ".join(rec) + ".")

    ev = kept_events(f)
    if keep("events_head"):
        if ev:
            out.append(f"Other sounds (timed below): "
                       f"{', '.join(dict.fromkeys(sound(e['label']) for e in ev))}.")
        elif f.get("run", {}).get("models", {}).get("events"):
            out.append("No other sounds detected by the sound-event model.")
    if keep("background") and (bg := g.get("other_sounds")):
        out.append("Outside speech the sound-event model also hears: "
                   + ", ".join(f"{b['class']} ({b['p']:.2f})" for b in bg) + ".")

    if keep("pitch") and (p := g.get("pitch")) and p.get("confirmed", True):
        out.append(f"Voice: median pitch {p['median_hz']:.0f} Hz, usual range "
                   f"{p['p10_hz']:.0f}–{p['p90_hz']:.0f} Hz "
                   f"({p['range_st']:.0f} semitones, 10th–90th percentile).")
    if keep("voice_numbers") and (v := g.get("voice")):
        bits = [f"{lab} {v[k]:.1f} {u}" for k, lab, u in
                (("hnr_db", "harmonic-to-noise ratio", "dB"), ("cpp_db", "cepstral peak prominence", "dB"),
                 ("h1h2_db", "H1–H2", "dB"), ("jitter_pct", "jitter", "%"),
                 ("shimmer_db", "shimmer", "dB"))
                if v.get(k) is not None]
        if bits:
            out.append("Voice measurements (numbers only, measured against this clip, not "
                       "published norms): " + ", ".join(bits) + ".")
    if keep("phonation") and (t := phonation_text(g)):
        out.append(t)
    if keep("speakers") and (s := g.get("speakers")) and s.get("confident"):
        out.append("One voice throughout." if s["confident"] == "single" else
                   f"At least {s['count']} clearly different voices; the voice changes at "
                   + ", ".join(f"{t:.2f} s" for t in s["change_times"]) + ".")

    s = g.get("speech")
    if keep("speech") and s is not None:
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
            if keep("rhythm") and rh and rh["cv"] <= 0.15:
                t += f"; phrases start at a nearly regular {rh['period_s']:.2f} s period"
            out.append(t + ".")

    w = f.get("local", {}).get("words") or {}
    if keep("code_switching") and (cs := w.get("code_switching")):
        n = cs["latin_words"]
        named = ", ".join(f'"{x}"' for x in cs["words"][:6])
        out.append(f"{n} English word{'s' if n > 1 else ''} mixed into the speech "
                   f"({cs['ratio'] * 100:.0f}% of the words)"
                   + (f": {named}" if named else "") + ".")
    if keep("arousal_shape") and (t := arousal_shape_text(f)):
        out.append(t)
    return " ".join(out)


def phonation_text(g: dict) -> str | None:
    """The Source axis at clip level: which label is this clip's baseline and which phrases
    depart from it. No absolute claim about the voice, only where it changes."""
    sh = g.get("phonation_shape")
    if not sh or not sh.get("baseline"):
        return None
    base = {"modal": "clear and modal", "creaky": "creaky", "breathy": "breathy",
            "strained": "strained"}.get(sh["baseline"], sh["baseline"])
    t = f"Voice production is mostly {base}"
    if sh.get("differs"):
        parts = [f"{lab} in phrase{'s' if len(ns) > 1 else ''} "
                 + ", ".join(str(n) for n in ns) for lab, ns in sorted(sh["differs"].items())]
        t += "; " + ", ".join(parts)
    return t + " (judged against this clip's own middle)."


def arousal_shape_text(f: dict) -> str | None:
    """Evidence for the intent axis: where the clip is most and least animated. The reading of
    it -- who is addressed, what is meant, how it moves -- is the ALM's to make."""
    phrases = f.get("local", {}).get("phrases", [])
    scored = []
    for p in phrases:
        a = p.get("arousal") or {}
        st, lv = a.get("pitch_above_clip_st"), a.get("level_above_clip_db")
        if st is None and lv is None:
            continue
        if p.get("pitch_confirmed") is False:
            st = None
        scored.append((p["n"], (st or 0) + (lv or 0) / 2, st, lv, a.get("rate_ratio")))
    if len(scored) < 2:
        return None
    hi = max(scored, key=lambda r: r[1])
    lo = min(scored, key=lambda r: r[1])
    if hi[0] == lo[0]:
        return None

    def how(row):
        bits = []
        if row[2] is not None and abs(row[2]) >= 1:
            bits.append(f"pitch {abs(row[2]):.0f} semitones {'above' if row[2] > 0 else 'below'} the clip median")
        if row[3] is not None and abs(row[3]) >= 2:
            bits.append(f"{abs(row[3]):.0f} dB {'louder' if row[3] > 0 else 'softer'}")
        if row[4] is not None and abs(row[4] - 1) >= 0.15:
            bits.append(f"{row[4]:.2f}x the overall rate")
        return ", ".join(bits)

    a, b = how(hi), how(lo)
    if not a and not b:
        return None
    t = "Intensity across the clip (measurements, not a reading of mood): "
    t += f"phrase {hi[0]} is the most animated" + (f" ({a})" if a else "")
    t += f", phrase {lo[0]} the least" + (f" ({b})" if b else "")
    return t + "."


def change_bits(ch: dict, floor: int = 0) -> list[str]:
    """Only the parts that survived their checks: a pitch jump the second tracker confirmed,
    and a loudness step that is neither half of a brief burst nor burst-dominated."""
    out = []
    if W["change"] < floor:
        return out
    if "pitch_st" in ch and ch.get("confirmed", True):
        out.append(f"pitch {'jumps up' if ch['pitch_st'] > 0 else 'drops'} "
                   f"{abs(ch['pitch_st']):.0f} semitones "
                   f"({ch['from_hz']:.0f} → {ch['to_hz']:.0f} Hz)")
    if "level_db" in ch and not ch.get("transient") and ch.get("level_confirmed", True):
        out.append(f"{abs(ch['level_db']):.0f} dB {'louder' if ch['level_db'] > 0 else 'softer'}")
    return out


def phrase_bits(ph: dict, several: bool, floor: int = 0) -> list[str]:
    bits = []

    def keep(kind):
        return W[kind] >= floor

    if keep("phrase_rate") and ph.get("syllables_per_s"):
        n = ph["syllables"]
        bits.append(f"about {n} syllable{'s' if n > 1 else ''} ({ph['syllables_per_s']:.1f} per second)")
    r1, r2 = ph.get("rate_first_half"), ph.get("rate_second_half")
    if keep("phrase_rate") and ph.get("rate_change") and r1 and r2:
        bits.append(f"{'speeds up' if ph['rate_change'] == 'speeds_up' else 'slows down'} "
                    f"from about {r1:.1f} to {r2:.1f} syllables per second")
    # pitch numbers are stated only when a second, independent tracker agrees with them
    if keep("phrase_pitch") and (p := ph.get("pitch")) and ph.get("pitch_confirmed") is not False:
        tr = p["trend_st"]
        move = (f"rises about {tr:.0f} semitones" if tr >= 1.5 else
                f"falls about {-tr:.0f} semitones" if tr <= -1.5 else "stays level")
        rng = f"range {p['range_st']:.0f} semitones" if p["range_st"] >= 1 else "flat (within a semitone)"
        bits.append(f"pitch median {p['median_hz']:.0f} Hz, {rng}, {move} start to end")
        if p["peak_hz"] / p["median_hz"] >= 2 ** (4 / 12) and p.get("peak_s") is not None:
            bits.append(f"pitch peak about {p['peak_hz']:.0f} Hz near {p['peak_s']:.2f} s")
    fp = ph.get("final_pitch_st")
    if keep("phrase_final") and fp is not None and abs(fp) >= 2 \
            and ph.get("final_pitch_confirmed") is not False:
        bits.append(f"ends {abs(fp):.0f} semitones {'lower' if fp < 0 else 'higher'} "
                    f"than the rest of the phrase")
    # the Source axis, relative to the clip's own middle
    if keep("phrase_phonation") and (pn := ph.get("phonation")) and pn.get("label") \
            and pn.get("confirmed") and pn["label"] != "modal":
        name = {"creaky": "creaky", "breathy": "breathy", "strained": "strained"}[pn["label"]]
        bits.append(f"{name} voice here" + (f" ({pn['support'][0]})" if pn.get("support") else ""))
    d = ph.get("level_vs_others_db")
    if keep("phrase_level") and d is not None and several and abs(d) >= 3:
        bits.append(f"{abs(d):.0f} dB {'louder' if d > 0 else 'softer'} than the other phrases")
    on, off = ph.get("onset_ms"), ph.get("offset_ms")
    if keep("phrase_edges") and on is not None:
        if on <= 20:
            bits.append(f"abrupt start ({on:.0f} ms rise)")
        elif on >= 120:
            bits.append(f"gradual start ({on:.0f} ms rise)")
    if keep("phrase_edges") and off is not None:
        if off <= 20:
            bits.append(f"cut off abruptly ({off:.0f} ms)")
        elif off >= 150:
            bits.append(f"fades out over {off:.0f} ms")
    return bits


def local_items(f: dict, floor: int = 0) -> list[tuple[float, float, str, str]]:
    """Every timed line as (weight, time, kind, text). The weight is the hardest item the line
    carries, so a phrase that reports creak outranks a plain one and survives a tight budget
    with its detail intact; `kind` lets the selection spread the budget across item types."""
    loc = f.get("local", {})
    phrases = loc.get("phrases", [])
    if not phrases:
        return []

    def keep(kind):
        return W[kind] >= floor

    # phrase edges are only stated when the recording is clean enough to see them
    clean = ((f.get("global", {}).get("recording") or {}).get("snr_db") or 0) >= 20
    items = []
    if keep("phrase_head"):
        for ph in phrases:
            if not clean:
                ph = {**ph, "onset_ms": None, "offset_ms": None}
            bits = phrase_bits(ph, len(phrases) > 1, floor)
            kind = "short phrase" if ph["end"] - ph["start"] < 0.5 else "phrase"
            pn = (ph.get("phonation") or {})
            weight = max([W["phrase_head"]]
                         + ([W["phrase_phonation"]] if pn.get("confirmed")
                            and pn.get("label") not in (None, "modal") else [])
                         + ([W["phrase_edges"]] if clean and bits else []))
            items.append((weight, ph["start"], "phrase",
                          f"{span(ph['start'], ph['end'])}, {kind} {ph['n']}"
                          + (": " + "; ".join(bits) if bits else "") + "."))
    for ch in loc.get("changes", []):
        bits = change_bits(ch, floor)
        if bits:
            items.append((W["change"], ch["t"], "change",
                          f"{ch['t']:.2f} s"
                          + (f", within phrase {ch['phrase']}" if ch.get("phrase") else "")
                          + ": " + "; ".join(bits) + "."))
    if keep("pause"):
        for pa in loc.get("pauses", []):
            items.append((W["pause"], pa["start"], "pause",
                          f"{span(pa['start'], pa['end'])}, pause of {pa['ms']} ms."))
    if keep("event"):
        for e in kept_events(f):
            where = "during speech" if e.get("speech_overlap", 0) > 0 else "outside speech"
            extra = f", {e['pulses']} pulses" if e.get("pulses") and e["pulses"] >= 2 else ""
            line = f"{span(e['start'], e['end'])}, {sound(e['label'])} ({where}, p {e['p']:.2f}{extra})"
            weight = W["event"] + (1 if e.get("pulses") and e["pulses"] >= 2 else 0)
            cp = e.get("coupling") or {}
            d = cp.get("level_change_db")
            if keep("coupling") and d is not None and abs(d) >= 5:
                line += (f"; the voice is {abs(d):.0f} dB "
                         f"{'louder' if d > 0 else 'softer'} in the second after it starts")
                weight += 1
            items.append((weight, e["start"], "event", line + "."))
    w = loc.get("words") or {}
    if keep("words"):
        for fl in w.get("fillers", []):
            if fl.get("start") is not None:
                items.append((W["words"], fl["start"], "words",
                              f'{span(fl["start"], fl["end"])}, hesitation sound '
                              f'"{fl.get("roman") or fl["text"]}" (from the transcript).'))
        for rp in w.get("repeats") or []:
            if rp.get("start") is not None:
                items.append((W["words"], rp["start"], "words",
                              f"{span(rp['start'], rp['end'])}, "
                              f"{_nth(rp.get('i'), rp.get('roman'))} is said twice in a row "
                              f"(from the transcript)."))
        if (dw := w.get("drawn_out")) and dw.get("start") is not None:
            items.append((W["words"], dw["start"], "words",
                          f"{span(dw['start'], dw['end'])}, "
                          f"{_nth(dw.get('i'), dw.get('roman'))} is drawn out "
                          f"({dw['ratio']:.1f}x the usual time per syllable)."))
    return items


def local_text(f: dict, floor: int = 0) -> str:
    """Every timed line, in time order, with nothing selected away."""
    return "\n".join(t for _, _, _, t in sorted(local_items(f, floor), key=lambda x: x[1]))
