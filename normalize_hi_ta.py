"""
normalize_hi_ta.py - frozen text normalizer for CER/WER on Hindi and Tamil ONLY.

Apply the SAME function (same arguments) to the reference and the hypothesis
before computing any error rate. Do not change it between runs.

    from normalize_hi_ta import normalize_text, normalize_no_space, graphemes

    ref = normalize_text(ref_raw, "hi")                       # strict: raises on other scripts
    hyp = normalize_text(hyp_raw, "hi", on_foreign="keep")    # keep stray English etc. so it is penalised
"""
import unicodedata

NORMALIZER_VERSION = "2.0"
LANGS = ("hi", "ta")

# Unicode blocks: Devanagari (Hindi) and Tamil.
_RANGES = {"hi": (0x0900, 0x097F), "ta": (0x0B80, 0x0BFF)}

# Zero-width / directional formatting characters (ZWSP, ZWNJ, ZWJ, WJ, BOM, LRM, RLM)
_ZERO_WIDTH = {ord(c): None for c in "\u200b\u200c\u200d\u2060\ufeff\u200e\u200f"}

_NUKTA = "\u093c"
_CHANDRABINDU = "\u0901"
_ANUSVARA = "\u0902"
_DEVANAGARI_VIRAMA = "\u094d"  # joins the NEXT consonant into one akshara (Tamil pulli does not)
_ASCII_DIGITS = set("0123456789")


def _script_of(ch):
    cp = ord(ch)
    for lang, (lo, hi) in _RANGES.items():
        if lo <= cp <= hi:
            return lang
    return None


def _core(text):
    # 1. NFC: composes Tamil vowel signs (ொ ோ ௌ) and decomposes Devanagari
    #    nukta letters (क़ -> क + ़), so equivalent spellings match.
    text = unicodedata.normalize("NFC", text)
    # 2. Drop zero-width joiners / non-joiners / direction marks.
    text = text.translate(_ZERO_WIDTH)
    # 3. Digits of any script -> ASCII; keep letters (L), marks (M, incl. virama
    #    and vowel signs), other numbers (N); everything else (punctuation,
    #    danda ।॥, symbols, quotes, emoji, control chars) -> space.
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Nd":
            out.append(str(unicodedata.digit(ch)))
        elif cat[0] in ("L", "M", "N"):
            out.append(ch)
        else:
            out.append(" ")
    # 4. Collapse whitespace.
    return " ".join("".join(out).split())


def detect_language(text):
    """Return 'hi' or 'ta' by majority letter script; ValueError if no Hindi/Tamil letters."""
    counts = {"hi": 0, "ta": 0}
    for ch in text:
        if unicodedata.category(ch)[0] == "L":
            s = _script_of(ch)
            if s:
                counts[s] += 1
    if not any(counts.values()):
        raise ValueError("No Hindi (Devanagari) or Tamil letters found in transcript.")
    return max(counts, key=counts.get)


def foreign_chars(text, lang):
    """Characters in already-normalized text that are not `lang` script, ASCII digits or space."""
    return sorted({
        ch for ch in text
        if ch != " " and ch not in _ASCII_DIGITS and _script_of(ch) != lang
    })


def normalize_text(text, lang=None, *, hindi_lenient=False, on_foreign="raise"):
    """Normalize one transcript.

    lang:          'hi' or 'ta'. If None, it is auto-detected (prefer passing it
                   explicitly for the whole dataset).
    hindi_lenient: Hindi only, lossy. Deletes nukta (़) and maps chandrabindu (ँ)
                   to anusvara (ं), since ASR often confuses them. Default OFF.
    on_foreign:    what to do with characters from any other script (English, etc.):
                   'raise' - ValueError (use for references / clean data)
                   'keep'  - leave them in, so the metric penalises them (use for hypotheses)
                   'drop'  - replace them with a space
    """
    if lang is not None and lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
    if on_foreign not in ("raise", "keep", "drop"):
        raise ValueError("on_foreign must be 'raise', 'keep' or 'drop'")
    if not text:
        return ""

    out = _core(text)
    if not out:
        return ""
    if lang is None:
        lang = detect_language(out)

    if lang == "hi" and hindi_lenient:
        out = out.replace(_NUKTA, "").replace(_CHANDRABINDU, _ANUSVARA)

    bad = foreign_chars(out, lang)
    if bad:
        if on_foreign == "raise":
            raise ValueError(f"Non-{lang} characters found: {bad} in {text!r}")
        if on_foreign == "drop":
            badset = set(bad)
            out = " ".join("".join(" " if c in badset else c for c in out).split())
    return out


def normalize_no_space(text, lang=None, **kwargs):
    """Normalized text with spaces removed, for the 'CER without spaces' variant
    (ignores word-boundary ambiguity such as joined vs separated postpositions)."""
    return normalize_text(text, lang, **kwargs).replace(" ", "")


def graphemes(text):
    """Split into aksharas for grapheme-level CER. Combining marks attach to the
    previous character; after a Devanagari virama the next consonant joins the
    cluster (क्ष is one unit). Tamil pulli does not join (க்க = க் + க).
    Pure Python, so results do not depend on the `regex` package version."""
    clusters, join_next = [], False
    for ch in text:
        if clusters and ch != " " and (unicodedata.category(ch)[0] == "M" or join_next):
            clusters[-1] += ch
        else:
            clusters.append(ch)
        join_next = ch == _DEVANAGARI_VIRAMA
    return clusters


if __name__ == "__main__":
    tests = [
        # (lang, raw, expected)
        ("hi", "मुझे ट्रेन पकड़नी है।", "मुझे ट्रेन पकड\u093cनी है"),
        ("hi", "मुझे ट्रेन पक\u095cनी है", "मुझे ट्रेन पक\u0921\u093cनी है"),  # composed ड़ -> ड + nukta
        ("hi", "मेरे ३ भाई हैं ।", "मेरे 3 भाई हैं"),
        ("hi", "वह  \u200dघर\u200c गया ॥", "वह घर गया"),
        ("ta", "எனக்கு ௧௨ வயது, சரியா?", "எனக்கு 12 வயது சரியா"),
        ("ta", "வணக்கம்\u200d  நண்பா!", "வணக்கம் நண்பா"),
        (None, "வணக்கம் நண்பா", "வணக்கம் நண்பா"),  # auto-detect
    ]
    ok = True
    for lang, raw, expected in tests:
        got = normalize_text(raw, lang)
        good = got == expected
        ok &= good
        print("OK  " if good else "FAIL", lang, repr(raw), "->", repr(got))

    # Other languages must be rejected / handled per policy
    for raw, lang in [("मुझे Train पकड़नी है", "hi"), ("வணக்கம் hello", "ta"), ("வணக்கம்", "hi")]:
        try:
            normalize_text(raw, lang)
            print("FAIL no error for", repr(raw)); ok = False
        except ValueError as e:
            print("OK   rejected:", e)
    k = normalize_text("मुझे Train है", "hi", on_foreign="keep")
    d = normalize_text("मुझे Train है", "hi", on_foreign="drop")
    ok &= (k == "मुझे Train है") and (d == "मुझे है")
    print("keep:", k, "| drop:", d)

    print("lenient:", normalize_text("आँख पकड़ा", "hi", hindi_lenient=True))
    print("no-space:", normalize_no_space("मुझे ट्रेन पकड़नी है।", "hi"))
    print("graphemes hi:", graphemes("क्षमा पकड\u093cनी"))
    print("graphemes ta:", graphemes("தமிழ் க்க"))
    print("ALL PASSED" if ok else "SOME FAILED")
