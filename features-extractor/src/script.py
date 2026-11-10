"""Hindi (Devanagari) and Tamil text: syllable counts, fillers, and word timing.

Both scripts are abugidas, so syllables are countable by rule, with no model and no
pronunciation dictionary:
  * an independent vowel letter is one syllable;
  * a consonant is one syllable unless the next character is the virama / pulli
    (्  U+094D, ்  U+0BCD), which removes its vowel;
  * vowel signs (matras), anusvara and visarga attach to a consonant already counted.
Devanagari only: a word-final bare consonant loses its inherent "a" in Hindi
(schwa deletion), so one syllable is taken off (घर = ghar, 1; राम = rām, 1).

Checked against two Tamil clips whose transcripts were counted by hand: வணக்கம் = 3,
கேட்குதா = 3, நினைக்கிறேன் = 4.

Word times come from the syllable nuclei the phoneme model found: the words' syllable
counts are laid over the nuclei in order (`align_words`). The result is monotonic and good
enough for "which word is drawn out"; it is not forced alignment, and it says so in the JSON
(`method: "syllable_proportional"`).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

DEV_VOWEL = (0x0904, 0x0914)            # अ … औ (independent vowels)
DEV_CONS = (0x0915, 0x0939)             # क … ह
DEV_CONS_EXTRA = (0x0958, 0x095F)       # क़ … य़ (nukta forms)
DEV_VIRAMA = "्"
TAM_VOWEL = (0x0B85, 0x0B94)            # அ … ஔ
TAM_CONS = (0x0B95, 0x0BB9)             # க … ஹ
TAM_PULLI = "்"

# Hesitation sounds as an ASR usually writes them. Conservative on purpose: a word only
# counts as a filler if it is exactly one of these (after stripping punctuation).
FILLERS = {
    "hi": {"हम्म", "हम", "हूँ", "हूं", "उम्म", "उम", "अं", "आं", "एँ", "एं", "अँ", "हँ", "हं"},
    "ta": {"ம்", "ம்ம்", "ஹ்ம்", "ஆ", "அ", "ஏ", "ஓ", "ஊ", "ஆஆ"},
}
def bare(word: str) -> str:
    """Letters and marks only: drops spaces, Latin punctuation, the danda (।॥) and digits,
    while keeping matras and the virama / pulli, which are combining marks."""
    return "".join(c for c in unicodedata.normalize("NFC", word) if unicodedata.category(c)[0] in "LM")


def _in(ch: str, lo_hi) -> bool:
    return lo_hi[0] <= ord(ch) <= lo_hi[1]


def syllables(word: str, language: str | None = None) -> int:
    """Syllables in one Devanagari or Tamil word (0 if it is neither script)."""
    w = bare(word)
    if not w:
        return 0
    n, last_cons = 0, -1
    for i, ch in enumerate(w):
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if _in(ch, DEV_VOWEL) or _in(ch, TAM_VOWEL):
            n += 1
        elif _in(ch, DEV_CONS) or _in(ch, DEV_CONS_EXTRA) or _in(ch, TAM_CONS):
            if nxt not in (DEV_VIRAMA, TAM_PULLI):
                n += 1
                last_cons = i
    if n and last_cons == len(w) - 1 and (_in(w[last_cons], DEV_CONS) or _in(w[last_cons], DEV_CONS_EXTRA)):
        n -= 1                                   # Hindi word-final schwa deletion
    return max(n, 1) if any(_in(c, DEV_VOWEL) or _in(c, DEV_CONS) or _in(c, DEV_CONS_EXTRA)
                            or _in(c, TAM_VOWEL) or _in(c, TAM_CONS) for c in w) else 0


LATIN = (0x41, 0x7A)


def script_of(word: str) -> str:
    """Which script a word is written in, by its first letter: the cheap, reliable way to spot
    an English insertion in Hindi or Tamil speech."""
    for ch in bare(word):
        if _in(ch, DEV_VOWEL) or _in(ch, DEV_CONS) or _in(ch, DEV_CONS_EXTRA):
            return "devanagari"
        if _in(ch, TAM_VOWEL) or _in(ch, TAM_CONS):
            return "tamil"
        if _in(ch.upper(), LATIN):
            return "latin"
    return "other"


def latin_syllables(word: str) -> int:
    """Syllables in a Latin-script word, by vowel groups: "WhatsApp" 2, "phone" 1, "battery" 3.
    A rough count, but the Devanagari and Tamil rules return 0 for a Latin word, and a word
    counted as 0 syllables drags the whole rate down -- code-switching is normal in both
    languages, so those words have to count for something."""
    w = "".join(c for c in bare(word).lower() if _in(c, LATIN))
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1 and not re.search(r"[aeiouy]{2}e$|le$", w):
        n -= 1                                   # silent final e: "phone" 1, not 2
    return max(n, 1)


def words(text: str, language: str | None = None) -> list[dict]:
    """-> [{"text", "syllables", "filler", "script"}] for the words of `text` in script order."""
    fillers = FILLERS.get(language or "", set())
    out = []
    for w in text.split():
        b = bare(w)
        if not b:
            continue
        sc = script_of(b)
        n = latin_syllables(b) if sc == "latin" else syllables(b, language)
        out.append({"i": len(out) + 1, "text": b, "roman": romanize(b), "syllables": n,
                    "filler": b in fillers, "script": sc})
    return out


# ------------------------------------------------------------------ romanisation

# Plain ASCII, no diacritics: a long vowel is doubled (aa, ii, uu, ee, oo). Not a scholarly
# transliteration -- enough that an English reader, and a generator holding the native-script
# text, can identify which word is meant.
DEV_INDEP = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ऋ": "ri",
             "ऌ": "li", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऍ": "e", "ऑ": "o"}
DEV_MATRA = {"ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri", "ॄ": "ri",
             "ॆ": "e", "े": "e", "ै": "ai", "ॊ": "o", "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o"}
DEV_CONSONANT = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch", "छ": "chh", "ज": "j",
    "झ": "jh", "ञ": "ny", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t",
    "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
    "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v", "श": "sh", "ष": "sh",
    "स": "s", "ह": "h", "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh",
    "फ़": "f", "य़": "y"}
TAM_INDEP = {"அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ii", "உ": "u", "ஊ": "uu", "எ": "e",
             "ஏ": "ee", "ஐ": "ai", "ஒ": "o", "ஓ": "oo", "ஔ": "au"}
TAM_MATRA = {"ா": "aa", "ி": "i", "ீ": "ii", "ு": "u", "ூ": "uu", "ெ": "e", "ே": "ee",
             "ை": "ai", "ொ": "o", "ோ": "oo", "ௌ": "au"}
TAM_CONSONANT = {
    "க": "k", "ங": "ng", "ச": "ch", "ஞ": "ny", "ட": "t", "ண": "n", "த": "th", "ந": "n",
    "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "zh", "ள": "l",
    "ற": "r", "ன": "n", "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h"}
MARKS = {"ं": "n", "ँ": "n", "ः": "h", "ஂ": "n", "ஃ": "kh"}


def romanize(word: str) -> str:
    """A Devanagari or Tamil word in Latin letters. Both scripts are abugidas: a consonant
    carries an inherent "a" unless a vowel sign or the virama / pulli follows it, which is the
    same walk `syllables()` makes. A word already in Latin script comes back unchanged."""
    w = bare(word)
    if not w:
        return ""
    out, i = [], 0
    while i < len(w):
        ch, nxt = w[i], (w[i + 1] if i + 1 < len(w) else "")
        if ch in MARKS:
            out.append(MARKS[ch])
        elif ch in DEV_INDEP or ch in TAM_INDEP:
            out.append(DEV_INDEP.get(ch) or TAM_INDEP[ch])
        elif ch in DEV_CONSONANT or ch in TAM_CONSONANT:
            out.append(DEV_CONSONANT.get(ch) or TAM_CONSONANT[ch])
            if nxt in (DEV_VIRAMA, TAM_PULLI):
                i += 2                                  # vowel removed
                continue
            if nxt in DEV_MATRA or nxt in TAM_MATRA:
                out.append(DEV_MATRA.get(nxt) or TAM_MATRA[nxt])
                i += 2
                continue
            out.append("a")                             # the inherent vowel
        elif _in(ch.upper(), LATIN):
            out.append(ch)
        i += 1
    r = "".join(out)
    # Hindi drops the word-final inherent "a" (ghar, not ghara); Tamil keeps its vowel
    if script_of(w) == "devanagari" and r.endswith("a") and len(r) > 2:
        r = r[:-1]
    return r


def code_switching(word_list: list[dict]) -> dict | None:
    """English (Latin-script) words inside Hindi or Tamil speech. Script alone decides this, so
    it is as reliable as the transcript -- no model and no judgement involved."""
    if not word_list:
        return None
    latin = [w for w in word_list if w.get("script") == "latin"]
    if not latin:
        return None
    return {"latin_words": len(latin), "ratio": len(latin) / len(word_list),
            "words": [w["text"] for w in latin][:12],
            "first_s": next((w["start"] for w in latin if w.get("start") is not None), None)}


def repeats(word_list: list[dict], max_gap_s: float = 1.0) -> list[dict]:
    """Adjacent repeated words -- "main main", "அது அது" -- the kind of disfluency the
    Morphology axis asks about. Only an identical repeated token counts, so it never guesses
    at a stutter it cannot see."""
    out = []
    for a, b in zip(word_list, word_list[1:]):
        if a["text"] != b["text"] or a.get("filler"):
            continue
        if (a.get("end") is not None and b.get("start") is not None
                and b["start"] - a["end"] > max_gap_s):
            continue
        out.append({"i": a.get("i"), "text": a["text"], "roman": a.get("roman"),
                    "start": a.get("start"), "end": b.get("end")})
    return out


def align_words(word_list: list[dict], nuclei: np.ndarray, phrases, pad_s: float = 0.04,
                max_mismatch: float = 0.3) -> list[dict] | None:
    """Lay the words' syllable counts over the detected nuclei, in order.

    Returns the words with "start"/"end" when the text's syllable total and the number of
    nuclei inside the phrases agree within max_mismatch; otherwise None (counts only), because
    stretched times from a bad match would be wrong, not merely vague."""
    from .dsp import in_spans
    inside = np.sort(np.asarray(nuclei)[in_spans(np.asarray(nuclei), list(phrases))]) if len(nuclei) else np.zeros(0)
    total = sum(w["syllables"] for w in word_list)
    if not total or not len(inside) or abs(len(inside) - total) / total > max_mismatch:
        return None
    # scale the text's syllable positions onto the nuclei we actually have
    scale = len(inside) / total
    out, at = [], 0
    for w in word_list:
        k = max(1, int(round(w["syllables"] * scale))) if w["syllables"] else 0
        take = inside[at:at + k]
        at += k
        if len(take):
            out.append({**w, "start": float(max(0.0, take[0] - pad_s)), "end": float(take[-1] + pad_s)})
        else:
            out.append({**w, "start": None, "end": None})
    return out


def longest_word(word_list: list[dict], min_ratio: float = 1.8) -> dict | None:
    """The word whose seconds-per-syllable is furthest above the median, if clearly so
    (>= min_ratio). "the last word is drawn out" in the Rime examples."""
    rates = [((w["end"] - w["start"]) / w["syllables"], w) for w in word_list
             if w.get("start") is not None and w["syllables"]]
    if len(rates) < 4:
        return None
    med = float(np.median([r for r, _ in rates]))
    r, w = max(rates, key=lambda x: x[0])
    return {**w, "seconds_per_syllable": r, "ratio": r / med} if med > 0 and r / med >= min_ratio else None


def load_transcripts(spec: dict | None) -> dict:
    """{"file": map.json} -> {key: text}; keys may be paths, relative paths or file stems.
    {"dir": folder} is handled per clip by `text_for`."""
    if not spec or "file" not in spec:
        return {}
    data = json.loads(Path(spec["file"]).expanduser().read_text(encoding="utf-8"))
    if isinstance(data, list):                   # [{"audio_path": ..., "text": ...}, ...]
        key = "audio_path" if data and "audio_path" in data[0] else "path"
        field = next((f for f in ("text", "transcript", "sentence") if data and f in data[0]), "text")
        data = {r[key]: r[field] for r in data if r.get(key)}
    return {str(k): str(v) for k, v in data.items()}


def text_for(spec: dict | None, table: dict, path: str | None) -> str | None:
    """The transcript of one clip: from the loaded table (by path, name or stem) or from
    <dir>/<stem>.txt."""
    if not spec or not path:
        return None
    p = Path(path)
    parts = p.as_posix().split("/")
    tails = ["/".join(parts[i:]) for i in range(len(parts))]          # full path -> file name
    for key in [str(p)] + tails + [p.stem]:
        if key in table:
            return table[key]
    if spec.get("dir"):
        f = Path(spec["dir"]).expanduser() / (p.stem + ".txt")
        if f.is_file():
            return f.read_text(encoding="utf-8").strip()
    return None
