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


def words(text: str, language: str | None = None) -> list[dict]:
    """-> [{"text", "syllables", "filler"}] for the words of `text` in script order."""
    fillers = FILLERS.get(language or "", set())
    out = []
    for w in text.split():
        b = bare(w)
        if not b:
            continue
        out.append({"text": b, "syllables": syllables(b, language), "filler": b in fillers})
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
