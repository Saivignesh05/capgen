"""Text normalisation shared by every intelligibility measure, and the character error rate.

The same normalisation is applied to the target text and to the ASR output, whichever ASR made it
(docs/evaluation_plan.md, section 4.1): Unicode NFC, lowercase, punctuation and symbols removed,
whitespace collapsed.
"""
import unicodedata


def normalise(text: str) -> str:
    """NFC, lowercase, no punctuation or symbols, single spaces. CAREFUL N1"""
    text = unicodedata.normalize("NFC", text).lower()
    kept = []
    for character in text:
        category = unicodedata.category(character)
        if category.startswith("P") or category.startswith("S"):
            kept.append(" ")        # a space, so that "a-b" does not become "ab"
        else:
            kept.append(character)
    return " ".join("".join(kept).split())


def edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance between two strings, counted in characters (Unicode code points)."""
    previous = list(range(len(hypothesis) + 1))
    for i in range(1, len(reference) + 1):
        current = [i]
        for j in range(1, len(hypothesis) + 1):
            substitution = previous[j - 1] + (0 if reference[i - 1] == hypothesis[j - 1] else 1)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitution))
        previous = current
    return previous[len(hypothesis)]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER of `hypothesis` against `reference`, both normalised here. Can be above 1."""
    reference = normalise(reference)
    hypothesis = normalise(hypothesis)
    if len(reference) == 0:
        raise ValueError("the reference text is empty after normalisation")
    return edit_distance(reference, hypothesis) / len(reference)


def intelligibility_value(cer: float) -> float:
    """The value reported for intelligibility: 1 - CER, floored at 0."""
    return max(0.0, 1.0 - cer)


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# N1   Vowel signs and the virama of Devanagari and Tamil are Unicode marks (category M), not
#      punctuation, so they are kept. Characters are code points, not grapheme clusters: one
#      akshara can count as several characters. Both ASRs are scored the same way, so they compare.
# ---------------------------------------------------------------------------------------------
