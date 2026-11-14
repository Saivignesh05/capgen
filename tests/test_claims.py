"""The claim model, the composer and the rule-based extractor."""
import pytest

from claims.composer import compose, expected_value, render
from claims.extractor import RuleBasedExtractor, split_sentences
from claims.schema import Claim
from claims.settings import ComposerOptions
from helpers import claims_settings

SETTINGS = claims_settings()
OPTIONS = SETTINGS.composer


def claim(statement: str, probability: float, item_type: str) -> Claim:
    return Claim(item_type=item_type, statement=statement, start_s=None, end_s=None, source="model", probability=probability)


def options_with(max_chars: int, speaker_prior: float) -> ComposerOptions:
    priors = dict(OPTIONS.type_priors)
    priors["Speaker"] = speaker_prior
    return ComposerOptions(max_chars=max_chars, min_probability=OPTIONS.min_probability, separator=OPTIONS.separator, type_priors=priors)


# ---- the claim model ---------------------------------------------------------------------------

def test_claim_checks_its_fields():
    with pytest.raises(ValueError):
        claim("A male voice.", 1.2, "Speaker")
    with pytest.raises(ValueError):
        claim("   ", 0.9, "Speaker")
    with pytest.raises(ValueError):
        claim("A male voice.", 0.9, "Gender")
    with pytest.raises(ValueError):
        Claim(item_type="Event", statement="A laugh", start_s=3.0, end_s=None, source="tool", probability=0.9)
    with pytest.raises(ValueError):
        Claim(item_type="Event", statement="A laugh", start_s=3.0, end_s=2.0, source="tool", probability=0.9)


# ---- the composer ------------------------------------------------------------------------------

def test_render():
    assert render(claim("A male voice", 0.9, "Speaker")) == "A male voice."
    assert render(claim("A  male\nvoice.", 0.9, "Speaker")) == "A male voice."
    timed = Claim(item_type="Event", statement="The speaker laughs.", start_s=3.2, end_s=3.94, source="tool", probability=0.9)
    assert render(timed) == "The speaker laughs (3.2-3.9 s)."


def test_expected_value_is_2p_minus_1_times_the_prior():
    assert expected_value(claim("A male voice.", 0.9, "Speaker"), options_with(1000, 0.5)) == pytest.approx(0.4)
    assert expected_value(claim("A male voice.", 0.5, "Speaker"), OPTIONS) == 0.0


def test_claims_below_the_threshold_are_dropped():
    composition = compose([claim("A male voice.", 0.9, "Speaker"), claim("Perhaps a phone call.", 0.4, "Channel")], OPTIONS)
    assert composition.text == "A male voice."
    assert composition.dropped_low_probability == 1


def test_ranking_by_value_per_character():
    # Same length (13 characters): the more probable claim comes first.
    likely = claim("A male voice.", 0.95, "Speaker")
    less_likely = claim("A calm voice.", 0.70, "Morphology")
    assert compose([less_likely, likely], OPTIONS).text == "A male voice. A calm voice."
    # Same probability: the shorter claim has more value per character and comes first.
    short = claim("A male voice.", 0.9, "Speaker")
    long = claim("The recording was made in a small furnished room.", 0.9, "Space")
    assert compose([long, short], OPTIONS).used == [short, long]
    # A low prior moves a claim down.
    assert compose([short, long], options_with(1000, 0.1)).used == [long, short]


def test_never_longer_than_the_budget_and_never_cuts_a_claim():
    claims = [claim(f"Fact number {number} about the voice in this recording is stated here.", 0.9, "Speaker") for number in range(60)]
    composition = compose(claims, OPTIONS)
    assert len(composition.text) <= OPTIONS.max_chars
    assert composition.text == OPTIONS.separator.join(render(used) for used in composition.used)
    assert composition.dropped_no_room == 60 - len(composition.used)
    assert composition.dropped_no_room > 0


def test_a_claim_that_does_not_fit_is_skipped_and_smaller_ones_still_enter():
    # value per character: big 0.98 / 41 = 0.0239, medium 0.60 / 31 = 0.0194, small 0.20 / 11 = 0.0182
    big = claim("B" * 39 + ".", 0.99, "Speaker")        # 40 characters, ranked first
    medium = claim("M" * 29 + ".", 0.8, "Speaker")      # 30 characters: 40 + 1 + 30 does not fit in 60
    small = claim("S" * 9 + ".", 0.6, "Speaker")        # 10 characters: 40 + 1 + 10 fits
    composition = compose([big, medium, small], options_with(60, 1.0))
    assert composition.used == [big, small]
    assert len(composition.text) <= 60
    assert composition.dropped_no_room == 1


def test_a_claim_longer_than_the_whole_budget_gives_an_empty_text():
    composition = compose([claim("x" * 80, 0.9, "Speaker")], options_with(60, 1.0))
    assert composition.text == ""
    assert composition.dropped_no_room == 1


def test_duplicates_are_stated_once():
    composition = compose([claim("A male voice.", 0.6, "Speaker"), claim("A male voice", 0.9, "Speaker")], OPTIONS)
    assert composition.text == "A male voice."
    assert composition.used[0].probability == 0.9
    assert composition.dropped_duplicate == 1


def test_composer_is_deterministic():
    claims = [claim(f"Claim {number} of equal length.", 0.8, "Speaker") for number in range(10)]
    assert compose(claims, OPTIONS) == compose(claims, OPTIONS)
    assert compose(claims, OPTIONS).used == claims      # equal value per character: input order is kept


def test_no_claims_gives_an_empty_text():
    composition = compose([], OPTIONS)
    assert composition.text == ""
    assert composition.used == []


# ---- the rule-based extractor ------------------------------------------------------------------

CAPTION = """**Speaker:** A young adult male speaks Hindi at a median pitch near 143 Hz.
- The recording is clean, with an SNR of about 23 dB.
He laughs briefly at 3.2 s. Ok.

The delivery is warm and unhurried!"""


def test_split_sentences_keeps_decimals_and_removes_markdown():
    assert split_sentences(CAPTION) == [
        "Speaker: A young adult male speaks Hindi at a median pitch near 143 Hz.",
        "The recording is clean, with an SNR of about 23 dB.",
        "He laughs briefly at 3.2 s.",
        "Ok.",
        "The delivery is warm and unhurried!",
    ]


def test_extractor_types_sentences_by_keyword():
    claims = RuleBasedExtractor(SETTINGS.extractor).extract(CAPTION)
    assert [c.item_type for c in claims] == ["Pitch", "Other sounds", "Event", "Morphology"]     # "Ok." is too short
    assert all(c.probability == SETTINGS.extractor.probability for c in claims)
    assert all(c.source == "model" and c.start_s is None for c in claims)
    assert claims[2].statement == "He laughs briefly at 3.2 s."


def test_extractor_uses_whole_words_and_the_fallback_type():
    extractor = RuleBasedExtractor(SETTINGS.extractor)
    assert extractor.extract("The manner of speaking is ordinary.")[0].item_type == SETTINGS.extractor.fallback_type
    assert extractor.extract("A man speaks in the clip here.")[0].item_type == "Speaker"


def test_extract_then_compose_fits_the_budget():
    caption = " ".join(f"Sentence {number} describes the male voice in some detail." for number in range(80))
    composition = compose(RuleBasedExtractor(SETTINGS.extractor).extract(caption), OPTIONS)
    assert 900 < len(composition.text) <= 1000
    assert composition.text.endswith(".")
