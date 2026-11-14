"""The composer: turn a list of claims into one description of at most max_chars characters.

The rubric pays +1 for a right claim that matches an item and -1 for a wrong one, and the grader
reads only the first 1,000 characters. So a claim with probability p of being right is worth about
(2p - 1) x prior, where prior is how likely its item type is to be in the rubric, and the claims to
state are those with the most expected value per character (docs/post_midterm_plan.md, section 3.1).

Deterministic: no model call, no randomness. The same claims and settings give the same text.
"""
from pydantic import BaseModel, ConfigDict

from claims.schema import Claim
from claims.settings import ComposerOptions

SENTENCE_ENDS = (".", "!", "?")


class Composition(BaseModel):
    """A composed description and what happened to every claim."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    used: list[Claim]                   # in the order they appear in the text
    dropped_low_probability: int        # below min_probability
    dropped_duplicate: int              # same sentence as a claim that was kept
    dropped_no_room: int                # did not fit in what was left of the budget


def render(claim: Claim) -> str:
    """The sentence a claim becomes in the description. A timed claim gets its span appended. CAREFUL P1"""
    statement = " ".join(claim.statement.split())
    if claim.start_s is None:
        if statement.endswith(SENTENCE_ENDS):
            return statement
        return statement + "."
    statement = statement.rstrip(".!?")
    return f"{statement} ({claim.start_s:.1f}-{claim.end_s:.1f} s)."


def expected_value(claim: Claim, options: ComposerOptions) -> float:
    """(2p - 1) x the prior of the claim's item type."""
    return (2.0 * claim.probability - 1.0) * options.type_priors[claim.item_type]


def value_per_character(claim: Claim, options: ComposerOptions) -> float:
    """Expected value divided by what the claim costs in the budget: its sentence and one separator."""
    return expected_value(claim, options) / (len(render(claim)) + len(options.separator))


def compose(claims: list[Claim], options: ComposerOptions) -> Composition:
    """Choose claims greedily by expected value per character and join them into one description.

    1. Drop every claim with probability below min_probability.
    2. Of claims that render to the same sentence, keep the one with the highest probability.
    3. Rank by value per character, highest first; equal values keep their input order.
    4. Go down the ranking and add each claim that still fits whole. A claim is never cut.
    The text holds the chosen claims in ranking order, so the most valuable come first.
    """
    confident = [claim for claim in claims if claim.probability >= options.min_probability]
    dropped_low_probability = len(claims) - len(confident)

    best_by_sentence: dict[str, Claim] = {}
    for claim in confident:
        sentence = render(claim)
        if sentence not in best_by_sentence or claim.probability > best_by_sentence[sentence].probability:
            best_by_sentence[sentence] = claim
    unique = [claim for claim in confident if best_by_sentence[render(claim)] is claim]
    dropped_duplicate = len(confident) - len(unique)

    # sorted() is stable: claims with equal value per character stay in input order.
    ranked = sorted(unique, key=lambda claim: -value_per_character(claim, options))

    used = []
    text = ""
    for claim in ranked:
        sentence = render(claim)
        if text == "":
            candidate = sentence
        else:
            candidate = text + options.separator + sentence
        if len(candidate) <= options.max_chars:
            text = candidate
            used.append(claim)
    return Composition(text=text, used=used, dropped_low_probability=dropped_low_probability,
                       dropped_duplicate=dropped_duplicate, dropped_no_room=len(ranked) - len(used))


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# P1   A claim's statement must be complete in itself. A claim that already names its time in the
#      statement ("laughs at 3.2 s") should have null start_s and end_s, or the time is written
#      twice. The rule-based extractor never sets times, for this reason.
# ---------------------------------------------------------------------------------------------
