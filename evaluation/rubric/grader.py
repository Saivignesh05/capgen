"""The rubric grader: one question per rubric item, answered by a language model in one word
(docs/evaluation_plan.md, section 3.1).

This file holds everything that does not need the model: cutting the description to what the
grader may read, the frozen prompt, strict parsing of the answer, and the loop over items. The
model itself is behind GraderInterface; the sarvam-m implementation is in grader_sarvam.py.
"""
from abc import ABC, abstractmethod

from evaluation.rubric.schema import GradedItem, RubricItem, Verdict
from evaluation.settings import GraderOptions

VERDICTS: tuple[Verdict, ...] = ("STATED", "CONTRADICTED", "NOT_MENTIONED")

# The frozen grader prompt. Ours: the official prompt is not published. Changing a word changes
# every score, so a change here is a new evaluation version. CAREFUL G1
PROMPT = """You check whether an audio description supports a statement.
Description: {description}
Statement: {statement}
{timing}Answer with exactly one word: STATED if the description clearly commits to the statement,
CONTRADICTED if it says something incompatible, NOT_MENTIONED otherwise.
A guess between alternatives is NOT_MENTIONED."""

TIMING_LINE = "For STATED, the description must place it between {start:.2f} s and {end:.2f} s.\n"


class GraderInterface(ABC):
    """Interface for grader models."""

    @abstractmethod
    def answer(self, prompt: str) -> str:
        """Return the model's raw answer to one prompt. No cleaning: the caller parses it."""


def cut_description(description: str, options: GraderOptions) -> str:
    """What the grader reads: the first max_description_chars characters, then at most
    max_description_words words of those."""
    cut = description[:options.max_description_chars]
    words = cut.split()
    if len(words) > options.max_description_words:
        cut = " ".join(words[:options.max_description_words])
    return cut


def description_is_valid(description: str, options: GraderOptions) -> bool:
    """False for a description that scores 0 whatever it says: empty, or not in English. CAREFUL G2"""
    letters = [character for character in description if character.isalpha()]
    if len(letters) == 0:
        return False
    non_ascii = [character for character in letters if not character.isascii()]
    return len(non_ascii) / len(letters) <= options.max_non_ascii_letter_share


def build_prompt(description: str, item: RubricItem, options: GraderOptions) -> str:
    """The prompt for one item. `description` is already cut by cut_description."""
    timing = ""
    if item.start_s is not None:
        timing = TIMING_LINE.format(start=max(0.0, item.start_s - options.timing_collar_seconds),
                                    end=item.end_s + options.timing_collar_seconds)
    return PROMPT.format(description=description, statement=item.statement, timing=timing)


def parse_verdict(raw_answer: str) -> Verdict | None:
    """The verdict in a raw answer, or None if it cannot be parsed. CAREFUL G3

    Strict: the first non-empty line of the answer, with surrounding whitespace removed, must be
    exactly one of the three words. "Stated", "STATED." and "The answer is STATED" are not parsed.
    """
    for line in raw_answer.splitlines():
        line = line.strip()
        if line == "":
            continue
        for verdict in VERDICTS:
            if line == verdict:
                return verdict
        return None
    return None


def grade_description(grader: GraderInterface, description: str, items: list[RubricItem], options: GraderOptions) -> list[GradedItem]:
    """Grade one description against the rubric items of its clip: one model call per item.

    An invalid description gets NOT_MENTIONED for every item without calling the model, which
    scores 0. An answer that cannot be parsed counts as NOT_MENTIONED, with parsed=False.
    """
    graded = []
    if not description_is_valid(description, options):
        for item in items:
            graded.append(GradedItem(item_id=item.item_id, verdict="NOT_MENTIONED", raw_answer="", parsed=True))
        return graded
    cut = cut_description(description, options)
    for item in items:
        raw_answer = grader.answer(build_prompt(cut, item, options))
        verdict = parse_verdict(raw_answer)
        if verdict is None:
            graded.append(GradedItem(item_id=item.item_id, verdict="NOT_MENTIONED", raw_answer=raw_answer, parsed=False))
        else:
            graded.append(GradedItem(item_id=item.item_id, verdict=verdict, raw_answer=raw_answer, parsed=True))
    return graded


def parse_failure_rate(graded: list[GradedItem]) -> float:
    """Share of answers that could not be parsed. Report it with every rubric score."""
    if len(graded) == 0:
        raise ValueError("no graded items")
    failures = [item for item in graded if not item.parsed]
    return len(failures) / len(graded)


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# G1   The prompt follows docs/evaluation_plan.md section 3.1 word for word. The timing line gives
#      the item's span widened by timing_collar_seconds on each side (not below 0 s).
# G2   The plan also asks to score 0 a description that addresses the grader or holds text that is
#      not about the audio. That check is not built: it needs a model or a person. Listed as open
#      in docs/EXPLANATION.md.
# G3   Strict parsing is deliberate: a lenient parser hides a grader that is not following the
#      prompt. If the parse-failure rate is high, fix the model call (thinking mode, chat template),
#      not the parser.
# ---------------------------------------------------------------------------------------------
