"""The grader's model-free parts: cutting, the prompt, strict parsing and the grading loop."""
from evaluation.rubric.grader import (GraderInterface, build_prompt, cut_description, description_is_valid,
                                      grade_description, parse_failure_rate, parse_verdict)
from evaluation.rubric.schema import RubricItem
from evaluation.rubric.score import clip_score
from helpers import evaluation_settings, rubric_item

OPTIONS = evaluation_settings().grader


class ScriptedGrader(GraderInterface):
    """Answers from a list, in order, and remembers the prompts it was given."""

    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.prompts = []

    def answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)


def test_cut_keeps_the_first_1000_characters():
    description = "a" * 600 + " " + "b" * 600
    cut = cut_description(description, OPTIONS)
    assert len(cut) == 1000
    assert cut == description[:1000]


def test_cut_keeps_at_most_300_words():
    description = " ".join(["ab"] * 400)       # 1,199 characters, 400 words
    cut = cut_description(description, OPTIONS)
    assert len(cut.split()) == 300
    assert len(cut) <= 1000


def test_short_description_is_not_changed():
    assert cut_description("A calm male voice.", OPTIONS) == "A calm male voice."


def test_validity():
    assert description_is_valid("A calm male voice in a quiet room.", OPTIONS)
    assert not description_is_valid("", OPTIONS)
    assert not description_is_valid("  12.5 !! ", OPTIONS)
    assert not description_is_valid("एक शांत पुरुष आवाज़", OPTIONS)
    assert description_is_valid('A man says "भैया" once, then pauses for a long time.', OPTIONS)


def test_parse_verdict_is_strict():
    assert parse_verdict("STATED") == "STATED"
    assert parse_verdict("  CONTRADICTED \n") == "CONTRADICTED"
    assert parse_verdict("\nNOT_MENTIONED\nbecause the description says nothing") == "NOT_MENTIONED"
    assert parse_verdict("Stated") is None
    assert parse_verdict("STATED.") is None
    assert parse_verdict("The answer is STATED") is None
    assert parse_verdict("NOT MENTIONED") is None
    assert parse_verdict("") is None


def test_prompt_for_a_global_item_has_no_timing_line():
    prompt = build_prompt("A calm voice.", rubric_item("a", True, 1), OPTIONS)
    assert "Description: A calm voice.\nStatement: statement a\nAnswer with exactly one word" in prompt
    assert "must place it" not in prompt


def test_prompt_for_a_timed_item_widens_the_span_by_the_collar():
    item = RubricItem(clip_id="hi/clip_a", item_id="t", axis="sound", item_type="Event", statement="The speaker laughs",
                      truth=True, start_s=3.2, end_s=3.9, weight=1, provenance="human:test")
    prompt = build_prompt("A laugh.", item, OPTIONS)
    assert "the description must place it between 2.95 s and 4.15 s." in prompt
    early = RubricItem(clip_id="hi/clip_a", item_id="t", axis="sound", item_type="Event", statement="A click",
                       truth=True, start_s=0.1, end_s=0.2, weight=1, provenance="human:test")
    assert "between 0.00 s and 0.45 s." in build_prompt("A click.", early, OPTIONS)


def test_grading_loop_scores_and_counts_parse_failures():
    items = [rubric_item("a", True, 1), rubric_item("b", True, 1), rubric_item("x", False, 1)]
    grader = ScriptedGrader(["STATED", "I think it is stated", "STATED"])
    graded = grade_description(grader, "A calm male voice.", items, OPTIONS)
    assert [g.verdict for g in graded] == ["STATED", "NOT_MENTIONED", "STATED"]
    assert [g.parsed for g in graded] == [True, False, True]
    assert graded[1].raw_answer == "I think it is stated"
    assert parse_failure_rate(graded) == 1 / 3
    verdicts = {g.item_id: g.verdict for g in graded}
    assert clip_score(items, verdicts, use_weights=False) == 0.0      # (1 + 0 - 1) / 2


def test_grader_sees_only_the_cut_description():
    grader = ScriptedGrader(["STATED"])
    description = "x" * 999 + " SECRET_AFTER_THE_LIMIT"
    grade_description(grader, description, [rubric_item("a", True, 1)], OPTIONS)
    assert "SECRET_AFTER_THE_LIMIT" not in grader.prompts[0]


def test_invalid_description_is_not_sent_to_the_model():
    grader = ScriptedGrader([])
    items = [rubric_item("a", True, 1), rubric_item("x", False, 1)]
    graded = grade_description(grader, "", items, OPTIONS)
    assert grader.prompts == []
    assert [g.verdict for g in graded] == ["NOT_MENTIONED", "NOT_MENTIONED"]
    assert clip_score(items, {g.item_id: g.verdict for g in graded}, use_weights=False) == 0.0
