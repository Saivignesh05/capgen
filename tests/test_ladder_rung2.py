"""Ladder rung 2 from start to finish on a synthetic round 0 run, with a grader that needs no model."""
import json
import logging

import pytest

pytest.importorskip("torch", reason="scripts/ladder_rung2.py imports the sarvam-m grader, which needs torch")
pytest.importorskip("transformers", reason="scripts/ladder_rung2.py imports the sarvam-m grader, which needs transformers")

from evaluation.captioner_eval import Description, grade_descriptions  # noqa: E402
from evaluation.rubric.grader import GraderInterface  # noqa: E402
from evaluation.rubric.schema import items_by_clip, read_rubric  # noqa: E402
from helpers import claims_settings, evaluation_settings  # noqa: E402
from scripts.ladder_rung2 import score_and_report  # noqa: E402

LOG = logging.getLogger("test_ladder_rung2")


class LiteralGrader(GraderInterface):
    """STATED if the statement appears word for word in the description it is shown, else NOT_MENTIONED."""

    def __init__(self):
        self.calls = 0

    def answer(self, prompt: str) -> str:
        self.calls += 1
        description = prompt.split("Description: ", 1)[1].split("\nStatement: ", 1)[0]
        statement = prompt.split("\nStatement: ", 1)[1].split("\n", 1)[0]
        if statement in description:
            return "STATED"
        return "NOT_MENTIONED"


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rubric_row(clip_id: str, item_id: str, statement: str, truth: bool) -> dict:
    return {"clip_id": clip_id, "item_id": item_id, "axis": "sound", "item_type": "Speaker", "statement": statement,
            "truth": truth, "start_s": None, "end_s": None, "weight": 1, "provenance": "human:test"}


@pytest.fixture
def synthetic_run(tmp_path):
    """A round 0 run of two clips and their rubric.

    hi/c1: the true fact "A man speaks." comes after 1,200 characters of long filler sentences, so
           the grader never sees it in the round 0 caption. Re-composed, the short sentence ranks
           first and is seen.                                  round_0: 0    recomposed: 1
    ta/c2: a short caption that states its fact.               round_0: 1    recomposed: 1
    """
    run = tmp_path / "run"
    run.mkdir()
    filler = " ".join(["This long sentence of filler says nothing that any rubric item asks about at all."] * 15)
    assert len(filler) > 1000
    write_jsonl(run / "clips.jsonl", [{"clip_id": "c1", "path": "/data/devset/hi/clips/c1.wav"},
                                      {"clip_id": "c2", "path": "/data/devset/ta/clips/c2.wav"},
                                      {"clip_id": "c3", "path": "/data/devset/ta/clips/c3.wav"}])
    write_jsonl(run / "descriptions.jsonl", [{"clip_id": "c1", "description": filler + " A man speaks."},
                                             {"clip_id": "c2", "description": "A woman speaks."},
                                             {"clip_id": "c3", "description": "No rubric for this clip."}])
    rubric = tmp_path / "rubric.jsonl"
    write_jsonl(rubric, [rubric_row("hi/c1", "hi/c1#1", "A man speaks.", True),
                         rubric_row("hi/c1", "hi/c1#2", "A child speaks.", False),
                         rubric_row("ta/c2", "ta/c2#1", "A woman speaks.", True)])
    output = tmp_path / "output"
    output.mkdir()
    return str(run), str(rubric), str(output)


def test_rung2_scores_both_systems_and_the_paired_difference(synthetic_run):
    run, rubric, output = synthetic_run
    grader = LiteralGrader()
    summary = score_and_report(grader, evaluation_settings(), claims_settings(), run, rubric, output, LOG)

    by_system = {entry["system"]: entry for entry in summary["systems"] if entry["measure"] == "rubric_score"}
    assert by_system["round_0"]["by_language"] == {"hi": 0.0, "ta": 1.0}
    assert by_system["round_0"]["overall"]["value"] == 0.5
    assert by_system["round_0_recomposed"]["by_language"] == {"hi": 1.0, "ta": 1.0}
    assert by_system["round_0_recomposed"]["parse_failure_rate"] == 0.0

    comparison = [entry for entry in summary["comparisons"] if entry["measure"] == "rubric_score"][0]
    assert comparison["difference"]["value"] == 0.5        # one clip per language: every draw is the same
    assert comparison["difference"]["low"] == 0.5
    assert comparison["gap_closed_level"] == 4              # g = 0.5 / (1.0 - 0.5) = 1

    with open(f"{output}/results.csv", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines[0] == "track,part,language,system,measure,n_items,value"
    assert "2,captioner,hi,round_0,rubric_score,1,0.000000" in lines
    assert "2,captioner,hi,round_0_recomposed,rubric_score,1,1.000000" in lines

    with open(f"{output}/compositions.jsonl", encoding="utf-8") as f:
        compositions = [json.loads(line) for line in f]
    assert [row["clip_id"] for row in compositions] == ["hi/c1", "ta/c2"]       # c3 has no rubric
    assert compositions[0]["round_0_recomposed"].startswith("A man speaks.")
    assert all(row["recomposed_chars"] <= 1000 for row in compositions)


def test_rung2_resumes_without_grading_again(synthetic_run):
    run, rubric, output = synthetic_run
    first = LiteralGrader()
    summary = score_and_report(first, evaluation_settings(), claims_settings(), run, rubric, output, LOG)
    assert first.calls == 6        # 3 rubric items x 2 systems
    second = LiteralGrader()
    assert score_and_report(second, evaluation_settings(), claims_settings(), run, rubric, output, LOG) == summary
    assert second.calls == 0


def test_gradings_of_another_description_are_refused(synthetic_run, tmp_path):
    _, rubric, _ = synthetic_run
    rubric_by_clip = items_by_clip(read_rubric(rubric))
    gradings = str(tmp_path / "gradings.jsonl")
    options = evaluation_settings().grader
    grade_descriptions(LiteralGrader(), [Description(system="round_0", clip_id="ta/c2", text="A woman speaks.")], rubric_by_clip, options, gradings, LOG)
    with pytest.raises(ValueError, match="another description"):
        grade_descriptions(LiteralGrader(), [Description(system="round_0", clip_id="ta/c2", text="A man speaks.")], rubric_by_clip, options, gradings, LOG)
