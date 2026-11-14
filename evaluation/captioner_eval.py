"""Captioner evaluation: grade the descriptions of one or more systems against a rubric, and
summarise the rubric score per system with confidence intervals (docs/evaluation_plan.md, section 3.1).

Grading is the slow part (one model call per rubric item), so it is resumable: every graded clip
is appended to a JSONL file, and a second call grades only what is missing.
"""
import hashlib
import json
import logging
import os

from pydantic import BaseModel, ConfigDict

from evaluation.rubric.grader import GraderInterface, grade_description
from evaluation.rubric.schema import RubricItem, Verdict, clip_language
from evaluation.rubric.score import clip_score
from evaluation.settings import BootstrapOptions, GraderOptions
from evaluation.stats import Interval, ValuesByLanguage, confidence_interval, mean, paired_difference


class Description(BaseModel):
    """What one system wrote about one clip."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    system: str         # e.g. "round_0"
    clip_id: str        # the rubric's clip id, "<language>/<name>"
    text: str           # empty if the system produced nothing: that scores 0


class GradingRecord(BaseModel):
    """One line of the gradings file: the grader's verdict on one item for one system's description."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    system: str
    clip_id: str
    description_sha256: str     # of the description text that was graded
    item_id: str
    verdict: Verdict
    raw_answer: str
    parsed: bool


class SystemSummary(BaseModel):
    """The rubric score of one system."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    system: str
    use_weights: bool
    by_language: dict[str, float]       # mean clip score per language
    n_clips_by_language: dict[str, int]
    overall: Interval                   # mean of the language means, with its confidence interval
    parse_failure_rate: float


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_gradings(path: str) -> list[GradingRecord]:
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(GradingRecord.model_validate(json.loads(line)))
    return records


def grade_descriptions(grader: GraderInterface, descriptions: list[Description], rubric_by_clip: dict[str, list[RubricItem]],
                       options: GraderOptions, output_path: str, log: logging.Logger) -> list[GradingRecord]:
    """Grade every description against the rubric of its clip, appending to `output_path`.

    A clip already graded for a system is skipped. If its description has changed since, that is an
    error: delete the gradings file (or use a new output folder) to grade again. A clip is written
    only when all its items are graded, so an interrupted run never leaves half a clip.
    """
    records = read_gradings(output_path)
    graded_sha: dict[tuple[str, str], str] = {}
    for record in records:
        graded_sha[(record.system, record.clip_id)] = record.description_sha256

    todo = []
    for description in descriptions:
        key = (description.system, description.clip_id)
        if key not in graded_sha:
            todo.append(description)
        elif graded_sha[key] != text_sha256(description.text):
            raise ValueError(f"{output_path} holds gradings of another description for {key}. "
                             f"Delete the file to grade again.")
    log.info("grading: %d descriptions to grade, %d already done", len(todo), len(descriptions) - len(todo))

    for number, description in enumerate(todo, 1):
        items = rubric_by_clip[description.clip_id]
        graded = grade_description(grader, description.text, items, options)
        new_records = []
        for graded_item in graded:
            new_records.append(GradingRecord(
                system=description.system, clip_id=description.clip_id, description_sha256=text_sha256(description.text),
                item_id=graded_item.item_id, verdict=graded_item.verdict, raw_answer=graded_item.raw_answer, parsed=graded_item.parsed))
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(record.model_dump(), ensure_ascii=False) + "\n" for record in new_records))
        records.extend(new_records)
        unparsed = len([record for record in new_records if not record.parsed])
        log.info("[grade %d/%d] %s %s: %d items, %d unparsed", number, len(todo), description.system, description.clip_id, len(new_records), unparsed)
    return records


def clip_scores(records: list[GradingRecord], system: str, rubric_by_clip: dict[str, list[RubricItem]], use_weights: bool) -> ValuesByLanguage:
    """{language: {clip_id: rubric score}} of one system."""
    verdicts_by_clip: dict[str, dict[str, Verdict]] = {}
    for record in records:
        if record.system == system:
            verdicts_by_clip.setdefault(record.clip_id, {})[record.item_id] = record.verdict
    scores: ValuesByLanguage = {}
    for clip_id in sorted(verdicts_by_clip):
        score = clip_score(rubric_by_clip[clip_id], verdicts_by_clip[clip_id], use_weights)
        scores.setdefault(clip_language(clip_id), {})[clip_id] = score
    return scores


def summarise_system(records: list[GradingRecord], system: str, rubric_by_clip: dict[str, list[RubricItem]],
                     use_weights: bool, measure: str, options: BootstrapOptions) -> SystemSummary:
    scores = clip_scores(records, system, rubric_by_clip, use_weights)
    if len(scores) == 0:
        raise ValueError(f"no gradings for system {system}")
    system_records = [record for record in records if record.system == system]
    unparsed = [record for record in system_records if not record.parsed]
    return SystemSummary(
        system=system, use_weights=use_weights,
        by_language={language: mean(list(by_clip.values())) for language, by_clip in scores.items()},
        n_clips_by_language={language: len(by_clip) for language, by_clip in scores.items()},
        overall=confidence_interval(scores, measure, options),
        parse_failure_rate=len(unparsed) / len(system_records))


def compare_systems(records: list[GradingRecord], system: str, baseline: str, rubric_by_clip: dict[str, list[RubricItem]],
                    use_weights: bool, measure: str, options: BootstrapOptions) -> Interval:
    """system minus baseline on the same clips, with a paired confidence interval."""
    return paired_difference(clip_scores(records, system, rubric_by_clip, use_weights),
                             clip_scores(records, baseline, rubric_by_clip, use_weights), measure, options)
