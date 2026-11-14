"""Rubric items and grader verdicts (docs/evaluation_plan.md, section 2.1).

A rubric is a JSONL file, one RubricItem per line. Every field is required: write null for a time
that does not apply. The clip's language is the first part of clip_id ("hi/..." or "ta/...").
"""
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator, model_validator

Axis = Literal["sound", "ecology", "intent"]
ItemType = Literal["Speaker", "Source", "Morphology", "Pitch", "Timing", "Event", "Channel", "Space",
                   "Other sounds", "Coupling", "Addressee", "Subtext", "Change"]
Verdict = Literal["STATED", "CONTRADICTED", "NOT_MENTIONED"]


class RubricItem(BaseModel):
    """One statement about a clip that a description may state, contradict or not mention."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    clip_id: str            # "<language>/<name>", e.g. "hi/own_0007"
    item_id: str            # unique, e.g. "hi/own_0007#05"
    axis: Axis
    item_type: ItemType
    statement: str
    truth: bool             # false items are plausible but wrong: stating one costs a point
    start_s: float | None   # for a timed item, where it happens in the clip; null for a global item
    end_s: float | None
    weight: PositiveInt     # difficulty: 1 easy, 2 medium, 3 hard. The unweighted score ignores it
    provenance: str         # "human:<annotator code>", "tool:<name>@<checkpoint sha>" or "transform:<id>"

    @field_validator("clip_id")
    @classmethod
    def clip_id_must_start_with_the_language(cls, clip_id: str) -> str:
        if not (clip_id.startswith("hi/") or clip_id.startswith("ta/")):
            raise ValueError(f"clip_id must start with hi/ or ta/, got: {clip_id}")
        return clip_id

    @field_validator("statement")
    @classmethod
    def statement_must_not_be_empty(cls, statement: str) -> str:
        if not statement.strip():
            raise ValueError("statement is empty")
        return statement

    @model_validator(mode="after")
    def times_must_be_a_span_or_absent(self) -> "RubricItem":
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError(f"{self.item_id}: start_s and end_s must both be set or both be null")
        if self.start_s is not None and self.end_s < self.start_s:
            raise ValueError(f"{self.item_id}: end_s is before start_s")
        return self


class GradedItem(BaseModel):
    """The grader's verdict on one item for one description."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    verdict: Verdict
    raw_answer: str         # what the grader model wrote, untouched
    parsed: bool            # false: the answer was not one of the three words, counted as NOT_MENTIONED


def clip_language(clip_id: str) -> str:
    """The language of a clip: the part of its id before the first slash."""
    return clip_id.split("/")[0]


def read_rubric(path: str) -> list[RubricItem]:
    """Read a rubric JSONL file. Item ids must be unique."""
    items = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = RubricItem.model_validate(json.loads(line))
            except ValueError as error:
                raise ValueError(f"{path}, line {number}: {error}") from error
            if item.item_id in seen:
                raise ValueError(f"{path}, line {number}: item_id {item.item_id} appears twice")
            seen.add(item.item_id)
            items.append(item)
    return items


def items_by_clip(items: list[RubricItem]) -> dict[str, list[RubricItem]]:
    by_clip: dict[str, list[RubricItem]] = {}
    for item in items:
        by_clip.setdefault(item.clip_id, []).append(item)
    return by_clip
