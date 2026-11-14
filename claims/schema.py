"""A claim: one typed statement about a clip, with the probability that it is right.

Claims are what the captioner and the generator exchange in the Claim-Calibrated Loop
(docs/post_midterm_plan.md, section 3). Every field is required: write null for a time that does
not apply.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from evaluation.rubric.schema import ItemType

ClaimSource = Literal["model", "tool", "transform"]


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_type: ItemType         # one of the 13 rubric item types
    statement: str              # plain English, complete in itself
    start_s: float | None       # where it happens in the clip; null for a claim about the whole clip
    end_s: float | None
    source: ClaimSource         # who made the claim: a model, a measuring tool, or a known transform
    probability: float          # the probability that the claim is right, between 0 and 1

    @field_validator("statement")
    @classmethod
    def statement_must_not_be_empty(cls, statement: str) -> str:
        if not statement.strip():
            raise ValueError("statement is empty")
        return statement

    @field_validator("probability")
    @classmethod
    def probability_must_be_between_0_and_1(cls, probability: float) -> float:
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"probability must be between 0 and 1, got: {probability}")
        return probability

    @model_validator(mode="after")
    def times_must_be_a_span_or_absent(self) -> "Claim":
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("start_s and end_s must both be set or both be null")
        if self.start_s is not None and self.end_s < self.start_s:
            raise ValueError("end_s is before start_s")
        return self
