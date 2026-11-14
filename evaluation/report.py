"""results.csv: one row per measure, language and system, with the columns the submission
guidelines ask for (docs/evaluation_plan.md, section 8).

Confidence intervals and breakdowns do not go in results.csv: they go in the files next to it.
"""
import csv
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt

COLUMNS = ["track", "part", "language", "system", "measure", "n_items", "value"]


class ResultRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    track: Literal[2]
    part: Literal["captioner", "generator"]
    language: Literal["hi", "ta"]
    system: str         # e.g. "round_0", "round_0_recomposed"
    measure: str        # a scoring-table name ("rubric_score") or a clearly named proxy
    n_items: NonNegativeInt
    value: float


def write_results_csv(rows: list[ResultRow], path: str) -> None:
    """Write the rows in the order given. Written under a temporary name first, so a crash never
    leaves half a file."""
    with open(path + ".partial", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row.track, row.part, row.language, row.system, row.measure, row.n_items, f"{row.value:.6f}"])
    os.replace(path + ".partial", path)
