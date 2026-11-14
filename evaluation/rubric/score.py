"""The rubric score of one description, from the grader's verdicts (docs/evaluation_plan.md, section 3.1).

    true item:   +w if stated, 0 if not mentioned, -w if contradicted
    false item:  -w if stated, 0 otherwise
    clip score:  max(0, sum of points / sum of w over the true items)

With use_weights false every w is 1: the doubt-session (Q31) version, taken as the official one.
With use_weights true w is the item's difficulty weight: the problem-statement version.
"""
from evaluation.rubric.schema import RubricItem, Verdict


def item_weight(item: RubricItem, use_weights: bool) -> int:
    if use_weights:
        return item.weight
    return 1


def item_points(item: RubricItem, verdict: Verdict, use_weights: bool) -> int:
    """What one verdict on one item adds to the clip's sum."""
    weight = item_weight(item, use_weights)
    if item.truth:
        if verdict == "STATED":
            return weight
        if verdict == "CONTRADICTED":
            return -weight
        return 0
    if verdict == "STATED":
        return -weight
    return 0


def clip_score(items: list[RubricItem], verdicts: dict[str, Verdict], use_weights: bool) -> float:
    """Score of one description against the rubric items of its clip. `verdicts` is {item_id: verdict}
    and must cover every item: a missing verdict is an error, not a silent zero."""
    total_points = 0
    true_weight = 0
    for item in items:
        if item.item_id not in verdicts:
            raise ValueError(f"no verdict for item {item.item_id}")
        total_points += item_points(item, verdicts[item.item_id], use_weights)
        if item.truth:
            true_weight += item_weight(item, use_weights)
    if true_weight == 0:
        raise ValueError("a clip's rubric needs at least one true item: the score divides by them")
    return max(0.0, total_points / true_weight)
