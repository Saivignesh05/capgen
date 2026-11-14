"""The rubric score against values computed by hand."""
import pytest

from evaluation.rubric.score import clip_score, item_points
from helpers import rubric_item


def test_mixed_verdicts_on_true_items():
    # 4 true items: stated, stated, not mentioned, contradicted -> (1 + 1 + 0 - 1) / 4
    items = [rubric_item("a", True, 1), rubric_item("b", True, 1), rubric_item("c", True, 1), rubric_item("d", True, 1)]
    verdicts = {"a": "STATED", "b": "STATED", "c": "NOT_MENTIONED", "d": "CONTRADICTED"}
    assert clip_score(items, verdicts, use_weights=False) == pytest.approx(0.25)


def test_negative_sum_is_floored_at_zero():
    # 2 true items, both contradicted -> -2 / 2 = -1 -> 0
    items = [rubric_item("a", True, 1), rubric_item("b", True, 1)]
    verdicts = {"a": "CONTRADICTED", "b": "CONTRADICTED"}
    assert clip_score(items, verdicts, use_weights=False) == 0.0


def test_stated_false_item_costs_a_point():
    # 4 true items all stated, 2 false items: one stated (-1), one contradicted (0) -> (4 - 1) / 4
    items = [rubric_item("a", True, 1), rubric_item("b", True, 1), rubric_item("c", True, 1), rubric_item("d", True, 1),
             rubric_item("x", False, 1), rubric_item("y", False, 1)]
    verdicts = {"a": "STATED", "b": "STATED", "c": "STATED", "d": "STATED", "x": "STATED", "y": "CONTRADICTED"}
    assert clip_score(items, verdicts, use_weights=False) == pytest.approx(0.75)


def test_false_items_do_not_count_in_the_divisor():
    # 3 true items (1 stated), 2 false items (1 stated) -> (1 - 1) / 3 = 0
    items = [rubric_item("a", True, 1), rubric_item("b", True, 1), rubric_item("c", True, 1),
             rubric_item("x", False, 1), rubric_item("y", False, 1)]
    verdicts = {"a": "STATED", "b": "NOT_MENTIONED", "c": "NOT_MENTIONED", "x": "STATED", "y": "NOT_MENTIONED"}
    assert clip_score(items, verdicts, use_weights=False) == 0.0


def test_everything_right_scores_one():
    items = [rubric_item("a", True, 2), rubric_item("b", True, 3), rubric_item("x", False, 1)]
    verdicts = {"a": "STATED", "b": "STATED", "x": "NOT_MENTIONED"}
    assert clip_score(items, verdicts, use_weights=False) == 1.0
    assert clip_score(items, verdicts, use_weights=True) == 1.0


def test_weights_change_the_score():
    # true items with weights 1, 3, 2: stated, contradicted, stated
    # weighted:   (1 - 3 + 2) / (1 + 3 + 2) = 0
    # unweighted: (1 - 1 + 1) / 3 = 1/3
    items = [rubric_item("a", True, 1), rubric_item("b", True, 3), rubric_item("c", True, 2)]
    verdicts = {"a": "STATED", "b": "CONTRADICTED", "c": "STATED"}
    assert clip_score(items, verdicts, use_weights=True) == 0.0
    assert clip_score(items, verdicts, use_weights=False) == pytest.approx(1 / 3)


def test_weighted_false_item():
    # true items w2 and w1, both stated; false item w3 stated
    # weighted:   (2 + 1 - 3) / (2 + 1) = 0
    # unweighted: (1 + 1 - 1) / 2 = 0.5
    items = [rubric_item("a", True, 2), rubric_item("b", True, 1), rubric_item("x", False, 3)]
    verdicts = {"a": "STATED", "b": "STATED", "x": "STATED"}
    assert clip_score(items, verdicts, use_weights=True) == 0.0
    assert clip_score(items, verdicts, use_weights=False) == pytest.approx(0.5)


def test_item_points_table():
    true_item = rubric_item("a", True, 2)
    false_item = rubric_item("x", False, 2)
    assert item_points(true_item, "STATED", use_weights=True) == 2
    assert item_points(true_item, "NOT_MENTIONED", use_weights=True) == 0
    assert item_points(true_item, "CONTRADICTED", use_weights=True) == -2
    assert item_points(false_item, "STATED", use_weights=True) == -2
    assert item_points(false_item, "NOT_MENTIONED", use_weights=True) == 0
    assert item_points(false_item, "CONTRADICTED", use_weights=True) == 0
    assert item_points(true_item, "STATED", use_weights=False) == 1


def test_missing_verdict_is_an_error():
    items = [rubric_item("a", True, 1), rubric_item("b", True, 1)]
    with pytest.raises(ValueError, match="no verdict"):
        clip_score(items, {"a": "STATED"}, use_weights=False)


def test_rubric_without_true_items_is_an_error():
    with pytest.raises(ValueError, match="true item"):
        clip_score([rubric_item("x", False, 1)], {"x": "STATED"}, use_weights=False)
