"""The bootstrap, the paired comparison and the gap-closed level, on small inputs with known answers."""
import hashlib

import pytest

from evaluation.settings import BootstrapOptions, GapClosedOptions
from evaluation.stats import (Interval, confidence_interval, draw_position, gap_closed, gap_closed_level,
                              mean_over_languages, paired_difference, replicate_statistics)

OPTIONS = BootstrapOptions(seed=0, replicates=1000, lower_position=24, upper_position=974)
LEVELS = GapClosedOptions(level_3_from=0.25, level_4_from=0.5)
VALUES = {"hi": {"a": 0.2, "b": 0.4, "c": 0.9, "d": 0.5}, "ta": {"e": 0.1, "f": 0.7, "g": 0.6}}


def test_draw_position_is_the_documented_hash():
    expected = int(hashlib.sha256(b"7|bootstrap|rubric_score|3|2").hexdigest(), 16) % 5
    assert draw_position(seed=7, measure="rubric_score", replicate=3, draw=2, n_items=5) == expected


def test_mean_over_languages_weights_languages_equally():
    # hi mean 0.5 (4 items), ta mean 0.4666... (3 items) -> the mean of the two, not of the 7 items
    assert mean_over_languages(VALUES) == pytest.approx((0.5 + 1.4 / 3) / 2)


def test_same_seed_gives_the_same_interval():
    first = confidence_interval(VALUES, "rubric_score", OPTIONS)
    second = confidence_interval(VALUES, "rubric_score", OPTIONS)
    assert first == second
    assert first.low <= first.value <= first.high
    assert first.n_items == 7


def test_other_seed_or_measure_gives_other_draws():
    base = replicate_statistics(VALUES, "rubric_score", OPTIONS)
    other_seed = replicate_statistics(VALUES, "rubric_score", BootstrapOptions(seed=1, replicates=1000, lower_position=24, upper_position=974))
    other_measure = replicate_statistics(VALUES, "intelligibility", OPTIONS)
    assert base != other_seed
    assert base != other_measure


def test_constant_values_give_a_zero_width_interval():
    values = {"hi": {"a": 0.3, "b": 0.3, "c": 0.3}, "ta": {"d": 0.3, "e": 0.3}}
    interval = confidence_interval(values, "rubric_score", OPTIONS)
    assert interval.value == pytest.approx(0.3)
    assert interval.low == pytest.approx(0.3)
    assert interval.high == pytest.approx(0.3)


def test_interval_positions_are_the_25th_and_975th_values():
    statistics = sorted(replicate_statistics(VALUES, "rubric_score", OPTIONS))
    interval = confidence_interval(VALUES, "rubric_score", OPTIONS)
    assert interval.low == statistics[24]
    assert interval.high == statistics[974]


def test_paired_difference_of_a_constant_shift_is_exact():
    # every item is 0.1 better: whatever is drawn, the difference is 0.1
    system = {language: {item: value + 0.1 for item, value in items.items()} for language, items in VALUES.items()}
    difference = paired_difference(system, VALUES, "rubric_score", OPTIONS)
    assert difference.value == pytest.approx(0.1)
    assert difference.low == pytest.approx(0.1)
    assert difference.high == pytest.approx(0.1)


def test_paired_difference_needs_the_same_items():
    system = {"hi": {"a": 0.2, "b": 0.4}, "ta": {"e": 0.1}}
    with pytest.raises(ValueError, match="same hi items"):
        paired_difference(system, VALUES, "rubric_score", OPTIONS)
    with pytest.raises(ValueError, match="different languages"):
        paired_difference({"hi": VALUES["hi"]}, VALUES, "rubric_score", OPTIONS)


def test_gap_closed():
    assert gap_closed(value=0.5, baseline=0.2, ceiling=1.0) == pytest.approx(0.375)
    with pytest.raises(ValueError):
        gap_closed(value=0.5, baseline=1.0, ceiling=1.0)


def test_gap_closed_levels():
    def level(value: float, low: float, high: float) -> int:
        return gap_closed_level(Interval(value=value, low=low, high=high, n_items=10), baseline=0.2, ceiling=1.0, options=LEVELS)

    assert level(-0.10, -0.15, -0.05) == 0      # interval fully on the worse side
    assert level(0.05, -0.02, 0.12) == 1        # interval includes 0
    assert level(0.10, 0.02, 0.18) == 2         # clearly better, g = 0.10 / 0.8 = 0.125
    assert level(0.24, 0.10, 0.38) == 3         # g = 0.30
    assert level(0.40, 0.30, 0.50) == 4         # g = 0.50
