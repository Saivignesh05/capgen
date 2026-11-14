"""Statistics: hash-based bootstrap confidence intervals, paired comparison of two systems, and the
gap-closed level (docs/evaluation_plan.md, section 6).

Every measure here is "higher is better". An error rate is turned into 1 - error before it comes here.

Values are passed as {language: {item id: value}}. A statistic is the mean of the per-language
means, so each language counts the same however many items it has. The bootstrap resamples each
language separately and then averages, as the plan asks.
"""
import hashlib

from pydantic import BaseModel, ConfigDict

from evaluation.settings import BootstrapOptions, GapClosedOptions

ValuesByLanguage = dict[str, dict[str, float]]


class Interval(BaseModel):
    """A point estimate with its 95% bootstrap confidence interval."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float
    low: float
    high: float
    n_items: int


def draw_position(seed: int, measure: str, replicate: int, draw: int, n_items: int) -> int:
    """Which item draw `draw` of replicate `replicate` picks: SHA-256("seed|bootstrap|measure|r|j") mod n.

    Depends only on its arguments, so two systems scored on the same items get identical draws.
    CAREFUL T1
    """
    digest = hashlib.sha256(f"{seed}|bootstrap|{measure}|{replicate}|{draw}".encode("utf-8")).hexdigest()
    return int(digest, 16) % n_items


def mean(values: list[float]) -> float:
    if len(values) == 0:
        raise ValueError("mean of no values")
    return sum(values) / len(values)


def mean_over_languages(values_by_language: ValuesByLanguage) -> float:
    """The mean of the per-language means."""
    language_means = []
    for language in sorted(values_by_language):
        language_means.append(mean(list(values_by_language[language].values())))
    return mean(language_means)


def count_items(values_by_language: ValuesByLanguage) -> int:
    total = 0
    for language in values_by_language:
        total += len(values_by_language[language])
    return total


def replicate_statistics(values_by_language: ValuesByLanguage, measure: str, options: BootstrapOptions) -> list[float]:
    """One statistic per bootstrap replicate: each language resampled separately, then averaged."""
    statistics = []
    for replicate in range(1, options.replicates + 1):
        language_means = []
        for language in sorted(values_by_language):
            by_item = values_by_language[language]
            item_ids = sorted(by_item)      # items in the order of their sorted ids
            n_items = len(item_ids)
            drawn = []
            for draw in range(n_items):
                position = draw_position(options.seed, measure, replicate, draw, n_items)
                drawn.append(by_item[item_ids[position]])
            language_means.append(mean(drawn))
        statistics.append(mean(language_means))
    return statistics


def interval_from_replicates(value: float, statistics: list[float], n_items: int, options: BootstrapOptions) -> Interval:
    ordered = sorted(statistics)
    return Interval(value=value, low=ordered[options.lower_position], high=ordered[options.upper_position], n_items=n_items)


def confidence_interval(values_by_language: ValuesByLanguage, measure: str, options: BootstrapOptions) -> Interval:
    """The mean over languages of one system's values, with its bootstrap confidence interval."""
    statistics = replicate_statistics(values_by_language, measure, options)
    return interval_from_replicates(mean_over_languages(values_by_language), statistics, count_items(values_by_language), options)


def paired_difference(system: ValuesByLanguage, baseline: ValuesByLanguage, measure: str, options: BootstrapOptions) -> Interval:
    """system minus baseline on the same items, with identical draws for both systems.

    Both must hold exactly the same languages and item ids: a paired comparison of different items
    would not be paired.
    """
    if sorted(system) != sorted(baseline):
        raise ValueError(f"different languages: {sorted(system)} and {sorted(baseline)}")
    differences: ValuesByLanguage = {}
    for language in system:
        if sorted(system[language]) != sorted(baseline[language]):
            raise ValueError(f"the two systems were not scored on the same {language} items")
        differences[language] = {}
        for item_id in system[language]:
            differences[language][item_id] = system[language][item_id] - baseline[language][item_id]
    # The same draws applied to both systems give the mean of the drawn differences. CAREFUL T2
    return confidence_interval(differences, measure, options)


def gap_closed(value: float, baseline: float, ceiling: float) -> float:
    """g = (x - b) / (c - b): the share of the gap between the baseline and the ceiling that was closed."""
    if ceiling <= baseline:
        raise ValueError(f"the ceiling {ceiling} must be above the baseline {baseline}")
    return (value - baseline) / (ceiling - baseline)


def gap_closed_level(difference: Interval, baseline: float, ceiling: float, options: GapClosedOptions) -> int:
    """Level 0 to 4 of a system against a baseline (doubt session Q3).

    `difference` is paired_difference(system, baseline): its value is x - b.
        0   the interval lies fully on the worse side (below 0)
        1   the interval includes 0
        2   clearly better, g below level_3_from
        3   g from level_3_from up to level_4_from
        4   g at or above level_4_from
    """
    if difference.high < 0:
        return 0
    if difference.low <= 0:
        return 1
    g = gap_closed(baseline + difference.value, baseline, ceiling)
    if g >= options.level_4_from:
        return 4
    if g >= options.level_3_from:
        return 3
    return 2


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# T1   Replicates are numbered from 1 and draws from 0. The plan gives the formula but not where the
#      counting starts; this is an open choice (docs/EXPLANATION.md) and is kept in one place.
#      The same positions are used for every language with the same number of items, because the
#      formula has no language in it.
# T2   Bootstrapping the per-item differences is the same computation as drawing identical items
#      for both systems and subtracting the two means; the code does the first because it is shorter.
# ---------------------------------------------------------------------------------------------
