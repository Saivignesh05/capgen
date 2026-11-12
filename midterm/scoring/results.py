"""Build results.csv (the midterm format) from whatever the stages wrote.

    track,part,language,system,measure,value,n_items

One row per number. Every value is first computed per item (one devset clip = one description) and
then averaged over items, as Rime averages over clips and descriptions; n_items is the number of
items behind the value. Corpus measures (FAD, PSD) are one number per language; their n_items is the
number of generated samples.

results_detail.json holds the same numbers with 95% bootstrap confidence intervals over items
(Rime's Appendix A draws), what each measure means and which direction is better, the level the
native devset clips get, and which measures are missing and why.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

from scoring.common import LANGUAGES, bootstrap_ci, read_jsonl, setup_logging, write_json
from scoring.settings import load_settings

COLUMNS = ["track", "part", "language", "system", "measure", "value", "n_items"]

MEANINGS = {
    "rubric_score": ("captioner", "higher", "rubric score vs the ground-truth reference descriptions (text-LLM grader, equal weights)"),
    "rubric_score_tool": ("captioner", "higher", "rubric score vs the features-extractor tool descriptions (measured facts only)"),
    "instruction_relevance": ("generator", "higher", "share of instruction checks met, judged from a caption of the generated audio"),
    "intelligibility": ("generator", "lower", "character error rate of IndicConformer on the generated audio (Rime: points = 1 - CER)"),
    "speaker_diversity": ("generator", "higher", "distinct speakers among the samples of a description (out of samples per item)"),
    "naturalness_utmos": ("generator", "higher", "UTMOS (1-5), automatic stand-in for Rime's naturalness CMOS"),
    "naturalness_sigmos_ovrl": ("generator", "higher", "SIGMOS overall (1-5), automatic stand-in for naturalness"),
    "naturalness_fad": ("generator", "lower", "Frechet distance of XLS-R embeddings to the native devset clips (corpus)"),
    "naturalness_psd": ("generator", "lower", "Frechet distance of prosody features to the native devset clips (corpus)"),
}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def per_item(rows, value_of, group=lambda r: (r["language"], r["clip_id"])):
    """{language: {clip_id: value}}, where value_of gets all rows of one item."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[group(row)].append(row)
    result = defaultdict(dict)
    for (language, clip_id), item_rows in grouped.items():
        value = value_of(item_rows)
        if value is not None:
            result[language][clip_id] = value
    return result


def compute(scores_dir: str, settings, system: str) -> tuple[list[dict], dict]:
    """(csv rows, detail) from the stage outputs in scores_dir."""
    measures = {}       # measure -> {language: {clip_id: value}}
    notes = {}

    def load(name):
        rows = read_jsonl(os.path.join(scores_dir, f"{name}.jsonl"))
        if not rows:
            notes[name] = "no output (stage not run, failed, or nothing to score: see logs/)"
        return rows

    for reference in settings.references:
        rows = [r for r in load("rubric") if r["reference"] == reference.name]
        if rows:
            measures[reference.measure] = per_item(rows, lambda rs: rs[0]["score"])
        else:
            notes[reference.measure] = f"no rubric gradings for reference {reference.name} (no references, or judge stage not run)"

    rows = load("relevance")
    if rows:
        measures["instruction_relevance"] = per_item(rows, lambda rs: sum(r["met"] for r in rs) / sum(r["n_checks"] for r in rs))
    rows = load("intelligibility")
    if rows:
        measures["intelligibility"] = per_item(rows, lambda rs: sum(r["edits"] for r in rs) / sum(r["n"] for r in rs))
    rows = load("speaker_diversity")
    if rows:
        measures["speaker_diversity"] = per_item(rows, lambda rs: rs[0]["n_speakers"])
    rows = load("naturalness")
    for key, measure in (("utmos", "naturalness_utmos"), ("sigmos.ovrl", "naturalness_sigmos_ovrl")):
        values = [r for r in rows if r.get(key) is not None]
        if values:
            measures[measure] = per_item(values, lambda rs, key=key: mean(r[key] for r in rs))

    csv_rows, detail = [], {"system": system, "measures": {}, "missing": notes, "native_level": {}}
    seed, replicates = settings.seed, settings.bootstrap_replicates
    for measure, by_language in measures.items():
        part, better, meaning = MEANINGS.get(measure, ("captioner", "higher", ""))
        for language in LANGUAGES:
            values = by_language.get(language, {})
            if not values:
                continue
            value = mean(values.values())
            ci = bootstrap_ci(values, seed, f"{part}/{measure}/{language}", replicates)
            csv_rows.append(row(part, language, system, measure, value, len(values)))
            detail["measures"].setdefault(measure, {"part": part, "better": better, "meaning": meaning})[language] = {
                "value": value, "ci95": ci, "n_items": len(values)}

    corpus_path = os.path.join(scores_dir, "naturalness_corpus.json")
    if os.path.isfile(corpus_path):
        import json
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)
        for language, numbers in corpus.items():
            for key in ("fad", "psd"):
                if numbers.get(key) is not None:
                    measure = f"naturalness_{key}"
                    part, better, meaning = MEANINGS[measure]
                    csv_rows.append(row(part, language, system, measure, numbers[key], numbers["n_generated"]))
                    detail["measures"].setdefault(measure, {"part": part, "better": better, "meaning": meaning})[language] = {
                        "value": numbers[key], "ci95": None, "n_items": numbers["n_generated"]}

    native = read_jsonl(os.path.join(scores_dir, "naturalness_native.jsonl"))
    for language in LANGUAGES:
        for key in ("utmos", "sigmos.ovrl"):
            values = [r[key] for r in native if r["language"] == language and r.get(key) is not None]
            if values:
                detail["native_level"].setdefault(language, {})[key] = {"mean": mean(values), "n": len(values)}

    csv_rows.sort(key=lambda r: (r["part"], r["measure"], r["language"]))
    return csv_rows, detail


def row(part, language, system, measure, value, n_items) -> dict:
    return {"track": 2, "part": part, "language": language, "system": system, "measure": measure,
            "value": round(float(value), 4), "n_items": n_items}


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path + ".partial", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(path + ".partial", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--system", required=True, help="round_0, round_1, ...")
    args = parser.parse_args()

    logger = setup_logging(args.scores, "results")
    settings = load_settings(args.settings)
    rows, detail = compute(args.scores, settings, args.system)
    write_csv(os.path.join(args.scores, "results.csv"), rows)
    write_json(os.path.join(args.scores, "results_detail.json"), detail)

    logger.info("%-10s %-4s %-26s %10s %22s %7s", "part", "lang", "measure", "value", "95% CI", "n")
    for r in rows:
        ci = detail["measures"][r["measure"]][r["language"]]["ci95"]
        ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
        logger.info("%-10s %-4s %-26s %10.4f %22s %7d", r["part"], r["language"], r["measure"], r["value"], ci_text, r["n_items"])
    for measure, why in detail["missing"].items():
        logger.warning("missing %s: %s", measure, why)
    logger.info("wrote %s (%d rows) and results_detail.json", os.path.join(args.scores, "results.csv"), len(rows))
    if not rows:
        logger.error("results.csv has no rows: no stage produced scores")
        sys.exit(1)


if __name__ == "__main__":
    main()
