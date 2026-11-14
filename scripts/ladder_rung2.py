"""Ladder rung 2: does re-composing a round 0 caption into the grader's 1,000 characters raise its
rubric score? (docs/post_midterm_plan.md, section 3.2)

Run from the repository root with the evaluation environment, on a GPU node:

    python -m scripts.ladder_rung2 --settings settings/phase0.yaml --paths settings/paths.yaml \
        --run /path/to/round0/run-folder --rubric /path/to/rubric.jsonl --name rung2-tune

Two systems are scored on the same clips with the same grader:
    round_0              the round 0 caption as it is (the grader reads its first 1,000 characters)
    round_0_recomposed   the same caption, split into claims and re-composed into 1,000 characters
No model is trained and no audio is read: the only difference is which sentences reach the grader.

Writes to <outputs>/<name>/:
    compositions.jsonl   per clip: both texts, their lengths, how many claims were used and dropped
    gradings.jsonl       per item and system: the grader's verdict and raw answer
    summary.json         per system: score per language, overall score with its confidence interval,
                         parse-failure rate; and the paired difference with its gap-closed level
    results.csv          the submission-format rows (rubric_score unweighted, rubric_score_weighted)
    manifest.json        code commit and SHA-256 of the settings, inputs and grader checkpoint
    run.log              everything printed

Resumable: run the same command again after a time limit; graded clips are kept.

The rubric decides what this run may be used for. A rubric of held-out clips is for reporting only:
never change the composer's settings after looking at its result. Tune on the tuning split.
"""
import argparse
import json
import logging
import os
import sys

from claims.composer import compose
from claims.extractor import ClaimExtractorInterface, RuleBasedExtractor
from claims.settings import ClaimsSettings
from claims.settings import load_settings as load_claims_settings
from evaluation.captioner_eval import Description, compare_systems, grade_descriptions, summarise_system
from evaluation.manifest import RunManifest, code_commit, file_sha256, folder_sha256, now_utc, write_manifest
from evaluation.report import ResultRow, write_results_csv
from evaluation.rubric.grader import GraderInterface
from evaluation.rubric.grader_sarvam import SarvamGrader, SarvamGraderSettings
from evaluation.rubric.schema import RubricItem, items_by_clip, read_rubric
from evaluation.settings import EvaluationSettings, load_paths, load_settings, model_folder
from evaluation.stats import gap_closed_level

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = "round_0"
RECOMPOSED = "round_0_recomposed"
RUBRIC_CEILING = 1.0        # the highest possible rubric score


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def setup_logging(output_dir: str) -> logging.Logger:
    log = logging.getLogger("ladder_rung2")
    log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def language_of_path(path: str) -> str | None:
    """hi or ta, from a folder named hi or ta in the clip's path (the devset layout)."""
    parts = path.replace("\\", "/").split("/")
    for part in reversed(parts[:-1]):
        if part in ("hi", "ta"):
            return part
    return None


def round0_captions(run_dir: str, log: logging.Logger) -> dict[str, str]:
    """{rubric clip id: round 0 caption} from a round 0 run folder. The rubric's clip id is
    "<language>/<round 0 clip_id>", with the language taken from the clip's folder."""
    captions = {record["clip_id"]: record["description"] for record in read_jsonl(os.path.join(run_dir, "descriptions.jsonl"))}
    by_rubric_id = {}
    for clip in read_jsonl(os.path.join(run_dir, "clips.jsonl")):
        language = language_of_path(clip["path"])
        if language is None:
            log.warning("%s: no hi/ or ta/ folder in its path %s; skipped", clip["clip_id"], clip["path"])
            continue
        if clip["clip_id"] not in captions:
            log.warning("%s: round 0 wrote no caption; scored as an empty description (0)", clip["clip_id"])
        by_rubric_id[f"{language}/{clip['clip_id']}"] = captions.get(clip["clip_id"], "")
    return by_rubric_id


def build_descriptions(captions: dict[str, str], rubric_by_clip: dict[str, list[RubricItem]], extractor: ClaimExtractorInterface,
                       claims_settings: ClaimsSettings, compositions_path: str, log: logging.Logger) -> list[Description]:
    """The two systems' descriptions of every clip that has both a rubric and a round 0 run record."""
    descriptions = []
    rows = []
    for clip_id in sorted(rubric_by_clip):
        if clip_id not in captions:
            log.warning("%s: in the rubric but not in the round 0 run; skipped", clip_id)
            continue
        caption = captions[clip_id]
        composition = compose(extractor.extract(caption), claims_settings.composer)
        descriptions.append(Description(system=BASELINE, clip_id=clip_id, text=caption))
        descriptions.append(Description(system=RECOMPOSED, clip_id=clip_id, text=composition.text))
        rows.append({"clip_id": clip_id, "round_0": caption, "round_0_chars": len(caption),
                     "round_0_recomposed": composition.text, "recomposed_chars": len(composition.text),
                     "claims_used": len(composition.used), "dropped_low_probability": composition.dropped_low_probability,
                     "dropped_duplicate": composition.dropped_duplicate, "dropped_no_room": composition.dropped_no_room})
    with open(compositions_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    without_rubric = sorted(set(captions) - set(rubric_by_clip))
    if without_rubric:
        log.info("%d round 0 clips have no rubric and are not scored", len(without_rubric))
    return descriptions


def score_and_report(grader: GraderInterface, settings: EvaluationSettings, claims_settings: ClaimsSettings,
                     run_dir: str, rubric_path: str, output_dir: str, log: logging.Logger) -> dict:
    """Everything after the models are loaded: build both systems' descriptions, grade them, and
    write compositions.jsonl, gradings.jsonl, summary.json and results.csv. Returns the summary."""
    rubric_by_clip = items_by_clip(read_rubric(rubric_path))
    captions = round0_captions(run_dir, log)
    extractor = RuleBasedExtractor(claims_settings.extractor)
    descriptions = build_descriptions(captions, rubric_by_clip, extractor, claims_settings,
                                      os.path.join(output_dir, "compositions.jsonl"), log)
    if len(descriptions) == 0:
        raise ValueError("no clip has both a rubric and a round 0 run record: nothing to score")
    log.info("%d clips to score, 2 systems", len(descriptions) // 2)

    records = grade_descriptions(grader, descriptions, rubric_by_clip, settings.grader,
                                 os.path.join(output_dir, "gradings.jsonl"), log)

    summary = {"systems": [], "comparisons": []}
    result_rows = []
    for use_weights, measure in ((False, "rubric_score"), (True, "rubric_score_weighted")):
        for system in (BASELINE, RECOMPOSED):
            system_summary = summarise_system(records, system, rubric_by_clip, use_weights, measure, settings.bootstrap)
            summary["systems"].append({"measure": measure, **system_summary.model_dump()})
            log.info("%s %s: %.4f [%.4f, %.4f], by language %s, parse failures %.1f%%", measure, system,
                     system_summary.overall.value, system_summary.overall.low, system_summary.overall.high,
                     system_summary.by_language, 100 * system_summary.parse_failure_rate)
            for language in sorted(system_summary.by_language):
                result_rows.append(ResultRow(track=2, part="captioner", language=language, system=system, measure=measure,
                                             n_items=system_summary.n_clips_by_language[language],
                                             value=system_summary.by_language[language]))
        baseline_summary = summarise_system(records, BASELINE, rubric_by_clip, use_weights, measure, settings.bootstrap)
        difference = compare_systems(records, RECOMPOSED, BASELINE, rubric_by_clip, use_weights, measure, settings.bootstrap)
        level = gap_closed_level(difference, baseline_summary.overall.value, RUBRIC_CEILING, settings.gap_closed)
        summary["comparisons"].append({"measure": measure, "system": RECOMPOSED, "baseline": BASELINE,
                                       "difference": difference.model_dump(), "gap_closed_level": level})
        log.info("%s %s minus %s: %+.4f [%+.4f, %+.4f], gap-closed level %d", measure, RECOMPOSED, BASELINE,
                 difference.value, difference.low, difference.high, level)

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_results_csv(result_rows, os.path.join(output_dir, "results.csv"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", required=True, help="settings/phase0.yaml")
    parser.add_argument("--paths", required=True, help="settings/paths.yaml")
    parser.add_argument("--run", required=True, help="a round 0 run folder (clips.jsonl, descriptions.jsonl)")
    parser.add_argument("--rubric", required=True, help="rubric JSONL (evaluation/rubric/schema.py)")
    parser.add_argument("--name", required=True, help="name of the output folder under <outputs>")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    claims_settings = load_claims_settings(args.settings)
    paths = load_paths(args.paths)
    output_dir = os.path.join(paths.outputs, args.name)
    os.makedirs(output_dir, exist_ok=True)
    log = setup_logging(output_dir)
    started = now_utc()
    log.info("ladder rung 2: run=%s rubric=%s output=%s", args.run, args.rubric, output_dir)

    grader_dir = model_folder(paths, settings.downloads.grader)
    grader = SarvamGrader(SarvamGraderSettings(model_dir=grader_dir, **settings.grader.model_dump()))
    log.info("grader loaded from %s", grader_dir)
    score_and_report(grader, settings, claims_settings, args.run, args.rubric, output_dir, log)

    write_manifest(RunManifest(
        script="scripts/ladder_rung2.py", started_utc=started, code_commit=code_commit(REPO_DIR),
        settings_sha256=file_sha256(args.settings),
        input_sha256={"rubric": file_sha256(args.rubric),
                      "round0_descriptions": file_sha256(os.path.join(args.run, "descriptions.jsonl")),
                      "round0_clips": file_sha256(os.path.join(args.run, "clips.jsonl"))},
        checkpoint_sha256={"grader": folder_sha256(grader_dir)},
        notes={"grader": f"{settings.downloads.grader.repo_id}@{settings.downloads.grader.revision}",
               "extractor": claims_settings.extractor.implementation}),
        os.path.join(output_dir, "manifest.json"))
    log.info("done: %s", output_dir)


if __name__ == "__main__":
    main()
