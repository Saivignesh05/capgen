"""Speaker diversity on a subset: build the subset folder, and merge its results into the main results.

Speaker diversity needs 10 samples per description (Rime's count is out of 10), which is the expensive
part of round 0. So the devset is run twice:

    run A  all 60 clips, 1 sample   (round0/settings.yaml)       -> every measure except speaker diversity
    run B  a subset, 10 samples     (round0/settings_eval.yaml)  -> speaker diversity only

    python -m scoring.diversity subset --devset ../devset --out ../devset_div --per-language 10
    python -m scoring.diversity merge --main A/scores/results.csv --diversity B/scores/results.csv --out results.csv

The subset keeps the <language>/clips/ layout (the scoring code reads the language from the folder)
and spreads its clips over as many speakers as possible: one clip per speaker in turn, the speaker
being the clip name without its trailing number (kashish1, kashish2 -> kashish). Standard library
only, so it runs in any environment.
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import OrderedDict

COLUMNS = ["track", "part", "language", "system", "measure", "value", "n_items"]
MEASURE = "speaker_diversity"


def log(message: str) -> None:
    print(f"[diversity] {message}", flush=True)


def speaker_of(clip_name: str) -> str:
    """kashish1 -> kashish, skd_5_rrk_3 -> skd_5_rrk, Sr1_sample31 -> Sr1_sample."""
    return re.sub(r"[_-]?\d+$", "", clip_name) or clip_name


def choose(clips: list[str], count: int) -> list[str]:
    """Up to `count` clip names, one per speaker in turn, in sorted order: deterministic."""
    by_speaker: "OrderedDict[str, list[str]]" = OrderedDict()
    for name in sorted(clips):
        by_speaker.setdefault(speaker_of(name), []).append(name)
    chosen = []
    while len(chosen) < count and any(by_speaker.values()):
        for names in by_speaker.values():
            if names and len(chosen) < count:
                chosen.append(names.pop(0))
    return sorted(chosen)


def make_subset(devset: str, out: str, per_language: int) -> dict:
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} exists and is not empty: delete it or choose another --out (a subset is never mixed with an old one)")
    manifest = {"devset": os.path.abspath(devset), "per_language": per_language, "clips": {}}
    for language in ("hi", "ta"):
        source = os.path.join(devset, language, "clips")
        if not os.path.isdir(source):
            sys.exit(f"no folder {source}")
        names = [os.path.splitext(f)[0] for f in os.listdir(source) if f.lower().endswith(".wav")]
        chosen = choose(names, per_language)
        if len(chosen) < per_language:
            log(f"WARNING {language}: only {len(chosen)} clips available, asked for {per_language}")
        target = os.path.join(out, language, "clips")
        os.makedirs(target, exist_ok=True)
        for name in chosen:
            shutil.copy2(os.path.join(source, name + ".wav"), os.path.join(target, name + ".wav"))
        speakers = sorted({speaker_of(n) for n in chosen})
        log(f"{language}: {len(chosen)} clips from {len(speakers)} speakers -> {target}")
        manifest["clips"][language] = chosen
    with open(os.path.join(out, "subset.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"wrote {os.path.join(out, 'subset.json')}")
    return manifest


def read_results(path: str) -> list[dict]:
    if not os.path.isfile(path):
        sys.exit(f"no such file: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            sys.exit(f"{path}: columns are {reader.fieldnames}, expected {COLUMNS}")
        return list(reader)


def merge(main_path: str, diversity_path: str, out_path: str) -> list[dict]:
    """Main results without speaker diversity + the subset run's speaker diversity rows."""
    main_rows = read_results(main_path)
    subset_rows = read_results(diversity_path)
    diversity = [r for r in subset_rows if r["measure"] == MEASURE]
    if not diversity:
        sys.exit(f"{diversity_path} has no {MEASURE} rows: did its scoring run the speaker_diversity stage?")
    systems = {r["system"] for r in main_rows} | {r["system"] for r in diversity}
    if len(systems) > 1:
        sys.exit(f"the two files are for different systems {sorted(systems)}: not merging")
    dropped = [r for r in main_rows if r["measure"] == MEASURE]
    if dropped:
        log(f"replacing {len(dropped)} {MEASURE} rows of the main run (computed on too few samples) with the subset's")
    ignored = len(subset_rows) - len(diversity)
    if ignored:
        log(f"ignoring {ignored} other rows of the subset run (the main run covers every clip)")
    rows = [r for r in main_rows if r["measure"] != MEASURE] + diversity
    rows.sort(key=lambda r: (r["part"], r["measure"], r["language"]))
    with open(out_path + ".partial", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(out_path + ".partial", out_path)
    for r in diversity:
        log(f"{MEASURE} {r['language']}: {r['value']} on {r['n_items']} items")
    log(f"wrote {out_path}: {len(rows)} rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    subset = commands.add_parser("subset", help="copy a speaker-balanced subset of the devset")
    subset.add_argument("--devset", required=True)
    subset.add_argument("--out", required=True)
    subset.add_argument("--per-language", type=int, default=10)
    joined = commands.add_parser("merge", help="main results.csv + the subset run's speaker diversity")
    joined.add_argument("--main", required=True, help="results.csv of run A (all clips, 1 sample)")
    joined.add_argument("--diversity", required=True, help="results.csv of run B (subset, 10 samples)")
    joined.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "subset":
        make_subset(args.devset, args.out, args.per_language)
    else:
        merge(args.main, args.diversity, args.out)


if __name__ == "__main__":
    main()
