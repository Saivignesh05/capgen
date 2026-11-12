"""Score a round 0 run: run every scoring stage, then build results.csv.

Run from midterm/ with the scoring environment's Python:

    python -m scoring.run --settings settings.yaml --run /path/to/runs/run-20261112-093000

Each stage runs in its own process (GPU memory is freed between stages, and one stage crashing does
not stop the others). Stages resume: a second call only does what is missing. Everything is logged
to <scores>/logs/<stage>.log; per-item failures are also in <scores>/errors.jsonl; the stage exit
codes and times are in <scores>/scoring_run.json.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time

from scoring.common import log_environment, now, read_jsonl, setup_logging, write_json
from scoring.settings import load_settings

MIDTERM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(MIDTERM)
ROUND0_CODE = os.path.join(REPO, "round0", "code")


def stage_environment(settings) -> dict:
    """Environment for the stage processes: offline, pinned caches, and every code folder importable."""
    paths = [MIDTERM, REPO, ROUND0_CODE, settings.sigmos_dir]       # CAREFUL R1
    if settings.naturalness.psp_eval_dir:
        paths.append(settings.naturalness.psp_eval_dir)
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    return {**os.environ,
            "PYTHONPATH": os.pathsep.join(paths),
            "HF_HUB_CACHE": settings.hf_cache,
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",       # never download on the HPC
            "TORCH_HOME": settings.torch_home,
            "PYTHONDONTWRITEBYTECODE": "1",     # keep model folders clean (checkpoint hashes)
            "PYTHONUNBUFFERED": "1",            # logs appear in the SLURM output as they happen
            "TOKENIZERS_PARALLELISM": "false"}


def git_commit() -> str | None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
        return commit + ("+uncommitted-changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", required=True, help="midterm/settings.yaml")
    parser.add_argument("--run", required=True, help="round 0 run folder (<runs>/run-...)")
    parser.add_argument("--scores", help="where scores go (default: <run>/scores)")
    parser.add_argument("--system", default="round_0", help="system column of results.csv (default round_0)")
    parser.add_argument("--only", help="comma-separated stages to run instead of settings.stages")
    args = parser.parse_args()

    run_folder = os.path.abspath(args.run)
    scores = os.path.abspath(args.scores or os.path.join(run_folder, "scores"))
    os.makedirs(scores, exist_ok=True)
    logger = setup_logging(scores, "run")
    settings = load_settings(args.settings)
    stages = args.only.split(",") if args.only else list(settings.stages)
    logger.info("== scoring %s -> %s, stages %s", run_folder, scores, stages)
    log_environment(logger)

    for folder, what in ((settings.model_dir, "model_dir"), (settings.devset, "devset"), (run_folder, "run")):
        if not os.path.isdir(folder):
            logger.critical("%s does not exist: %s", what, folder)
            sys.exit(2)
    shutil.copyfile(args.settings, os.path.join(scores, "settings.yaml"))
    with open(args.settings, "rb") as f:
        settings_sha = hashlib.sha256(f.read()).hexdigest()

    environment = stage_environment(settings)
    record = {"started": now(), "run": run_folder, "code_commit": git_commit(), "settings_sha256": settings_sha,
              "models": {name: f"{repo.repo_id}@{repo.revision}" for name, repo in settings.models.named()},
              "stages": []}
    errors_before = len(read_jsonl(os.path.join(scores, "errors.jsonl")))
    for stage in stages:
        logger.info("== stage %s: starting (its log: %s)", stage, os.path.join(scores, "logs", f"{stage}.log"))
        started = time.time()
        result = subprocess.run([sys.executable, "-m", f"scoring.stages.{stage}",
                                 "--settings", os.path.abspath(args.settings), "--run", run_folder, "--scores", scores],
                                cwd=MIDTERM, env=environment)
        minutes = (time.time() - started) / 60
        status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        (logger.info if result.returncode == 0 else logger.error)("== stage %s: %s after %.1f min", stage, status, minutes)
        record["stages"].append({"stage": stage, "exit_code": result.returncode, "minutes": round(minutes, 2)})
        write_json(os.path.join(scores, "scoring_run.json"), record)

    result = subprocess.run([sys.executable, "-m", "scoring.results", "--settings", os.path.abspath(args.settings),
                             "--scores", scores, "--system", args.system], cwd=MIDTERM, env=environment)
    record["results_exit_code"] = result.returncode
    record["finished"] = now()
    write_json(os.path.join(scores, "scoring_run.json"), record)

    new_errors = len(read_jsonl(os.path.join(scores, "errors.jsonl"))) - errors_before
    failed = [s["stage"] for s in record["stages"] if s["exit_code"] != 0]
    logger.info("== done. results: %s", os.path.join(scores, "results.csv"))
    if new_errors:
        logger.warning("%d item failures in this attempt: see %s", new_errors, os.path.join(scores, "errors.jsonl"))
    if failed or result.returncode != 0:
        logger.error("stages that failed: %s. Fix the cause (see their logs), then run the same command again: "
                     "finished work is kept.", failed or ["results"])
        sys.exit(1)


if __name__ == "__main__":
    main()

# CAREFUL R1  Import order: midterm/ (scoring), the repo root (nat.py, score_hi_ta.py, normalize_hi_ta.py),
#             round0/code (pipeline, for the captioner), SIGMOS, then psp-eval (its package is named
#             `evaluation`, so nothing of ours may use that name).
