"""What every stage shares: its command line, logging and the summary line it ends with.

A stage is started by scoring.run as `python -m scoring.stages.<name> --settings S --run R --scores D`.
It reads the round 0 run R, appends one line per finished unit of work to D/<name>.jsonl, logs a
failure of one item and goes on, and exits non-zero only if it could not run at all (a model would
not load, say) or if every item failed.
"""
import argparse
import sys
import time

from scoring.common import load_run, log_environment, setup_logging
from scoring.settings import load_settings


def main(name: str, run) -> None:
    """Parse the command line, set up logging, call run(settings, items, scores_dir, run_folder, logger).
    `run` returns (done, failed) counts for this attempt."""
    parser = argparse.ArgumentParser(description=f"scoring stage {name}")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--run", required=True, help="round 0 run folder")
    parser.add_argument("--scores", required=True, help="folder for the scores of this run")
    args = parser.parse_args()

    logger = setup_logging(args.scores, name)
    started = time.time()
    logger.info("== stage %s starts: run=%s scores=%s", name, args.run, args.scores)
    log_environment(logger)
    settings = load_settings(args.settings)
    items = load_run(args.run, logger)

    done, failed = run(settings, items, args.scores, args.run, logger)

    minutes = (time.time() - started) / 60
    logger.info("== stage %s finished in %.1f min: %d done this attempt, %d failed (see errors.jsonl)", name, minutes, done, failed)
    if failed and not done:
        logger.error("every item of stage %s failed", name)
        sys.exit(1)
