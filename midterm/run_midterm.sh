#!/usr/bin/env bash
# The whole midterm pipeline on the devset: round 0 (ASR -> captioner -> generator), then scoring,
# then results.csv. Run on a GPU node (see slurm/run_midterm.sbatch):
#
#     bash midterm/run_midterm.sh <envs folder> <runs folder>
#
# Optional environment variables:
#     ROUND0_SETTINGS   round 0 settings (default round0/settings_eval.yaml: 10 samples per clip,
#                       which speaker diversity needs; round0/settings.yaml makes 1)
#     SCORING_SETTINGS  scoring settings (default midterm/settings.yaml)
#     CLIPS             input clips (default devset/)
#     RESUME_RUN        a round 0 run id (or "latest") to continue instead of starting a new run
#     SKIP_ROUND0=1     with RESUME_RUN: only score that run, do not touch round 0
#     SCORING_ONLY      comma-separated scoring stages to run instead of all (e.g. speaker_diversity)
#
# Use a separate <runs> folder for each kind of run (e.g. runs/main and runs/diversity): each
# <runs> has one latest.json, and two jobs sharing it could pick up each other's run.
#
# Every step logs with a timestamp. Round 0's own outputs and errors are in <runs>/<run_id>/
# (errors.jsonl); scoring logs are in <runs>/<run_id>/scores/logs/. Re-running with RESUME_RUN
# continues where a killed job stopped.
set -uo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: bash midterm/run_midterm.sh <envs folder> <runs folder>" >&2
    exit 1
fi
ENVS="$(realpath "$1")"
RUNS="$(realpath -m "$2")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROUND0_SETTINGS="$(realpath "${ROUND0_SETTINGS:-$REPO/round0/settings_eval.yaml}")"
SCORING_SETTINGS="$(realpath "${SCORING_SETTINGS:-$REPO/midterm/settings.yaml}")"
CLIPS="$(realpath "${CLIPS:-$REPO/devset}")"
ROUND0_PY="$ENVS/asr-captioner/bin/python"
SCORING_PY="$ENVS/scoring/bin/python"

log() { echo "$(date '+%F %T') [run_midterm] $*"; }
fail() { log "ERROR: $*"; exit 1; }

log "repo=$REPO envs=$ENVS runs=$RUNS clips=$CLIPS"
log "round0 settings=$ROUND0_SETTINGS scoring settings=$SCORING_SETTINGS"
log "host=$(hostname) slurm_job=${SLURM_JOB_ID:--} commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo none)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read -r line; do log "gpu: $line"; done
for python in "$ROUND0_PY" "$ENVS/generator/bin/python" "$SCORING_PY"; do
    [ -x "$python" ] || fail "$python not found: unpack the environments first (round0/code/scripts/unpack_envs.sh)"
done
grep -q "^model_dir: /path/to" "$ROUND0_SETTINGS" && fail "set model_dir in $ROUND0_SETTINGS (round0/README.md, Step 3)"
grep -q "^model_dir: /path/to" "$SCORING_SETTINGS" && fail "set model_dir in $SCORING_SETTINGS"
mkdir -p "$RUNS"

# ---- 1. round 0 -------------------------------------------------------------------------------
cd "$REPO/round0/code" || fail "no round0/code"
ROUND0_STATUS=0
if [ "${SKIP_ROUND0:-0}" = "1" ]; then
    [ -n "${RESUME_RUN:-}" ] || fail "SKIP_ROUND0=1 needs RESUME_RUN"
    log "1/2 round 0 skipped (SKIP_ROUND0=1)"
elif [ -n "${RESUME_RUN:-}" ]; then
    log "1/2 round 0: resuming run $RESUME_RUN"
    "$ROUND0_PY" -m scripts.run resume --run "$RESUME_RUN" --runs "$RUNS" --environments "$ENVS" || ROUND0_STATUS=$?
else
    log "1/2 round 0: new run on $CLIPS"
    "$ROUND0_PY" -m scripts.run new --settings "$ROUND0_SETTINGS" --clips "$CLIPS" --runs "$RUNS" --environments "$ENVS" || ROUND0_STATUS=$?
fi
if [ -n "${RESUME_RUN:-}" ] && [ "$RESUME_RUN" != "latest" ]; then
    RUN_ID="$RESUME_RUN"
else
    RUN_ID="$("$ROUND0_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$RUNS/latest.json")" || fail "no $RUNS/latest.json: round 0 did not start"
fi
RUN_FOLDER="$RUNS/$RUN_ID"
[ -f "$RUN_FOLDER/clips.jsonl" ] || fail "$RUN_FOLDER is not a round 0 run"
if [ "$ROUND0_STATUS" -ne 0 ]; then
    log "WARNING: round 0 exited with $ROUND0_STATUS (incomplete). Scoring what finished; failures are in $RUN_FOLDER/errors.jsonl:"
    [ -f "$RUN_FOLDER/errors.jsonl" ] && tail -n 5 "$RUN_FOLDER/errors.jsonl"
    log "to finish round 0 later: RESUME_RUN=$RUN_ID bash midterm/run_midterm.sh $ENVS $RUNS"
fi

# ---- 2. scoring ---------------------------------------------------------------------------------
cd "$REPO/midterm" || fail "no midterm/"
log "2/2 scoring $RUN_FOLDER"
SCORING_STATUS=0
ONLY_ARGS=()
[ -n "${SCORING_ONLY:-}" ] && ONLY_ARGS=(--only "$SCORING_ONLY") && log "only these scoring stages: $SCORING_ONLY"
"$SCORING_PY" -m scoring.run --settings "$SCORING_SETTINGS" --run "$RUN_FOLDER" ${ONLY_ARGS[@]+"${ONLY_ARGS[@]}"} || SCORING_STATUS=$?

RESULTS="$RUN_FOLDER/scores/results.csv"
if [ -f "$RESULTS" ]; then
    log "results.csv: $RESULTS"
    cat "$RESULTS"
else
    log "ERROR: no results.csv; see $RUN_FOLDER/scores/logs/"
fi
[ "$ROUND0_STATUS" -eq 0 ] && [ "$SCORING_STATUS" -eq 0 ] || { log "finished WITH PROBLEMS (round0=$ROUND0_STATUS scoring=$SCORING_STATUS)"; exit 1; }
log "finished: everything ran"
