#!/usr/bin/env bash
# SLURM job for the midterm pipeline. Edit the lines marked EDIT, then submit from the repo root:
#
#     sbatch midterm/slurm/run_midterm.sbatch                         # MODE=split (the default)
#     sbatch --export=ALL,MODE=main midterm/slurm/run_midterm.sbatch  # or another mode
#
# MODE        what runs                                                       results.csv you submit
# split       run A (60 clips x 1 sample, every measure but speaker          $RUNS/results.csv
#             diversity), then run B (20-clip subset x 10 samples, speaker
#             diversity only), then merges them. About 3 h.
# main        run A only                                                     $RUNS/main/<run>/scores/results.csv
# diversity   run B only (builds the subset first if it does not exist)      $RUNS/diversity/<run>/scores/results.csv
# full        60 clips x 10 samples, every stage, in one run. About 5-7 h.   $RUNS/full/<run>/scores/results.csv
#
# Needs about 70 GB of GPU memory in total (the Qwen3-Omni captioner): one 80 GB A100/H100 or two
# 40 GB A100s. V100s do not work (no bfloat16). If the time limit kills the job, resubmit the same
# MODE with RESUME=1: every finished item is kept.
#
#SBATCH --job-name=midterm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                # EDIT: gpu:2 for two 40 GB A100s
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=midterm-%j.out     # every log line of the job: round 0 and scoring
#SBATCH --error=midterm-%j.out
##SBATCH --partition=EDIT           # EDIT: your GPU partition (remove one leading # to use it)

set -uo pipefail
REPO=/full/path/to/capgen           # EDIT: the repo on the HPC (git checkout of the midterm branch)
ENVS=/full/path/to/envs             # EDIT: the folder holding the asr-captioner, generator and scoring envs
RUNS=/full/path/to/runs             # EDIT: where runs go (one subfolder per MODE)
MODE="${MODE:-split}"
RESUME="${RESUME:-0}"               # 1: continue the latest run of each part instead of starting new ones

log() { echo "$(date '+%F %T') [sbatch] $*"; }
log "job ${SLURM_JOB_ID:--} on $(hostname), MODE=$MODE RESUME=$RESUME"
[ -d "$REPO/midterm" ] || { log "ERROR: REPO=$REPO is not the repo (no midterm/)"; exit 1; }
cd "$REPO"

# One part = one call of run_midterm.sh: part <runs subfolder> <round 0 settings> <clips> <scoring stages>
part() {
    local folder="$RUNS/$1"
    log "---- part $1: settings=$2 clips=$3 stages=$4"
    if [ "$RESUME" = "1" ] && [ -f "$folder/latest.json" ]; then
        RESUME_RUN=latest ROUND0_SETTINGS="$2" CLIPS="$3" SCORING_ONLY="$4" bash midterm/run_midterm.sh "$ENVS" "$folder"
    else
        ROUND0_SETTINGS="$2" CLIPS="$3" SCORING_ONLY="$4" bash midterm/run_midterm.sh "$ENVS" "$folder"
    fi
}
latest_results() {     # results.csv of the latest run in a runs subfolder
    local run_id
    run_id="$("$ENVS/scoring/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$RUNS/$1/latest.json")" || return 1
    echo "$RUNS/$1/$run_id/scores/results.csv"
}
make_subset() {
    if [ -d "$REPO/devset_div" ]; then
        log "subset exists: $REPO/devset_div (delete it to choose again)"
    else
        (cd midterm && "$ENVS/scoring/bin/python" -m scoring.diversity subset --devset ../devset --out ../devset_div --per-language 10)
    fi
}

MAIN_STAGES=intelligibility,naturalness,caption_generated,judge
status=0
case "$MODE" in
    main)
        part main round0/settings.yaml devset "$MAIN_STAGES" || status=$? ;;
    diversity)
        make_subset || exit 1
        part diversity round0/settings_eval.yaml devset_div speaker_diversity || status=$? ;;
    full)
        part full round0/settings_eval.yaml devset intelligibility,speaker_diversity,naturalness,caption_generated,judge || status=$? ;;
    split)
        make_subset || exit 1
        part main round0/settings.yaml devset "$MAIN_STAGES" || status=$?
        part diversity round0/settings_eval.yaml devset_div speaker_diversity || status=$?
        main_csv="$(latest_results main)"; diversity_csv="$(latest_results diversity)"
        if [ -f "$main_csv" ] && [ -f "$diversity_csv" ]; then
            (cd midterm && "$ENVS/scoring/bin/python" -m scoring.diversity merge \
                --main "$main_csv" --diversity "$diversity_csv" --out "$RUNS/results.csv") || status=1
            [ -f "$RUNS/results.csv" ] && { log "merged results.csv: $RUNS/results.csv"; cat "$RUNS/results.csv"; }
        else
            log "ERROR: cannot merge, missing $main_csv or $diversity_csv"; status=1
        fi ;;
    *)
        log "ERROR: unknown MODE=$MODE (split, main, diversity or full)"; exit 1 ;;
esac

log "job ${SLURM_JOB_ID:--} finished with exit code $status"
exit $status
