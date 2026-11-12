#!/usr/bin/env bash
# SLURM job for the whole midterm pipeline. Edit the lines marked EDIT, then:
#
#     sbatch midterm/slurm/run_midterm.sbatch
#
# Needs about 70 GB of GPU memory in total (the Qwen3-Omni captioner): one 80 GB A100/H100 or two
# 40 GB A100s. V100s do not work (no bfloat16). The full devset run (60 clips x 10 samples, then
# scoring) takes a few hours; if the time limit kills it, resubmit with RESUME_RUN=latest.
#
#SBATCH --job-name=midterm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                # EDIT: gpu:2 for two 40 GB A100s
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=midterm-%j.out     # all logs of the job: round 0 and scoring
#SBATCH --error=midterm-%j.out
##SBATCH --partition=EDIT           # EDIT: your GPU partition

set -uo pipefail
REPO=/full/path/to/capgen           # EDIT: the repo on the HPC (git checkout of the midterm branch)
ENVS=/full/path/to/envs             # EDIT: where unpack_envs.sh unpacked asr-captioner, generator, scoring
RUNS=/full/path/to/runs             # EDIT: where round 0 runs go

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
bash "$REPO/midterm/run_midterm.sh" "$ENVS" "$RUNS"
status=$?
echo "job $SLURM_JOB_ID finished $(date) with exit code $status"
exit $status
