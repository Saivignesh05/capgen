#!/bin/bash
# Feature extraction on Madhava, one GPU per array task, fully offline.
#   cd ~/features-extractor && mkdir -p logs
#   sbatch slurm/extract.sbatch /path/to/hi        # 4 shards in parallel (see --array)
#   python -m src merge /path/to/hi --run job<JOBID>   # afterwards: builds + copies tool_descriptions.json
# All shards of one job write into the same run: features-extractor-results/<DIR name>/job<JOBID>/.
# Continue a job that hit its time limit: sbatch slurm/extract.sbatch /path/to/hi job<OLD JOBID>
# Set the GPU partition below (and add #SBATCH --account=... if your group needs one).
#SBATCH --job-name=fx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/fx_%A_%a.out

set -euo pipefail
FOLDER=${1:?usage: sbatch slurm/extract.sbatch /path/to/clip_folder [run-to-resume]}
RESUME=${2:-}
ENV=${FX_ENV:-$HOME/envs/fx}

source "$ENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # never try the network
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

I=${SLURM_ARRAY_TASK_ID:-0}
N=$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))
echo "host $(hostname)  shard $I/$N  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo no-gpu)"
cd "$SLURM_SUBMIT_DIR"
if [ -n "$RESUME" ]; then RUN=(--resume "$RESUME"); else RUN=(--run-id "job${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"); fi
exec python -m src run "$FOLDER" --shard "$I/$N" "${RUN[@]}"   # exec: python gets SLURM's SIGTERM
