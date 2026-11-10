#!/bin/bash
# Round 0 captioner on the dev set. Submit from the repo root:
#     sbatch round0/captioner.sbatch
# Madhava gpu partition: 2 x H100 per node, 12-hour limit. The BF16 model (~63 GB) fits on one H100.

#SBATCH --job-name=r0_captioner
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=round0/logs/%x_%j.out
#SBATCH --error=round0/logs/%x_%j.err

set -e
cd $SLURM_SUBMIT_DIR

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/$USER/hf_cache

module load anaconda3-2024
source /scratch/$USER/envs/omni/bin/activate

nvidia-smi
python round0/code/captioner.py --audio_dir devset --out_dir round0/outputs/devset
