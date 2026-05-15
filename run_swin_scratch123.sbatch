#!/bin/bash
#SBATCH --job-name=swin_scratch123
#SBATCH --partition=SallesInfo
#SBATCH --nodelist=nandou
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/Data/kristen.boitier/slurm_logs/%x-%j.out
#SBATCH --error=/Data/kristen.boitier/slurm_logs/%x-%j.err

cd ~/Documents/MODAL/What_Happens_Next_CSC_43M04_EP_2026-1
source .venv/bin/activate
rm -rf outputs src/outputs
rm -f src/checkpoints/*.tmp
mkdir -p /Data/kristen.boitier/checkpoints

cd src
python train2_kaggle_best.py experiment=video_swin_scratch_seed123 \
  training.save_training_state=false \
  training.checkpoint_path=/Data/kristen.boitier/checkpoints/video_swin_scratch_t_seed123_best.pt \
  hydra.run.dir=. \
  hydra.output_subdir=null
