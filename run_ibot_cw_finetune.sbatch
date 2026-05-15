#!/bin/bash
#SBATCH --job-name=ibot_cw_finetune
#SBATCH --partition=SallesInfo
#SBATCH --nodelist=traquet
#SBATCH --cpus-per-task=4
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
python train2_ibot_ready.py experiment=ibot_vit_video_closed_world_finetune \
  training.save_training_state=false \
  training.checkpoint_path=/Data/kristen.boitier/checkpoints/ibot_closed_world_video_best.pt \
  training.num_workers=2 \
  training.persistent_workers=false \
  training.prefetch_factor=2 \
  training.eval_batch_size=2 \
  hydra.run.dir=. \
  hydra.output_subdir=null
