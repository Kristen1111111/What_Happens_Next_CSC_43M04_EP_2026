#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src"
source ../.venv/bin/activate

echo "[1/4] Train Video Swin scratch seed 42"
tmux new -d -s swin_scratch42 "python train2_kaggle_best.py experiment=video_swin_scratch_essai training.save_training_state=true 2>&1 | tee -a ../swin_scratch42.log"

echo "[2/4] Train Video Swin scratch seed 123"
tmux new -d -s swin_scratch123 "python train2_kaggle_best.py experiment=video_swin_scratch_seed123 training.save_training_state=true 2>&1 | tee -a ../swin_scratch123.log"

echo "[3/4] Pretrain iBOT closed-world"
tmux new -d -s ibot_cw_pretrain "python pretrain_ibot.py experiment=ibot_pretrain_closed_world 2>&1 | tee -a ../ibot_cw_pretrain.log"

echo "Launched three tmux sessions."
echo "Check with: tmux ls && nvidia-smi"
echo "After iBOT pretraining is finished, launch fine-tuning with:"
echo "tmux new -d -s ibot_cw_finetune \"python train2_ibot_ready.py experiment=ibot_vit_video_closed_world_finetune 2>&1 | tee -a ../ibot_cw_finetune.log\""
