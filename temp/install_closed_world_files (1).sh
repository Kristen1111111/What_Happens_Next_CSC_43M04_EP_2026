#!/usr/bin/env bash
set -euo pipefail

mkdir -p src/configs/experiment

cp video_swin_scratch_essai.yaml src/configs/experiment/video_swin_scratch_essai.yaml
cp video_swin_scratch_seed123.yaml src/configs/experiment/video_swin_scratch_seed123.yaml
cp ibot_pretrain_closed_world.yaml src/configs/experiment/ibot_pretrain_closed_world.yaml
cp ibot_vit_video_closed_world_finetune.yaml src/configs/experiment/ibot_vit_video_closed_world_finetune.yaml
cp closed_world_submission_ensemble.yaml src/configs/experiment/closed_world_submission_ensemble.yaml
cp run_closed_world_from_scratch.sh run_closed_world_from_scratch.sh
chmod +x run_closed_world_from_scratch.sh
echo "Closed-world configs installed."
