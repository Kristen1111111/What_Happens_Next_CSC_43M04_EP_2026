# LLM Quick Context

Purpose: fast restart note for future assistant sessions.

## Project Goal

Train and evaluate a video action classifier from extracted image frames (33 classes), then generate a submission CSV.

## Main Entry Points

- src/train.py: training loop, internal train/val split from train_dir, saves best checkpoint.
- src/evaluate.py: evaluates on full dataset.val_dir, reports top-1 and top-5.
- src/create_submission.py: runs inference on test videos and writes submission CSV.

## Model And Data Pipeline

- Input clip shape: (B, T, C, H, W).
- Dataset loader: src/dataset/video_dataset.py.
- Models:
  - src/models/cnn_baseline.py: ResNet18 per frame + temporal average pooling.
  - src/models/cnn_lstm.py: ResNet18 per frame + LSTM over time.
- Shared utils: src/utils.py (seed, transforms, top-k helper, split).

## Hydra Config Layout

- Root config: src/configs/config.yaml
- Groups:
  - src/configs/data/default.yaml
  - src/configs/train/default.yaml
  - src/configs/model/cnn_baseline.yaml
  - src/configs/model/cnn_lstm.yaml
  - src/configs/experiment/baseline_from_scratch.yaml
  - src/configs/experiment/baseline_pretrained.yaml

Important: current data defaults point to processed_data/val2/{train,val,test}.

## Typical Commands

From repo root:

- python src/train.py experiment=baseline_from_scratch
- python src/evaluate.py training.checkpoint_path=best_model.pt
- python src/create_submission.py training.checkpoint_path=best_model.pt

From src folder (equivalent style):

- python train.py experiment=baseline_from_scratch
- python evaluate.py training.checkpoint_path=best_model.pt
- python create_submission.py training.checkpoint_path=best_model.pt

## Checkpoint Contract

Best checkpoint contains model_state_dict plus full Hydra config under key "config".
evaluate.py and create_submission.py rebuild the model from this saved config.

## Data Preparation Utilities

- src/misc/preprocess_ssv2.py: builds processed_data from raw videos and annotations.
- src/misc/download_data.py: Kaggle API helper script.

## Known Notes

- README examples mention processed_data/{train,val,test}, while active config uses processed_data/val2/... by default.
- TEST.py is a simple placeholder print script.
- Last observed terminal status: uv sync exited with code 1 (reason not captured here).

## Fast Resume Checklist

1. Confirm Python env and dependency install status.
2. Verify dataset paths in src/configs/data/default.yaml.
3. Check that checkpoint path in src/configs/train/default.yaml exists.
4. Run train/evaluate/submission with Hydra overrides as needed.
