# Standalone Skeleton Models

This folder runs MediaPipe sign skeleton models without MMAction2, mmengine, or mmcv.
The default config is CNN1D and matches the previous MMAction2 config:

- `SkeletonCNN1D(in_channels=2, num_joints=65, hidden_channels=(64, 128, 64), dropout=0.1)`
- `GCNHead(num_classes=67, in_channels=64, dropout=0.5)`
- annotation keypoints are sampled to `clip_len=100`
- validation/test use `num_clips=5`
- `GenSkeFeat(feats=['j'])` behavior is reproduced by keeping only x/y channels

## Train

Run from the FairyTaleSL directory:

```bash
cd /home/ubuntu/FairyTaleSL
python -m model.train \
  --ann-file ../dataset/cropped_holistic_results_interpolated_split/mediapipe_sign_3d_without_face_pose_score_1.pkl \
  --work-dir work_dirs/cnn1d_standalone \
  --device auto
```

Outputs are saved under a run folder named by execution time:

```text
work_dirs/cnn1d_standalone/YYYYMMDD_HHMMSS/
  best.pth
  last.pth
  train.log
  train_args.json
```

The log records each epoch in a compact train/val/checkpoint block. When `best.pth`
is updated, that epoch includes a `best.pth saved` line, and training ends with
the final best epoch summary.

Weights & Biases logging is enabled by default and records `train/loss`,
`train/top1_acc`, `val/loss`, and `val/top1_acc` by epoch. Disable it with:

```bash
python -m model.train --no-wandb
```

You can initialize from an old MMAction2 CNN1D checkpoint because the local
module names are kept as `backbone.*` and `cls_head.*`:

```bash
python -m model.train --resume /path/to/best_acc_top1_epoch_36.pth
```

## BiLSTM

BiLSTM uses the same train/eval/predict scripts with a different config:

```bash
python -m model.train \
  --config model/configs/bilstm_mediapipe_sign_without_face.py
```

Evaluate a BiLSTM checkpoint with the same config:

```bash
python -m model.eval work_dirs/bilstm_standalone/YYYYMMDD_HHMMSS/best.pth \
  --config model/configs/bilstm_mediapipe_sign_without_face.py \
  --split test
```

Predict one npz with a BiLSTM checkpoint:

```bash
python -m model.test_npz examples/00_07.npz work_dirs/bilstm_standalone/YYYYMMDD_HHMMSS/best.pth \
  --config model/configs/bilstm_mediapipe_sign_without_face.py
```

CNN1D remains the default, so the old command still trains CNN1D:

```bash
python -m model.train
```

## LSTM

LSTM is the unidirectional version of BiLSTM and uses the same scripts:

```bash
python -m model.train \
  --config model/configs/lstm_mediapipe_sign_without_face.py
```

Evaluate an LSTM checkpoint:

```bash
python -m model.eval work_dirs/lstm_standalone/YYYYMMDD_HHMMSS/best.pth \
  --config model/configs/lstm_mediapipe_sign_without_face.py \
  --split test
```

Predict one npz with an LSTM checkpoint:

```bash
python -m model.test_npz examples/00_07.npz work_dirs/lstm_standalone/YYYYMMDD_HHMMSS/best.pth \
  --config model/configs/lstm_mediapipe_sign_without_face.py
```

## CNN+LSTM

CNN+LSTM first extracts local temporal features with Conv1d, then feeds the
resulting feature sequence into an LSTM:

```bash
python -m model.train \
  --config model/configs/cnn_lstm_mediapipe_sign_without_face.py
```

Evaluate a CNN+LSTM checkpoint:

```bash
python -m model.eval work_dirs/cnn_lstm_standalone/YYYYMMDD_HHMMSS/best.pth \
  --config model/configs/cnn_lstm_mediapipe_sign_without_face.py \
  --split test
```

Predict one npz with a CNN+LSTM checkpoint:

```bash
python -m model.test_npz examples/00_07.npz work_dirs/cnn_lstm_standalone/YYYYMMDD_HHMMSS/best.pth \
  --config model/configs/cnn_lstm_mediapipe_sign_without_face.py
```

## Evaluate

```bash
python -m model.eval work_dirs/cnn1d_standalone/YYYYMMDD_HHMMSS/best.pth \
  --ann-file ../dataset/cropped_holistic_results_interpolated_split/mediapipe_sign_3d_without_face_pose_score_1.pkl \
  --split test
```

## Predict One NPZ

```bash
python -m model.test_npz examples/00_07.npz work_dirs/cnn1d_standalone/YYYYMMDD_HHMMSS/best.pth \
  --label-map src/class_labels.json \
  --topk 5
```

## Input Channels

Set `INPUT_MODE` in the config to choose keypoint channels:

```python
INPUT_MODE = "xy"        # x, y -> 2 channels
INPUT_MODE = "xyz"       # x, y, z -> 3 channels
INPUT_MODE = "xyscore"   # x, y, score -> 3 channels
INPUT_MODE = "xyzscore"  # x, y, z, score -> 4 channels
```

`IN_CHANNELS` is kept for compatibility, but model input channels are derived
from `INPUT_MODE` when it is set.

## Config

Edit `model/configs/cnn1d_mediapipe_sign_without_face.py` or
`model/configs/bilstm_mediapipe_sign_without_face.py`, or
`model/configs/lstm_mediapipe_sign_without_face.py`, or
`model/configs/cnn_lstm_mediapipe_sign_without_face.py` for model and training settings.
