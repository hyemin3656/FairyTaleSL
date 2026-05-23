# FairyTaleSL Local Capture Client

Run this folder on your own laptop, not on the remote GPU server. It captures the
local webcam, extracts MediaPipe keypoints with the same parameters used during
training, applies the hand-detection crop rule, saves npy files, and uploads the
segment directory to the server with `scp`.

## Install

```bash
cd local_client
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-local.txt
```

If Python cannot install `mediapipe==0.10.13`, use a Python version supported by
that wheel. The Colab training notebook used MediaPipe 0.10.13.

## Run without upload

```bash
python capture_and_upload.py --display
```

## Run and upload to the GPU server

```bash
python capture_and_upload.py \
  --display \
  --upload-target ubuntu@YOUR_SERVER:/home/ubuntu/FairyTaleSL/realtime_inputs/
```

With an SSH key or non-default port:

```bash
python capture_and_upload.py \
  --display \
  --ssh-key ~/.ssh/id_rsa \
  --ssh-port 22 \
  --upload-target ubuntu@YOUR_SERVER:/home/ubuntu/FairyTaleSL/realtime_inputs/
```

The uploaded directory contains:

- pose_33.npy
- left_hand_21.npy
- right_hand_21.npy
- summary.csv

On the server, run inference with:

```bash
/opt/conda/envs/openmmlab/bin/python /home/ubuntu/FairyTaleSL/test_stgcn_ctc.py \
  --checkpoint /home/ubuntu/checkpoints/best_wer_epoch_85.pth \
  --keypoint-dir /home/ubuntu/FairyTaleSL/realtime_inputs/SAMPLE_DIR_NAME
```
