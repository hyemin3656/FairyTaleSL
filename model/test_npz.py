from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Union

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_LABEL_MAP = PROJECT_ROOT / "src/class_labels.json"
DEFAULT_CONFIG = PROJECT_ROOT / "model/configs/cnn1d_mediapipe_sign_without_face.py"

from model.builder import build_model
from model.config_utils import load_config
from model.model import load_checkpoint
from model.data import build_npz_sample, preprocess_keypoint_sample
from model.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one MediaPipe npz with standalone CNN1D.")
    parser.add_argument("npz")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def load_label_map(path: Union[str, Path]) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    model = build_model(cfg).to(device)
    info = load_checkpoint(model, args.checkpoint, map_location=device, strict=False)
    if info["missing_keys"] or info["unexpected_keys"]:
        print("missing=", info["missing_keys"], "unexpected=", info["unexpected_keys"])

    sample = build_npz_sample(args.npz)
    keypoint = preprocess_keypoint_sample(
        sample,
        clip_len=cfg.CLIP_LEN,
        num_clips=cfg.TEST_NUM_CLIPS,
        test_mode=True,
        zero_pad_short=getattr(cfg, "ZERO_PAD_SHORT", False),
        input_mode=getattr(cfg, "INPUT_MODE", "xy"),
        keypoint_normalize=getattr(cfg, "KEYPOINT_NORMALIZE", None),
        random_horizontal_flip=getattr(cfg, "RANDOM_HORIZONTAL_FLIP", None),
        short_sample_interpolation=getattr(cfg, "SHORT_SAMPLE_INTERPOLATION", None),
    )
    inputs = torch.from_numpy(keypoint[None]).to(device)
    label_map = load_label_map(args.label_map)
    model.eval()
    with torch.no_grad():
        scores = model.predict(inputs)[0]

    values, indices = scores.topk(min(args.topk, scores.numel()))
    print("NPZ:", Path(args.npz))
    print("frames:", sample["total_frames"])
    for rank, (class_id, score) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        label = label_map.get(str(class_id), str(class_id))
        print(f"top{rank}: class_id={class_id} label={label} score={score:.6f}")


if __name__ == "__main__":
    main()

