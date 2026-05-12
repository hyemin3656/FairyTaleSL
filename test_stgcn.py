import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate

from mmaction.apis import init_recognizer
from mmaction.utils import register_all_modules


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/stgcn/stgcn_8xb16-joint-u100-80e_mediapipe-sign-keypoint-3d.py"
)
DEFAULT_CHECKPOINT = (
    WORKSPACE_ROOT / "runyourai/ksl/baseline/best_acc_top1_epoch_24.pth"
)
DEFAULT_KEYPOINT_DIR = PROJECT_ROOT / "examples/key_points"
DEFAULT_LABEL_MAP = PROJECT_ROOT / "src/class_labels.json"

POSE_FILE = "pose_33.npy"
LEFT_HAND_FILE = "left_hand_21.npy"
RIGHT_HAND_FILE = "right_hand_21.npy"

NUM_POSE = 23
NUM_HAND = 21
NUM_NODE = 65
COORD_DIM = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one ST-GCN prediction from examples/key_points.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--keypoint-dir", default=str(DEFAULT_KEYPOINT_DIR))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. model.backbone.in_channels=4.")
    return parser.parse_args()


def load_npy(path):
    if not path.exists():
        return None
    return np.load(path)


def ensure_tvc(arr, expected_v, name):
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape [T, V, C], got {arr.shape}")
    if arr.shape[1] < expected_v:
        raise ValueError(
            f"{name} has too few keypoints: expected {expected_v}, "
            f"got {arr.shape[1]}")
    if arr.shape[2] < COORD_DIM:
        raise ValueError(
            f"{name} must have at least {COORD_DIM} coordinates, "
            f"got {arr.shape[2]}")
    return arr[:, :expected_v, :COORD_DIM].astype(np.float32)


def pad_or_trim_time(arr, target_t):
    t, v, c = arr.shape
    if t == target_t:
        return arr
    if t > target_t:
        return arr[:target_t]

    padded = np.zeros((target_t, v, c), dtype=arr.dtype)
    padded[:t] = arr
    return padded


def zero_hand(total_frames):
    return np.zeros((total_frames, NUM_HAND, COORD_DIM), dtype=np.float32)


def nan_to_zero_with_score(arr):
    invalid = np.isnan(arr).any(axis=-1)
    score = (~invalid).astype(np.float32)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[invalid] = 0.0
    return arr.astype(np.float32), score.astype(np.float32)


def build_keypoint_sample(keypoint_dir):
    keypoint_dir = Path(keypoint_dir).expanduser().resolve()

    pose = load_npy(keypoint_dir / POSE_FILE)
    left = load_npy(keypoint_dir / LEFT_HAND_FILE)
    right = load_npy(keypoint_dir / RIGHT_HAND_FILE)

    if pose is None:
        raise FileNotFoundError(f"Missing {POSE_FILE}: {keypoint_dir}")

    pose = ensure_tvc(pose, NUM_POSE, POSE_FILE)
    total_frames = pose.shape[0]

    if left is None:
        left = zero_hand(total_frames)
    else:
        left = ensure_tvc(left, NUM_HAND, LEFT_HAND_FILE)
        left = pad_or_trim_time(left, total_frames)

    if right is None:
        right = zero_hand(total_frames)
    else:
        right = ensure_tvc(right, NUM_HAND, RIGHT_HAND_FILE)
        right = pad_or_trim_time(right, total_frames)

    pose, pose_score = nan_to_zero_with_score(pose)
    left, left_score = nan_to_zero_with_score(left)
    right, right_score = nan_to_zero_with_score(right)

    keypoint = np.concatenate([pose, left, right], axis=1)
    keypoint_score = np.concatenate([pose_score, left_score, right_score], axis=1)

    assert keypoint.shape == (total_frames, NUM_NODE, COORD_DIM), keypoint.shape
    assert keypoint_score.shape == (total_frames, NUM_NODE), keypoint_score.shape

    return {
        "frame_dir": keypoint_dir.name,
        "total_frames": total_frames,
        "label": -1,
        "keypoint": keypoint[None, ...].astype(np.float32),
        "keypoint_score": keypoint_score[None, ...].astype(np.float32),
    }


def load_label_map(path):
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_predictions(pred_score, label_map, topk):
    topk = min(topk, pred_score.numel())
    scores, indices = pred_score.topk(topk)

    predictions = []
    for class_id, score in zip(indices.tolist(), scores.tolist()):
        predictions.append({
            "class_id": int(class_id),
            "label": label_map.get(str(int(class_id)), str(int(class_id))),
            "score": float(score),
        })
    return predictions


def synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(model, data_batch, device, warmup, repeat):
    with torch.no_grad():
        for _ in range(warmup):
            model.test_step(data_batch)
        synchronize(device)

        start = time.perf_counter()
        for _ in range(repeat):
            result = model.test_step(data_batch)[0]
        synchronize(device)
        elapsed = time.perf_counter() - start

    return result, elapsed / repeat


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(
        cfg,
        checkpoint=str(Path(args.checkpoint).expanduser().resolve()),
        device=args.device)

    sample = build_keypoint_sample(args.keypoint_dir)

    preprocess_start = time.perf_counter()
    data = Compose(cfg.test_pipeline)(sample)
    data_batch = pseudo_collate([data])
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

    result, avg_latency = benchmark(
        model,
        data_batch,
        args.device,
        warmup=args.warmup,
        repeat=args.repeat)

    label_map = load_label_map(args.label_map)
    predictions = format_predictions(result.pred_score, label_map, args.topk)

    print("Sample:", str(Path(args.keypoint_dir).expanduser().resolve()))
    print("Ground truth: unknown")
    print("Input shape:", tuple(data["inputs"].shape))
    print("Preprocess time: {:.3f} ms".format(preprocess_ms))
    print("Average inference time: {:.3f} ms".format(avg_latency * 1000))
    print("Throughput: {:.2f} samples/s".format(1.0 / avg_latency))
    print("Predictions:")
    for rank, pred in enumerate(predictions, start=1):
        print(
            f"  top{rank}: class_id={pred['class_id']} "
            f"label={pred['label']} score={pred['score']:.6f}")


if __name__ == "__main__":
    main()
