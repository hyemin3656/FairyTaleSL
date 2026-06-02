import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
MMACTION_ROOT = WORKSPACE_ROOT / "mmaction2"
TOOL_ROOT = PROJECT_ROOT / "tool"

if str(MMACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(MMACTION_ROOT))
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from mmaction.apis import init_recognizer  # noqa: E402
from mmaction.utils import register_all_modules  # noqa: E402
from create_mediapipe_sign_ann import ensure_tvc as annotation_ensure_tvc  # noqa: E402


DEFAULT_CONFIG = (
    MMACTION_ROOT
    / "configs/skeleton/cnn1d/"
    / "cnn1d_8xb16-joint-u100-50e_mediapipe-sign-keypoint-3d_without_face.py"
)
DEFAULT_CHECKPOINT = (
    MMACTION_ROOT
    / "work_dirs/cnn1d_8xb16-joint-u100-50e_mediapipe-sign-keypoint-3d_without_face/"
    / "20260602_193745/best_acc_top1_epoch_36.pth"
)
DEFAULT_LABEL_MAP = PROJECT_ROOT / "src/class_labels.json"

NUM_POSE_FULL = 33
NUM_POSE_USED = 23
NUM_HAND = 21
NUM_NODE = NUM_POSE_USED + NUM_HAND + NUM_HAND
COORD_DIM = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CNN1D prediction from one MediaPipe holistic npz file.")
    parser.add_argument("npz", help="Path to npz with pose, left_hand, right_hand.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, or a torch device such as cuda:0.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. model.backbone.dropout=0.")
    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is not available. Falling back from {device} to cpu.")
        return "cpu"
    return device


def load_npz_arrays(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    required = ["pose", "left_hand", "right_hand"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing arrays in {npz_path}: {missing}")

    pose = annotation_ensure_tvc(data["pose"], NUM_POSE_FULL, "pose")
    left = annotation_ensure_tvc(data["left_hand"], NUM_HAND, "left_hand")
    right = annotation_ensure_tvc(data["right_hand"], NUM_HAND, "right_hand")

    return pose, left, right


def build_sample_from_npz(npz_path):
    pose, left, right = load_npz_arrays(npz_path)
    pose = pose[:, :NUM_POSE_USED]

    arrays = {
        "pose": pose,
        "left_hand": left,
        "right_hand": right,
    }
    for name, arr in arrays.items():
        if np.isnan(arr).any():
            raise ValueError(f"{Path(npz_path).stem}[{name}] contains NaN")

    t_values = {name: arr.shape[0] for name, arr in arrays.items()}
    if len(set(t_values.values())) != 1:
        raise ValueError(
            f"{Path(npz_path).stem} has inconsistent T: "
            + ", ".join([f"{name}={t}" for name, t in t_values.items()]))

    keypoints = {name: arr[..., :-1] for name, arr in arrays.items()}
    scores = {name: arr[..., -1] for name, arr in arrays.items()}
    scores["pose"] = np.ones(scores["pose"].shape, dtype=np.float32)

    keypoint = np.concatenate(
        [keypoints["pose"], keypoints["left_hand"], keypoints["right_hand"]],
        axis=1)
    keypoint_score = np.concatenate(
        [scores["pose"], scores["left_hand"], scores["right_hand"]],
        axis=1)

    total_frames = keypoint.shape[0]
    if keypoint.shape != (total_frames, NUM_NODE, COORD_DIM):
        raise ValueError(f"Unexpected keypoint shape: {keypoint.shape}")
    if keypoint_score.shape != (total_frames, NUM_NODE):
        raise ValueError(f"Unexpected keypoint_score shape: {keypoint_score.shape}")

    return {
        "frame_dir": Path(npz_path).stem,
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
    device = resolve_device(args.device)
    npz_path = Path(args.npz).expanduser().resolve()

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(
        cfg,
        checkpoint=str(Path(args.checkpoint).expanduser().resolve()),
        device=device)

    sample = build_sample_from_npz(npz_path)

    preprocess_start = time.perf_counter()
    data = Compose(cfg.test_pipeline)(sample)
    data_batch = pseudo_collate([data])
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

    result, avg_latency = benchmark(model, data_batch, device, args.warmup, args.repeat)
    predictions = format_predictions(
        result.pred_score, load_label_map(args.label_map), args.topk)

    print("NPZ:", npz_path)
    print("Using device:", device)
    print("Sample frames:", sample["total_frames"])
    print("MMACTION input shape:", tuple(data["inputs"].shape))
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
