import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate

from mmaction.apis import init_recognizer
from mmaction.utils import register_all_modules

from tool.create_mediapipe_sign_ann import (
    COORD_DIM, NUM_HAND, NUM_NODE, NUM_POSE, ensure_tvc, load_npy, nan_to_zero_with_score, pad_or_trim_time)


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_CONFIG = (
    WORKSPACE_ROOT
    / "mmaction2/configs/skeleton/stgcn/stgcn_ctc_sign.py"
)
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "checkpoints/best_wer_epoch_85.pth"
DEFAULT_KEYPOINT_DIR = PROJECT_ROOT / "examples/sequenced_key_points/00_004"
DEFAULT_LABEL_MAP = PROJECT_ROOT / "src/class_labels.json"

POSE_FILE = "pose_33.npy"
LEFT_HAND_FILE = "left_hand_21.npy"
RIGHT_HAND_FILE = "right_hand_21.npy"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one ST-GCN-BiLSTM-CTC prediction from keypoint npy files.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--keypoint-dir",
        default=str(DEFAULT_KEYPOINT_DIR),
        help=(
            "Directory containing pose_33.npy, left_hand_21.npy, "
            "and right_hand_21.npy."))
    parser.add_argument(
        "--keypoint-npy",
        default=None,
        help=(
            "Optional single keypoint npy file. Expected shape is "
            "[T, 65, 3] or [1, T, 65, 3]. If set, --keypoint-dir is ignored."))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on. Use 'auto' to prefer CUDA when available.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. model.cls_head.dropout=0.0.")
    return parser.parse_args()

def zero_hand(total_frames):
    return np.zeros((total_frames, NUM_HAND, COORD_DIM), dtype=np.float32)


def build_keypoint_sample_from_dir(keypoint_dir):
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

    return build_keypoint_sample(
        keypoint=keypoint,
        keypoint_score=keypoint_score,
        sample_name=keypoint_dir.name)


def build_keypoint_sample_from_npy(keypoint_npy):
    keypoint_npy = Path(keypoint_npy).expanduser().resolve()
    keypoint = np.load(keypoint_npy)
    keypoint = np.asarray(keypoint)

    if keypoint.ndim == 4:
        if keypoint.shape[0] != 1:
            raise ValueError(
                "A 4D --keypoint-npy must have shape [1, T, 65, 3], "
                f"got {keypoint.shape}")
        keypoint = keypoint[0]

    keypoint = ensure_tvc(keypoint, NUM_NODE, keypoint_npy.name)
    keypoint, keypoint_score = nan_to_zero_with_score(keypoint)

    return build_keypoint_sample(
        keypoint=keypoint,
        keypoint_score=keypoint_score,
        sample_name=keypoint_npy.stem)


def build_keypoint_sample(keypoint, keypoint_score, sample_name):
    total_frames = keypoint.shape[0]

    assert keypoint.shape == (total_frames, NUM_NODE, COORD_DIM), keypoint.shape
    assert keypoint_score.shape == (total_frames, NUM_NODE), keypoint_score.shape

    return {
        "frame_dir": sample_name,
        "total_frames": total_frames,
        "keypoint": keypoint[None, ...].astype(np.float32),
        "keypoint_score": keypoint_score[None, ...].astype(np.float32),
    }


def load_label_map(path):
    path = Path(path).expanduser()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_int_list(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def map_gloss_ids(gloss_ids, label_map):
    return [label_map.get(str(gloss_id), str(gloss_id)) for gloss_id in gloss_ids]


def synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_device(device):
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(
            f"Warning: requested {device}, but this PyTorch install has no CUDA. "
            "Falling back to cpu.")
        return "cpu"

    return device


def build_data_batch(sample, pipeline):
    data = pipeline(copy.deepcopy(sample))
    return data, pseudo_collate([data])


def run_prediction(model, data_batch, device, warmup, repeat):
    if repeat < 1:
        raise ValueError("--repeat must be at least 1")

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


def run_pipeline_prediction(model, sample, pipeline, device, warmup, repeat):
    if repeat < 1:
        raise ValueError("--repeat must be at least 1")

    with torch.no_grad():
        for _ in range(warmup):
            _, data_batch = build_data_batch(sample, pipeline)
            model.test_step(data_batch)
        synchronize(device)

        start = time.perf_counter()
        for _ in range(repeat):
            data, data_batch = build_data_batch(sample, pipeline)
            result = model.test_step(data_batch)[0]
        synchronize(device)
        elapsed = time.perf_counter() - start

    return result, data, elapsed / repeat


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    device = resolve_device(args.device)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(
        cfg,
        checkpoint=str(Path(args.checkpoint).expanduser().resolve()),
        device=device)

    if args.keypoint_npy is not None:
        # If keypoints are concatenated into a single npy, use that directly. Otherwise, load from the directory of separate npy files.
        sample = build_keypoint_sample_from_npy(args.keypoint_npy) 
        sample_path = Path(args.keypoint_npy).expanduser().resolve()
    else:
        sample = build_keypoint_sample_from_dir(args.keypoint_dir)
        sample_path = Path(args.keypoint_dir).expanduser().resolve()

    pipeline = Compose(cfg.test_pipeline)
    data, data_batch = build_data_batch(sample, pipeline)

    model_result, model_latency = run_prediction(
        model=model,
        data_batch=data_batch,
        device=device,
        warmup=args.warmup,
        repeat=args.repeat)
    result, data, pipeline_latency = run_pipeline_prediction(
        model=model,
        sample=sample,
        pipeline=pipeline,
        device=device,
        warmup=args.warmup,
        repeat=args.repeat)

    if not hasattr(result, "pred_gloss"):
        raise AttributeError(
            "Model output has no `pred_gloss`. Check that the config uses "
            "STGCNCTCHead and the checkpoint matches the CTC model.")

    gloss_ids = to_int_list(result.pred_gloss)
    label_map = load_label_map(args.label_map)
    gloss_labels = map_gloss_ids(gloss_ids, label_map)

    print("Sample:", sample_path)
    print("Device:", device)
    print("Input shape:", tuple(data["inputs"].shape))
    print("Model prediction time: {:.3f} ms".format(model_latency * 1000))
    print(
        "Preprocess + prediction time: {:.3f} ms".format(
            pipeline_latency * 1000))
    print(
        "Pipeline throughput: {:.2f} samples/s".format(
            1.0 / pipeline_latency))
    print("Pred gloss ids:", gloss_ids)
    print("Pred gloss labels:", gloss_labels)


if __name__ == "__main__":
    main()
