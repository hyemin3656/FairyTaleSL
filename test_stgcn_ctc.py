import argparse
import copy
import json
import time
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate

from mmaction.apis import init_recognizer
from mmaction.utils import register_all_modules

from src.build_keypoint_sample import (
    build_keypoint_sample_from_dir_or_arr,
    build_keypoint_sample_from_total_npy,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_CONFIG = (
    WORKSPACE_ROOT
    / "mmaction2/configs/skeleton/stgcn/stgcn_ctc_sign.py"
)
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "checkpoints/best_wer_epoch_85.pth"
DEFAULT_KEYPOINT_DIR = PROJECT_ROOT / "examples/sequenced_key_points/00_004"
DEFAULT_LABEL_MAP = PROJECT_ROOT / "src/class_labels.json"

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
        "--total-keypoint-npy",
        default=None,
        help=(
            "Optional single keypoint npy file. Expected shape is "
            "[T, 65, 3] or [1, T, 65, 3]. If set, --keypoint-dir is ignored."))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on. Use 'auto' to prefer CUDA when available.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. model.cls_head.dropout=0.0.")
    return parser.parse_args()


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


def predict_from_sample(model, pipeline, sample, label_map, device, warmup=0, repeat=1):
    start = time.perf_counter()
    data, data_batch = build_data_batch(sample, pipeline)
    pipeline_latency = time.perf_counter() - start

    if repeat < 1:
        raise ValueError("--repeat must be at least 1")

    with torch.no_grad():
        for _ in range(warmup):
            model.test_step(data_batch)
        synchronize(device)

        start = time.perf_counter()
        for _ in range(repeat):
            model_result = model.test_step(data_batch)[0]
        synchronize(device)
        elapsed = time.perf_counter() - start
    model_latency = elapsed / repeat

    if not hasattr(model_result, "pred_gloss"):
        raise AttributeError(
            "Model output has no `pred_gloss`. Check that the config uses "
            "STGCNCTCHead and the checkpoint matches the CTC model.")

    gloss_ids = to_int_list(model_result.pred_gloss)
    gloss_labels = map_gloss_ids(gloss_ids, label_map)

    return {
        "input_shape": tuple(data["inputs"].shape),
        "pipeline_latency": pipeline_latency,
        "model_latency": model_latency,
        "gloss_ids": gloss_ids,
        "gloss_labels": gloss_labels,
    }


def predict_from_each_keypoint(
    model,
    pipeline,
    label_map,
    device,
    keypoint_dir=None,
    arrs = None,
    warmup=0,
    repeat=1,
):
    if not keypoint_dir and not arrs:
        raise ValueError("Either keypoint_dir or arrs must be provided.")
    start = time.perf_counter()
    sample = build_keypoint_sample_from_dir_or_arr(keypoint_dir=keypoint_dir, arrs=arrs)
    keypoint_build_latency = time.perf_counter() - start

    pred = predict_from_sample(
        model=model,
        pipeline=pipeline,
        sample=sample,
        label_map=label_map,
        device=device,
        warmup=warmup,
        repeat=repeat)
    pred["keypoint_build_latency"] = keypoint_build_latency
    return pred


def predict_from_total_keypoint_npy(
    model,
    pipeline,
    keypoint_npy,
    label_map,
    device,
    warmup=0,
    repeat=1,
):
    start = time.perf_counter()
    sample = build_keypoint_sample_from_total_npy(keypoint_npy)
    keypoint_build_latency = time.perf_counter() - start

    pred = predict_from_sample(
        model=model,
        pipeline=pipeline,
        sample=sample,
        label_map=label_map,
        device=device,
        warmup=warmup,
        repeat=repeat)
    pred["keypoint_build_latency"] = keypoint_build_latency
    return pred

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

    pipeline = Compose(cfg.test_pipeline)
    label_map = load_label_map(args.label_map)

    if args.total_keypoint_npy is not None:
        # If keypoints are concatenated into a single npy, use that directly. Otherwise, load from the directory of separate npy files.
        sample_path = Path(args.total_keypoint_npy).expanduser().resolve()
        pred = predict_from_total_keypoint_npy(
            model=model,
            pipeline=pipeline,
            keypoint_npy=sample_path,
            label_map=label_map,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat)
    else:
        sample_path = Path(args.keypoint_dir).expanduser().resolve()
        pred = predict_from_each_keypoint(
            model=model,
            pipeline=pipeline,
            keypoint_dir=sample_path,
            label_map=label_map,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat)

    preprocess_latency = pred["keypoint_build_latency"]
    pipeline_latency = pred["pipeline_latency"]
    model_latency = pred["model_latency"]
    gloss_ids = pred["gloss_ids"]
    gloss_labels = pred["gloss_labels"]
    total_latency = preprocess_latency + pipeline_latency + model_latency

    print("Sample:", sample_path)
    print("Device:", device)
    print("Input shape:", pred["input_shape"])
    print("Preprocess time: {:.3f} ms".format(preprocess_latency * 1000))
    print("Pipeline time: {:.3f} ms".format(pipeline_latency * 1000))
    print("Model prediction time: {:.3f} ms".format(model_latency * 1000))
    print("total_latency: {:.3f} ms".format(total_latency * 1000))
    print("Pred gloss ids:", gloss_ids)
    print("Pred gloss labels:", gloss_labels)


if __name__ == "__main__":
    main()
