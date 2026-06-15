from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    parser = argparse.ArgumentParser(description="Predict MediaPipe npz file(s) with standalone CNN1D.")
    parser.add_argument("npz", help="Path to one npz file or a directory containing npz files.")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--label-source",
        default="auto",
        choices=["auto", "npz", "stem", "none"],
        help="Where to read ground-truth labels. auto tries npz['label'], then integer filename stem.",
    )
    parser.add_argument(
        "--saliency-out",
        default=None,
        help="Path to save gradient saliency heatmap jpg. Defaults to <npz>.saliency.jpg.",
    )
    parser.add_argument(
        "--saliency-class",
        type=int,
        default=None,
        help="Class id to explain. Defaults to the predicted top-1 class.",
    )
    parser.add_argument(
        "--saliency-clip",
        type=int,
        default=0,
        help="Test clip index to visualize in the saliency heatmap. Defaults to 0.",
    )
    return parser.parse_args()



def to_int_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size != 1:
        return None
    try:
        return int(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def infer_label(npz_path: Union[str, Path], label_source: str) -> Optional[int]:
    npz_path = Path(npz_path).expanduser()
    if label_source in {"auto", "npz"}:
        with np.load(npz_path, allow_pickle=True) as data:
            if "label" in data:
                label = to_int_label(data["label"])
                if label is not None:
                    return label
        if label_source == "npz":
            return None

    if label_source in {"auto", "stem"}:
        try:
            return int(npz_path.stem)
        except ValueError:
            return None
    return None


def collect_npz_paths(path: Union[str, Path]) -> List[Path]:
    path = Path(path).expanduser()
    if path.is_dir():
        npz_paths = sorted(path.glob("*.npz"))
        if not npz_paths:
            raise FileNotFoundError(f"No npz files found in {path}")
        return npz_paths
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def predict_npz(
    model: torch.nn.Module,
    npz_path: Union[str, Path],
    cfg,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    sample = build_npz_sample(npz_path)
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
    with torch.no_grad():
        scores = model.predict(inputs)[0].detach().cpu()
    return sample, inputs, scores


def format_topk(scores: torch.Tensor, topk: int, label_map: Dict[str, str]) -> List[Dict[str, Any]]:
    values, indices = scores.topk(min(topk, scores.numel()))
    return [
        {
            "class_id": int(class_id),
            "label": label_map.get(str(int(class_id)), str(int(class_id))),
            "score": float(score),
        }
        for class_id, score in zip(indices.tolist(), values.tolist())
    ]


def compute_input_saliency(model, inputs: torch.Tensor, target_class: int | None = None):
    """Return prediction scores and gradient*input saliency with shape [clips, T, V, C]."""
    saliency_inputs = inputs.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)

    if hasattr(model, "forward_clip_logits") and saliency_inputs.ndim == 6:
        clip_logits = model.forward_clip_logits(saliency_inputs)
        logits = clip_logits.mean(dim=1)
    else:
        logits = model(saliency_inputs)

    scores = logits.softmax(dim=-1)[0]
    if target_class is None:
        target_class = int(scores.argmax().item())

    target_logit = logits[0, target_class]
    target_logit.backward()

    grad = saliency_inputs.grad.detach()
    saliency = (grad * saliency_inputs.detach()).abs()

    if saliency.ndim == 6:
        # [N, clips, M, T, V, C] -> [clips, T, V, C]
        saliency = saliency.sum(dim=(0, 2))
    elif saliency.ndim == 5:
        # [N, M, T, V, C] -> [1, T, V, C]
        saliency = saliency.sum(dim=(0, 1))[None]
    else:
        raise ValueError(f"Unexpected saliency shape: {tuple(saliency.shape)}")

    return scores.detach(), target_class, saliency.cpu().numpy()


def saliency_feature_groups(input_mode: str, num_channels: int) -> List[tuple[str, List[int]]]:
    if input_mode == "xyhandrel_bone" and num_channels >= 6:
        return [("xy", [0, 1]), ("hand_rel", [2, 3]), ("bone", [4, 5])]
    if input_mode == "xyhandrel_norm" and num_channels >= 6:
        return [("xy", [0, 1]), ("hand_rel", [2, 3]), ("hand_rel_norm", [4, 5])]
    if input_mode == "xyhandrel" and num_channels >= 4:
        return [("xy", [0, 1]), ("hand_rel", [2, 3])]
    if input_mode == "xyzscore" and num_channels >= 4:
        return [("xyz", [0, 1, 2]), ("score", [3])]
    if input_mode == "xyscore" and num_channels >= 3:
        return [("xy", [0, 1]), ("score", [2])]
    if input_mode == "xyz" and num_channels >= 3:
        return [("xyz", [0, 1, 2])]
    return [("all_features", list(range(num_channels)))]


def save_saliency_heatmap(
    saliency: np.ndarray,
    output_path: Union[str, Path],
    title: str,
    input_mode: str,
    clip_index: int = 0,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clips, clip_len, num_joints, num_channels = saliency.shape
    if not (0 <= clip_index < clips):
        raise ValueError(f"saliency_clip={clip_index} is out of range for {clips} clips.")

    clip_saliency = saliency[clip_index]
    groups = saliency_feature_groups(input_mode, num_channels)
    group_heatmaps = []
    for group_name, channels in groups:
        valid_channels = [idx for idx in channels if idx < num_channels]
        if not valid_channels:
            continue
        heatmap = clip_saliency[..., valid_channels].sum(axis=-1).T
        group_heatmaps.append((group_name, heatmap))

    if not group_heatmaps:
        raise ValueError(f"No saliency groups available for C={num_channels}.")

    max_value = max(float(heatmap.max()) for _, heatmap in group_heatmaps)
    if max_value <= 0:
        max_value = 1.0

    total_heatmap = sum(heatmap for _, heatmap in group_heatmaps)
    joint_importance = total_heatmap.mean(axis=1) if total_heatmap.size else np.zeros(num_joints)
    top_joint_indices = np.argsort(-joint_importance)[:10]

    fig, axes = plt.subplots(
        len(group_heatmaps),
        2,
        figsize=(18, max(4, 3.2 * len(group_heatmaps))),
        gridspec_kw={"width_ratios": [5, 1], "wspace": 0.06, "hspace": 0.24},
        squeeze=False,
    )

    im = None
    x = np.arange(clip_len)
    for row, (group_name, heatmap) in enumerate(group_heatmaps):
        heatmap = heatmap / max_value
        frame_importance = heatmap.mean(axis=0) if heatmap.size else np.zeros(clip_len)

        ax_heat = axes[row, 0]
        ax_bar = axes[row, 1]
        im = ax_heat.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=1)
        ax_heat.set_ylabel("Keypoint index")
        ax_heat.set_title(f"{group_name} | clip {clip_index}")
        ax_heat.plot(x, frame_importance * max(num_joints - 1, 1), color="white", linewidth=0.9, alpha=0.8)
        for boundary in (23, 44):
            if boundary < num_joints:
                ax_heat.axhline(boundary - 0.5, color="cyan", linewidth=1.0, alpha=0.85)
        if row == len(group_heatmaps) - 1:
            ax_heat.set_xlabel("Sampled frame index within selected test clip")
        else:
            ax_heat.set_xticklabels([])

        group_joint_importance = heatmap.mean(axis=1) if heatmap.size else np.zeros(num_joints)
        ax_bar.barh(np.arange(num_joints), group_joint_importance, color="steelblue")
        ax_bar.invert_yaxis()
        ax_bar.set_title("Joint mean")
        ax_bar.set_yticks(top_joint_indices)
        ax_bar.set_yticklabels([str(i) for i in top_joint_indices])
        ax_bar.grid(True, axis="x", linestyle="--", alpha=0.25)

    fig.suptitle(title, y=0.995)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012)
        cbar.set_label("Normalized abs(gradient * input), shared scale")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

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
    npz_paths = collect_npz_paths(args.npz)
    single_file_mode = len(npz_paths) == 1 and npz_paths[0].is_file()

    model = build_model(cfg).to(device)
    info = load_checkpoint(model, args.checkpoint, map_location=device, strict=False)
    if info["missing_keys"] or info["unexpected_keys"]:
        print("missing=", info["missing_keys"], "unexpected=", info["unexpected_keys"])
    model.eval()

    label_map = load_label_map(args.label_map)
    wrong_gt_labels: List[int] = []
    num_labeled = 0
    top1_correct = 0

    for npz_path in npz_paths:
        sample, inputs, scores = predict_npz(model, npz_path, cfg, device)
        gt = infer_label(npz_path, args.label_source)

        if single_file_mode:
            saliency_scores, saliency_class, saliency = compute_input_saliency(
                model,
                inputs,
                target_class=args.saliency_class,
            )
            scores = saliency_scores

        topk = format_topk(scores, args.topk, label_map)
        top1 = topk[0]
        correct = gt is not None and top1["class_id"] == gt
        if gt is not None:
            num_labeled += 1
            top1_correct += int(correct)
            if not correct:
                wrong_gt_labels.append(gt)

        if single_file_mode:
            print("NPZ:", npz_path)
            print("frames:", sample["total_frames"])
            if gt is not None:
                print("gt:", gt)
            for rank, item in enumerate(topk, start=1):
                marker = " |" if rank == 1 and correct else ""
                print(
                    f"top{rank}: class_id={item['class_id']} "
                    f"label={item['label']} score={item['score']:.6f}{marker}"
                )
        else:
            gt_text = "" if gt is None else f" gt={gt}"
            marker = " |" if correct else ""
            print(
                f"{npz_path.name}: frames={int(sample['total_frames'])} "
                f"pred={top1['class_id']} label={top1['label']} "
                f"score={top1['score']:.6f}{gt_text}{marker}"
            )

    if single_file_mode:
        saliency_out = Path(args.saliency_out) if args.saliency_out else npz_paths[0].with_suffix(".saliency.jpg")
        saliency_label = label_map.get(str(saliency_class), str(saliency_class))
        save_saliency_heatmap(
            saliency,
            saliency_out,
            title=f"Saliency for class_id={saliency_class} label={saliency_label} | {npz_paths[0].name}",
            input_mode=getattr(cfg, "INPUT_MODE", "xy"),
            clip_index=args.saliency_clip,
        )
        print("saliency_jpg:", saliency_out)

    top1 = (top1_correct / num_labeled) if num_labeled else None
    print(f"summary: samples={len(npz_paths)} labeled={num_labeled} top1={top1}")
    print("wrong_gt_labels:", wrong_gt_labels)
    print("wrong_gt_label_counts:", dict(sorted(Counter(wrong_gt_labels).items())))


if __name__ == "__main__":
    main()

