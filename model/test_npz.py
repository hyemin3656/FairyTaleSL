from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Union

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
    parser = argparse.ArgumentParser(description="Predict one MediaPipe npz with standalone CNN1D.")
    parser.add_argument("npz")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=5)
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
    return parser.parse_args()



def compute_input_saliency(model, inputs: torch.Tensor, target_class: int | None = None):
    """Return prediction scores and gradient*input saliency with shape [clips, T, V]."""
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
        # [N, clips, M, T, V, C] -> [clips, T, V]
        saliency = saliency.sum(dim=(0, 2, 5))
    elif saliency.ndim == 5:
        # [N, M, T, V, C] -> [1, T, V]
        saliency = saliency.sum(dim=(0, 1, 4))[None]
    else:
        raise ValueError(f"Unexpected saliency shape: {tuple(saliency.shape)}")

    return scores.detach(), target_class, saliency.cpu().numpy()


def save_saliency_heatmap(saliency: np.ndarray, output_path: Union[str, Path], title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clips, clip_len, num_joints = saliency.shape
    heatmap = saliency.reshape(clips * clip_len, num_joints).T
    max_value = float(heatmap.max()) if heatmap.size else 0.0
    if max_value > 0:
        heatmap = heatmap / max_value

    frame_importance = heatmap.mean(axis=0) if heatmap.size else np.zeros(clips * clip_len)
    joint_importance = heatmap.mean(axis=1) if heatmap.size else np.zeros(num_joints)
    top_joint_indices = np.argsort(-joint_importance)[:10]

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 5], width_ratios=[5, 1], hspace=0.08, wspace=0.06)
    ax_frame = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_joint = fig.add_subplot(gs[1, 1])

    x = np.arange(clips * clip_len)
    ax_frame.plot(x, frame_importance, color="black", linewidth=1.2)
    ax_frame.set_xlim(0, max(clips * clip_len - 1, 1))
    ax_frame.set_ylabel("Frame\nmean")
    ax_frame.set_xticks([])
    ax_frame.grid(True, linestyle="--", alpha=0.25)

    im = ax_heat.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=1)
    ax_heat.set_xlabel("Sampled frame index (clips flattened)")
    ax_heat.set_ylabel("Keypoint index")
    ax_heat.set_title(title)

    for boundary in (23, 44):
        if boundary < num_joints:
            ax_heat.axhline(boundary - 0.5, color="cyan", linewidth=1.0, alpha=0.85)
    for clip_idx in range(1, clips):
        ax_heat.axvline(clip_idx * clip_len - 0.5, color="white", linewidth=0.8, alpha=0.55)

    ax_joint.barh(np.arange(num_joints), joint_importance, color="steelblue")
    ax_joint.invert_yaxis()
    ax_joint.set_title("Joint mean")
    ax_joint.set_xlabel("Importance")
    ax_joint.set_yticks(top_joint_indices)
    ax_joint.set_yticklabels([str(i) for i in top_joint_indices])
    ax_joint.grid(True, axis="x", linestyle="--", alpha=0.25)

    cbar = fig.colorbar(im, ax=[ax_frame, ax_heat, ax_joint], fraction=0.025, pad=0.015)
    cbar.set_label("Normalized abs(gradient * input)")

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

    saliency_scores, saliency_class, saliency = compute_input_saliency(
        model,
        inputs,
        target_class=args.saliency_class,
    )
    scores = saliency_scores
    saliency_out = Path(args.saliency_out) if args.saliency_out else Path(args.npz).with_suffix(".saliency.jpg")

    values, indices = scores.topk(min(args.topk, scores.numel()))
    print("NPZ:", Path(args.npz))
    print("frames:", sample["total_frames"])
    for rank, (class_id, score) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        label = label_map.get(str(class_id), str(class_id))
        print(f"top{rank}: class_id={class_id} label={label} score={score:.6f}")

    saliency_label = label_map.get(str(saliency_class), str(saliency_class))
    save_saliency_heatmap(
        saliency,
        saliency_out,
        title=f"Saliency for class_id={saliency_class} label={saliency_label} | {Path(args.npz).name}",
    )
    print("saliency_jpg:", saliency_out)


if __name__ == "__main__":
    main()

