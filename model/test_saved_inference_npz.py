from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_NPZ_DIR = Path("/home/ubuntu/saved_inference_npz")
DEFAULT_CLASS_TEST_DIR = Path("/home/ubuntu/dataset/cropped_holistic_results_interpolated_remapped_direct/test_selected_from_attachment")
DEFAULT_CONFIG = PROJECT_ROOT / "model/configs/cnn1d_mediapipe_sign_without_face.py"
DEFAULT_LABEL_MAP = PROJECT_ROOT / "src/class_labels.json"
DEFAULT_LOG_FILE = PROJECT_ROOT / "work_dirs/test_saved_inference_npz.log"

from model.builder import build_model
from model.config_utils import load_config
from model.data import build_npz_sample, preprocess_keypoint_sample
from model.model import load_checkpoint
from model.train import resolve_device


class TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_output_logging(log_file: str | Path):
    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)
    return log_handle, original_stdout, original_stderr, path


def restore_output_logging(log_handle, original_stdout, original_stderr) -> None:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test standalone FairyTaleSL model on saved npz files with pre-concat keypoint arrays."
    )
    parser.add_argument("checkpoint", help="Path to standalone checkpoint, e.g. work_dirs/.../best.pth")
    parser.add_argument("--npz-dir", default=str(DEFAULT_NPZ_DIR))
    parser.add_argument("--class-test-dir", default=str(DEFAULT_CLASS_TEST_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE))
    parser.add_argument(
        "--label-source",
        default="auto",
        choices=["auto", "npz", "stem", "none"],
        help="Where to read ground-truth labels for accuracy. auto tries npz['label'], then integer filename stem.",
    )
    return parser.parse_args()


def load_label_map(path: str | Path) -> Dict[str, str]:
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def infer_label(npz_path: Path, data: np.lib.npyio.NpzFile, label_source: str) -> Optional[int]:
    if label_source in {"auto", "npz"} and "label" in data:
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


def infer_split_suffix_label(npz_path: Path) -> Optional[int]:
    parts = npz_path.stem.split("_")
    if not parts:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def select_one_npz_per_class(npz_paths: Sequence[Path]) -> List[Path]:
    by_label: Dict[int, Path] = {}
    for npz_path in sorted(npz_paths):
        label = infer_split_suffix_label(npz_path)
        if label is None:
            continue
        by_label.setdefault(label, npz_path)
    return [by_label[label] for label in sorted(by_label)]


def ensure_m_t_v_c(keypoint: np.ndarray, npz_path: Path) -> np.ndarray:
    keypoint = np.asarray(keypoint, dtype=np.float32)
    if keypoint.ndim == 3:
        keypoint = keypoint[None]
    elif keypoint.ndim != 4:
        raise ValueError(
            f"{npz_path} keypoint must have shape [T,V,C] or [M,T,V,C], got {keypoint.shape}."
        )

    if keypoint.shape[2] != 65:
        raise ValueError(f"{npz_path} expected 65 joints, got keypoint shape {keypoint.shape}.")
    if keypoint.shape[-1] < 2:
        raise ValueError(f"{npz_path} keypoint must have at least x/y channels, got {keypoint.shape}.")
    if np.isnan(keypoint).any():
        raise ValueError(f"{npz_path} keypoint contains NaN.")
    return keypoint


def load_saved_npz_sample(npz_path: str | Path, label_source: str = "auto") -> Dict[str, Any]:
    npz_path = Path(npz_path).expanduser()
    data = np.load(npz_path, allow_pickle=True)

    if "keypoint" not in data:
        sample = build_npz_sample(npz_path)
        sample.pop("keypoint_score", None)
        label = infer_label(npz_path, data, label_source)
        if label is not None:
            sample["label"] = label
        return sample

    keypoint = ensure_m_t_v_c(data["keypoint"], npz_path)
    sample: Dict[str, Any] = {
        "frame_dir": npz_path.stem,
        "label": -1,
        "total_frames": keypoint.shape[1],
        "keypoint": keypoint,
    }


    label = infer_label(npz_path, data, label_source)
    if label is not None:
        sample["label"] = label
    return sample


def load_split_suffix_label_sample(npz_path: str | Path) -> Dict[str, Any]:
    sample = load_saved_npz_sample(npz_path, label_source="none")
    label = infer_split_suffix_label(Path(npz_path).expanduser())
    if label is not None:
        sample["label"] = label
    return sample


def predict_sample(model: torch.nn.Module, sample: Dict[str, Any], cfg, device: torch.device) -> torch.Tensor:
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
        return model.predict(inputs)[0].detach().cpu()


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


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_csv(path: str | Path, results: Sequence[Dict[str, Any]]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "npz", "frames", "gt", "pred", "pred_label", "score", "correct"],
        )
        writer.writeheader()
        for item in results:
            top1 = item["topk"][0]
            gt = item.get("gt")
            writer.writerow(
                {
                    "dataset": item.get("dataset", ""),
                    "npz": item["npz"],
                    "frames": item["frames"],
                    "gt": "" if gt is None else gt,
                    "pred": top1["class_id"],
                    "pred_label": top1["label"],
                    "score": f"{top1['score']:.8f}",
                    "correct": "" if gt is None else int(gt == top1["class_id"]),
                }
            )


def empty_metrics(num_samples: int = 0) -> Dict[str, Any]:
    return {
        "num_samples": num_samples,
        "num_labeled": 0,
        "top1": None,
        "top5": None,
    }


def evaluate_npz_paths(
    *,
    name: str,
    npz_paths: Sequence[Path],
    model: torch.nn.Module,
    cfg,
    device: torch.device,
    label_map: Dict[str, str],
    topk_value: int,
    loader,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    num_labeled = 0
    top1_correct = 0
    top5_correct = 0

    print(f"[{name}] evaluating {len(npz_paths)} samples")
    for npz_path in npz_paths:
        sample = loader(npz_path)
        scores = predict_sample(model, sample, cfg, device)
        topk = format_topk(scores, topk_value, label_map)
        gt = None if int(sample["label"]) < 0 else int(sample["label"])

        if gt is not None:
            num_labeled += 1
            pred_ids = [item["class_id"] for item in topk]
            top1_correct += int(pred_ids[0] == gt)
            top5_correct += int(gt in pred_ids[: min(5, len(pred_ids))])

        top1 = topk[0]
        gt_text = "" if gt is None else f" gt={gt}"
        correct_marker = " |" if gt is not None and top1["class_id"] == gt else ""
        print(
            f"[{name}] {npz_path.name}: frames={int(sample['total_frames'])} "
            f"pred={top1['class_id']} label={top1['label']} "
            f"score={top1['score']:.6f}{gt_text}{correct_marker}"
        )
        results.append(
            {
                "dataset": name,
                "npz": str(npz_path),
                "frames": int(sample["total_frames"]),
                "gt": gt,
                "topk": topk,
            }
        )

    metrics = {
        "num_samples": len(results),
        "num_labeled": num_labeled,
        "top1": (top1_correct / num_labeled) if num_labeled else None,
        "top5": (top5_correct / num_labeled) if num_labeled else None,
    }
    print(
        f"[{name}] summary: samples={metrics['num_samples']} labeled={num_labeled} "
        f"top1={metrics['top1']} top5={metrics['top5']}"
    )
    return results, metrics


def combine_metrics(metrics_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_samples = sum(int(metrics["num_samples"]) for metrics in metrics_list)
    total_labeled = sum(int(metrics["num_labeled"]) for metrics in metrics_list)
    if total_labeled == 0:
        return empty_metrics(total_samples)

    top1_correct = 0.0
    top5_correct = 0.0
    for metrics in metrics_list:
        labeled = int(metrics["num_labeled"])
        if labeled == 0:
            continue
        top1_correct += float(metrics["top1"]) * labeled
        top5_correct += float(metrics["top5"]) * labeled

    return {
        "num_samples": total_samples,
        "num_labeled": total_labeled,
        "top1": top1_correct / total_labeled,
        "top5": top5_correct / total_labeled,
    }


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    label_map = load_label_map(args.label_map)

    saved_npz_paths = sorted(Path(args.npz_dir).expanduser().glob("*.npz"))
    if not saved_npz_paths:
        raise FileNotFoundError(f"No npz files found in {args.npz_dir}")

    class_test_dir = Path(args.class_test_dir).expanduser()
    class_test_paths = sorted(class_test_dir.glob("*.npz"))
    if not class_test_paths:
        raise FileNotFoundError(f"No npz files found in {args.class_test_dir}")

    model = build_model(cfg).to(device)
    info = load_checkpoint(model, args.checkpoint, map_location=device, strict=False)
    if info["missing_keys"] or info["unexpected_keys"]:
        print("missing=", info["missing_keys"], "unexpected=", info["unexpected_keys"])
    model.eval()

    saved_results, saved_metrics = evaluate_npz_paths(
        name="saved_inference_npz",
        npz_paths=saved_npz_paths,
        model=model,
        cfg=cfg,
        device=device,
        label_map=label_map,
        topk_value=args.topk,
        loader=lambda path: load_saved_npz_sample(path, label_source=args.label_source),
    )
    class_test_results, class_test_metrics = evaluate_npz_paths(
        name="class_test_one_per_class",
        npz_paths=class_test_paths,
        model=model,
        cfg=cfg,
        device=device,
        label_map=label_map,
        topk_value=args.topk,
        loader=load_split_suffix_label_sample,
    )
    results = saved_results + class_test_results
    metrics = combine_metrics([saved_metrics, class_test_metrics])
    print(
        f"[combined] summary: samples={metrics['num_samples']} labeled={metrics['num_labeled']} "
        f"top1={metrics['top1']} top5={metrics['top5']}"
    )

    payload = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()),
        "npz_dir": str(Path(args.npz_dir).expanduser().resolve()),
        "class_test_dir": str(class_test_dir.resolve()),
        "class_test_selection": "all npz files in class_test_dir, labels parsed from the final '_' separated filename token",
        "metrics": metrics,
        "metrics_by_dataset": {
            "saved_inference_npz": saved_metrics,
            "class_test_one_per_class": class_test_metrics,
        },
        "results": results,
    }
    if args.output_json:
        save_json(args.output_json, payload)
        print(f"saved json: {args.output_json}")
    if args.output_csv:
        save_csv(args.output_csv, results)
        print(f"saved csv: {args.output_csv}")


def main() -> None:
    args = parse_args()
    log_handle, original_stdout, original_stderr, log_path = setup_output_logging(args.log_file)
    try:
        print(f"logging output to: {log_path}")
        run(args)
    finally:
        print(f"saved log: {log_path}")
        restore_output_logging(log_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
