from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ANN_FILE = WORKSPACE_ROOT / "dataset/cropped_holistic_results_split/mediapipe_sign_3d_without_face_pose_score_1.pkl"
DEFAULT_CONFIG = PROJECT_ROOT / "model/configs/cnn1d_mediapipe_sign_without_face.py"

from model.builder import build_model
from model.config_utils import load_config, resolve_config_path
from model.data import MediapipeSignDataset, load_annotation, preprocess_keypoint_sample
from model.model import load_checkpoint
from model.train import resolve_device, run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standalone FairyTaleSL skeleton models.")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--ann-file",
        default=None,
        help="Annotation pkl path. Defaults to ANN_FILE in config, then script default.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to save eval logs. Defaults to the checkpoint directory.",
    )
    parser.add_argument("--sequence", action="store_true", help="Evaluate sequence annotation with sliding windows and WER.")
    parser.add_argument("--sequence-window", type=int, default=None)
    parser.add_argument(
        "--sequence-windows",
        default=None,
        help="Comma-separated multi-scale sequence windows, e.g. 25,35,50,65.",
    )
    parser.add_argument("--sequence-stride", type=int, default=None)
    parser.add_argument("--sequence-score-threshold", type=float, default=None)
    parser.add_argument(
        "--no-sequence-collapse",
        action="store_true",
        default=None,
        help="Do not remove consecutive duplicate window predictions.",
    )
    parser.add_argument(
        "--no-sequence-include-tail",
        dest="sequence_include_tail",
        action="store_false",
        help="Do not add a final tail window when stride does not cover the end.",
    )
    parser.set_defaults(sequence_include_tail=None)
    return parser.parse_args()


def append_eval_log(log_path: Path, lines: Sequence[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def normalize_label_sequence(label: Any) -> List[str]:
    if isinstance(label, np.ndarray):
        label = label.tolist()
    if isinstance(label, str):
        label = label.replace(",", " ").split()
        return [str(item) for item in label]
    if isinstance(label, (list, tuple)):
        out = []
        for item in label:
            if isinstance(item, (list, tuple, np.ndarray)):
                out.extend(normalize_label_sequence(item))
            else:
                out.append(str(item))
        return out
    return [str(label)]


def window_ranges(total_frames: int, window: int, stride: int, include_tail: bool = True) -> List[Tuple[int, int]]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")
    if window <= 0 or stride <= 0:
        raise ValueError("sequence window and stride must be positive.")
    if total_frames <= window:
        return [(0, total_frames)]

    starts = list(range(0, total_frames - window + 1, stride))
    if include_tail and starts[-1] != total_frames - window:
        starts.append(total_frames - window)
    return [(start, start + window) for start in starts]


def parse_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def sequence_window_specs(args: argparse.Namespace, cfg) -> List[Tuple[int, int]]:
    windows = parse_int_list(args.sequence_windows)
    if not windows and args.sequence_window is not None:
        windows = [args.sequence_window]
    if not windows:
        windows = parse_int_list(getattr(cfg, "SEQUENCE_WINDOWS", None))
    if not windows:
        windows = [getattr(cfg, "SEQUENCE_WINDOW", 90)]

    stride_ratio = float(getattr(cfg, "SEQUENCE_STRIDE_RATIO", 0.5))
    strides = parse_int_list(getattr(cfg, "SEQUENCE_STRIDES", None))
    if args.sequence_stride is not None:
        strides = [args.sequence_stride] * len(windows)
    elif not strides:
        default_stride = getattr(cfg, "SEQUENCE_STRIDE", None)
        if len(windows) == 1 and default_stride is not None:
            strides = [int(default_stride)]
        else:
            strides = [max(1, int(round(window * stride_ratio))) for window in windows]
    elif len(strides) == 1 and len(windows) > 1:
        strides = strides * len(windows)

    if len(strides) != len(windows):
        raise ValueError(
            f"sequence windows and strides must have the same length, "
            f"got windows={windows}, strides={strides}."
        )
    return [(int(window), int(stride)) for window, stride in zip(windows, strides)]


def windows_overlap(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return max(left["start"], right["start"]) < min(left["end"], right["end"])


def merge_score_peaks(candidates: List[Dict[str, Any]], collapse: bool) -> List[Dict[str, Any]]:
    selected = []
    #score 높은 순서로 정렬
    for candidate in sorted(candidates, key=lambda item: (-item["score"], item["start"], item["end"])):
        #예측 label이 같고 시간 구간이 겹치는 후보 있다면 패스 (score peak merge)
        overlaps_same_label = any(
            candidate["pred"] == kept["pred"] and windows_overlap(candidate, kept)
            for kept in selected
        )
        if not overlaps_same_label:
            selected.append(candidate)
    #선택된 후보들을 다시 시간 순서대로 정렬합니다.
    selected = sorted(selected, key=lambda item: (item["start"], item["end"], -item["score"]))
    if not collapse:
        return selected

    collapsed = []
    for candidate in selected:
        if collapsed and collapsed[-1]["pred"] == candidate["pred"]:
            if candidate["score"] > collapsed[-1]["score"]:
                collapsed[-1] = candidate
            continue
        collapsed.append(candidate)
    return collapsed


def slice_sample_window(sample: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    window_sample = dict(sample)
    window_sample["keypoint"] = sample["keypoint"][:, start:end]
    if "keypoint_score" in sample:
        window_sample["keypoint_score"] = sample["keypoint_score"][:, start:end]
    window_sample["total_frames"] = end - start
    return window_sample


def wer_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> Dict[str, float]:
    n = len(reference)
    m = len(hypothesis)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost, s, d, ins = dp[i - 1][0]
        dp[i][0] = (cost + 1, s, d + 1, ins)
    for j in range(1, m + 1):
        cost, s, d, ins = dp[0][j - 1]
        dp[0][j] = (cost + 1, s, d, ins + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = []
            cost, s, d, ins = dp[i - 1][j - 1]
            if reference[i - 1] == hypothesis[j - 1]:
                candidates.append((cost, s, d, ins))
            else:
                candidates.append((cost + 1, s + 1, d, ins))

            cost, s, d, ins = dp[i - 1][j]
            candidates.append((cost + 1, s, d + 1, ins))

            cost, s, d, ins = dp[i][j - 1]
            candidates.append((cost + 1, s, d, ins + 1))
            dp[i][j] = min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))

    cost, s, d, ins = dp[n][m]
    wer = ((s + d + ins) / n * 100.0) if n > 0 else (0.0 if m == 0 else 100.0)
    return {"S": s, "D": d, "I": ins, "N": n, "WER": wer, "edit_distance": cost}


def predict_window(
    model: torch.nn.Module,
    sample: Dict[str, Any],
    cfg,
    device: torch.device,
) -> Tuple[str, float]:
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
        scores = model.predict(inputs)[0]
    score, class_id = torch.max(scores, dim=0)
    return str(int(class_id.item())), float(score.item())


def update_wer_totals(totals: Dict[str, float], counts: Dict[str, float]) -> None:
    totals["num_samples"] += 1
    totals["S"] += int(counts["S"])
    totals["D"] += int(counts["D"])
    totals["I"] += int(counts["I"])
    totals["N"] += int(counts["N"])


def finalize_wer_totals(totals: Dict[str, float]) -> None:
    totals["WER"] = (
        (totals["S"] + totals["D"] + totals["I"]) / totals["N"] * 100.0
        if totals["N"] > 0
        else 0.0
    )


def run_sequence_eval(model: torch.nn.Module, ann_file: str, split: str, cfg, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    ann = load_annotation(ann_file)
    annotations = {item["frame_dir"]: item for item in ann["annotations"]}
    split_names = ann["split"][split] #split
    samples = [annotations[name] for name in split_names]

    model.eval()
    per_sample = []
    wer_by_label_len = {}
    total_counts = {"num_samples": 0, "S": 0, "D": 0, "I": 0, "N": 0, "WER": 0.0}
    collapse = not args.no_sequence_collapse
    skipped_label_len_3_samples = 0

    for sample in samples:
        reference = normalize_label_sequence(sample["label"]) #ex) ['18', '37']
        label_len = len(reference)
        include_in_wer = True
        if not include_in_wer:
            skipped_label_len_3_samples += 1

        total_frames = int(sample.get("total_frames", sample["keypoint"].shape[1]))
        window_predictions = []
        candidate_predictions = []
        for window, stride in args.sequence_window_specs: #multl-scale window
            ranges = window_ranges(
                total_frames,
                window,
                stride,
                include_tail=args.sequence_include_tail,
            )
            for start, end in ranges:
                pred, score = predict_window(model, slice_sample_window(sample, start, end), cfg, device)
                window_prediction = {
                    "start": start,
                    "end": end,
                    "window": window,
                    "stride": stride,
                    "pred": pred,
                    "score": score,
                    "kept": False,
                }
                window_predictions.append(window_prediction)
                if score >= args.sequence_score_threshold:
                    candidate_predictions.append(window_prediction)

        selected_predictions = merge_score_peaks(candidate_predictions, collapse=collapse)
        selected_ids = {id(item) for item in selected_predictions}
        for window_prediction in window_predictions:
            window_prediction["kept"] = id(window_prediction) in selected_ids
        pred_sequence = [item["pred"] for item in selected_predictions]

        counts = wer_counts(reference, pred_sequence)
        result = {
            "frame_dir": sample.get("frame_dir"),
            "total_frames": total_frames,
            "reference": reference,
            "label_len": label_len,
            "prediction": pred_sequence,
            "wer": counts,
            "windows": window_predictions,
            "excluded_from_wer": not include_in_wer,
        }
        if not include_in_wer:
            continue

        label_len_stats = wer_by_label_len.setdefault(
            str(label_len),
            {"num_samples": 0, "S": 0, "D": 0, "I": 0, "N": 0, "WER": 0.0},
        )
        update_wer_totals(label_len_stats, counts)
        update_wer_totals(total_counts, counts)
        per_sample.append(result)

    for stats in wer_by_label_len.values():
        finalize_wer_totals(stats)
    finalize_wer_totals(total_counts)

    return {
        "split": split,
        "num_samples": len(samples),
        "num_eval_samples": len(per_sample),
        "skipped_label_len_3_samples": skipped_label_len_3_samples,
        "window_specs": [{"window": window, "stride": stride} for window, stride in args.sequence_window_specs],
        "score_threshold": args.sequence_score_threshold,
        "collapse_repeats": collapse,
        "include_tail": args.sequence_include_tail,
        "S": total_counts["S"],
        "D": total_counts["D"],
        "I": total_counts["I"],
        "N": total_counts["N"],
        "WER": total_counts["WER"],
        "wer_by_label_len": dict(sorted(wer_by_label_len.items(), key=lambda item: int(item[0]))),
        "samples": per_sample,
    }


def save_regular_eval(args, cfg, checkpoint_path: Path, log_dir: Path, device: torch.device, info: Dict[str, Any], metrics: Dict[str, float], dataset) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_line = (
        f"split={args.split} loss={metrics['loss']:.4f} "
        f"top1={metrics['top1']:.4f} top5={metrics['top5']:.4f}"
    )
    lines = [
        f"{timestamp} | checkpoint={checkpoint_path}",
        f"{timestamp} | config={Path(args.config).expanduser().resolve()}",
        f"{timestamp} | ann_file={Path(args.ann_file).expanduser().resolve()}",
        f"{timestamp} | device={device} batch_size={args.batch_size} num_workers={args.num_workers}",
        f"{timestamp} | samples={len(dataset)} clip_len={cfg.CLIP_LEN} num_clips={cfg.TEST_NUM_CLIPS}",
    ]
    if info["missing_keys"] or info["unexpected_keys"]:
        lines.append(f"{timestamp} | missing={info['missing_keys']} unexpected={info['unexpected_keys']}")
    lines.append(f"{timestamp} | {metric_line}")

    append_eval_log(log_dir / f"{args.split}.log", lines)
    with open(log_dir / f"{args.split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "split": args.split,
                "checkpoint": str(checkpoint_path),
                "config": str(Path(args.config).expanduser().resolve()),
                "ann_file": str(Path(args.ann_file).expanduser().resolve()),
                "device": str(device),
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "num_samples": len(dataset),
                "clip_len": cfg.CLIP_LEN,
                "num_clips": cfg.TEST_NUM_CLIPS,
                "missing_keys": info["missing_keys"],
                "unexpected_keys": info["unexpected_keys"],
                "metrics": metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(metric_line)
    print(f"Saved eval log: {log_dir / f'{args.split}.log'}")
    print(f"Saved eval metrics: {log_dir / f'{args.split}_metrics.json'}")


def save_sequence_predictions_by_label_len(log_dir: Path, split: str, metrics: Dict[str, Any]) -> List[Path]:
    samples_by_label_len = {}
    for sample in metrics["samples"]:
        label_len = str(sample["label_len"])
        included_windows = [
            {
                "start": window["start"],
                "end": window["end"],
                "window": window["window"],
                "stride": window["stride"],
                "pred": window["pred"],
                "score": window["score"],
            }
            for window in sample["windows"]
            if window["kept"]
        ]
        samples_by_label_len.setdefault(label_len, []).append(
            {
                "frame_dir": sample["frame_dir"],
                "label_len": sample["label_len"],
                "total_frames": sample["total_frames"],
                "reference": sample["reference"],
                "prediction": sample["prediction"],
                "included_windows": included_windows,
            }
        )

    saved_paths = []
    for label_len, samples in sorted(samples_by_label_len.items(), key=lambda item: int(item[0])):
        path = log_dir / f"{split}_sequence_predictions_label_len_{label_len}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
            f.write("\n")
        saved_paths.append(path)
    return saved_paths


def save_sequence_eval(args, checkpoint_path: Path, log_dir: Path, device: torch.device, info: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_line = (
        f"sequence split={args.split} WER={metrics['WER']:.2f}% "
        f"S={metrics['S']} D={metrics['D']} I={metrics['I']} N={metrics['N']} "
        f"eval_samples={metrics['num_eval_samples']} "
        f"skipped_label_len_3_samples={metrics['skipped_label_len_3_samples']}"
    )
    lines = [
        f"{timestamp} | checkpoint={checkpoint_path}",
        f"{timestamp} | config={Path(args.config).expanduser().resolve()}",
        f"{timestamp} | ann_file={Path(args.ann_file).expanduser().resolve()}",
        f"{timestamp} | device={device}",
        f"{timestamp} | window_specs={metrics['window_specs']} score_threshold={metrics['score_threshold']} collapse_repeats={metrics['collapse_repeats']} include_tail={metrics['include_tail']}",
    ]
    label_len_lines = []
    for label_len, stats in metrics["wer_by_label_len"].items():
        label_len_lines.append(
            f"label_len={label_len} WER={stats['WER']:.2f}% "
            f"S={stats['S']} D={stats['D']} I={stats['I']} N={stats['N']} "
            f"samples={stats['num_samples']}"
        )
    if info["missing_keys"] or info["unexpected_keys"]:
        lines.append(f"{timestamp} | missing={info['missing_keys']} unexpected={info['unexpected_keys']}")
    lines.append(f"{timestamp} | {metric_line}")
    lines.extend(f"{timestamp} | {line}" for line in label_len_lines)

    log_name = f"{args.split}_sequence"
    prediction_paths = save_sequence_predictions_by_label_len(log_dir, args.split, metrics)
    lines.extend(f"{timestamp} | saved sequence predictions: {path}" for path in prediction_paths)
    append_eval_log(log_dir / f"{log_name}.log", lines)
    with open(log_dir / f"{log_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "checkpoint": str(checkpoint_path),
                "config": str(Path(args.config).expanduser().resolve()),
                "ann_file": str(Path(args.ann_file).expanduser().resolve()),
                "device": str(device),
                "missing_keys": info["missing_keys"],
                "unexpected_keys": info["unexpected_keys"],
                "metrics": metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(metric_line)
    for line in label_len_lines:
        print(line)
    for path in prediction_paths:
        print(f"Saved sequence predictions: {path}")
    print(f"Saved sequence eval log: {log_dir / f'{log_name}.log'}")
    print(f"Saved sequence eval metrics: {log_dir / f'{log_name}_metrics.json'}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    args.batch_size = args.batch_size if args.batch_size is not None else cfg.BATCH_SIZE
    args.ann_file = resolve_config_path(cfg, args.ann_file or getattr(cfg, "ANN_FILE", str(DEFAULT_ANN_FILE)))
    args.sequence_window_specs = sequence_window_specs(args, cfg)
    args.sequence_window = args.sequence_window_specs[0][0]
    args.sequence_stride = args.sequence_window_specs[0][1]
    args.sequence_score_threshold = (
        args.sequence_score_threshold
        if args.sequence_score_threshold is not None
        else getattr(cfg, "SEQUENCE_SCORE_THRESHOLD", 0.0)
    )
    if args.no_sequence_collapse is None:
        args.no_sequence_collapse = not getattr(cfg, "SEQUENCE_COLLAPSE_REPEATS", True)
    if args.sequence_include_tail is None:
        args.sequence_include_tail = getattr(cfg, "SEQUENCE_INCLUDE_TAIL", True)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else checkpoint_path.parent

    model = build_model(cfg).to(device)
    info = load_checkpoint(model, checkpoint_path, map_location=device, strict=False)

    if args.sequence:
        metrics = run_sequence_eval(model, args.ann_file, args.split, cfg, args, device)
        save_sequence_eval(args, checkpoint_path, log_dir, device, info, metrics)
        return

    dataset = MediapipeSignDataset(
        args.ann_file,
        split=args.split,
        clip_len=cfg.CLIP_LEN,
        num_clips=cfg.TEST_NUM_CLIPS,
        test_mode=True,
        zero_pad_short=getattr(cfg, "ZERO_PAD_SHORT", False),
        input_mode=getattr(cfg, "INPUT_MODE", "xy"),
        keypoint_normalize=getattr(cfg, "KEYPOINT_NORMALIZE", None),
        random_horizontal_flip=getattr(cfg, "RANDOM_HORIZONTAL_FLIP", None),
        short_sample_interpolation=getattr(cfg, "SHORT_SAMPLE_INTERPOLATION", None),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = run_eval(model, loader, device)
    save_regular_eval(args, cfg, checkpoint_path, log_dir, device, info, metrics, dataset)


if __name__ == "__main__":
    main()
