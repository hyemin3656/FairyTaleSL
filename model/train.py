from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ANN_FILE = WORKSPACE_ROOT / "dataset/cropped_holistic_results_interpolated_split/mediapipe_sign_3d_without_face_pose_score_1.pkl"
DEFAULT_CONFIG = PROJECT_ROOT / "model/configs/cnn1d_mediapipe_sign_without_face.py"
DEFAULT_WORK_ROOT = PROJECT_ROOT / "work_dirs"
DEFAULT_SAVED_VAL_NPZ_DIR = Path("/home/ubuntu/saved_inference_npz")
DEFAULT_SAVED_VAL_CLASS_TEST_DIR = Path("/home/ubuntu/dataset/cropped_holistic_results_interpolated_remapped_direct/test_selected_from_attachment")

from model.builder import build_model
from model.config_utils import config_to_dict, load_config, resolve_config_path
from model.model import load_checkpoint
from model.data import MediapipeSignDataset, build_npz_sample, preprocess_keypoint_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone FairyTaleSL skeleton models.")
    parser.add_argument(
        "--ann-file",
        default=None,
        help="Annotation pkl path. Defaults to ANN_FILE in config, then script default.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--resume", default=None, help="Path to standalone or MMAction2 checkpoint.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--val-mode",
        default="saved-npz",
        choices=["saved-npz", "ann"],
        help="Validation source. saved-npz matches model/test_saved_inference_npz.py behavior.",
    )
    parser.add_argument("--saved-val-npz-dir", default=str(DEFAULT_SAVED_VAL_NPZ_DIR))
    parser.add_argument("--saved-val-class-test-dir", default=str(DEFAULT_SAVED_VAL_CLASS_TEST_DIR))
    parser.add_argument(
        "--saved-val-label-source",
        default="auto",
        choices=["auto", "npz", "stem", "none"],
        help="Where to read labels for saved-NPZ validation.",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None, help="Run folder name. Defaults to current time.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="mediapipe-sign-3d")
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> dict:
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    out = {}
    for k in topk:
        k = min(k, logits.shape[1])
        out[f"top{k}"] = correct[:k].reshape(-1).float().sum().item() / labels.numel()
    return out


def run_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if inputs.ndim == 6:
                clip_logits = model.forward_clip_logits(inputs)
                logits = clip_logits.mean(dim=1)
                scores = clip_logits.softmax(dim=-1).mean(dim=1)
            else:
                logits = model(inputs)
                scores = logits
            loss = criterion(logits, labels)
            acc = accuracy(scores, labels)
            batch = labels.numel()
            total += batch
            loss_sum += loss.item() * batch
            top1_sum += acc["top1"] * batch
            top5_sum += acc["top5"] * batch
    return {"loss": loss_sum / total, "top1": top1_sum / total, "top5": top5_sum / total}


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


def infer_saved_npz_label(npz_path: Path, data: np.lib.npyio.NpzFile, label_source: str) -> Optional[int]:
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
        label = infer_saved_npz_label(npz_path, data, label_source)
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


    label = infer_saved_npz_label(npz_path, data, label_source)
    if label is not None:
        sample["label"] = label
    return sample


def load_saved_val_samples(npz_dir: str | Path, label_source: str) -> List[Dict[str, Any]]:
    npz_paths = sorted(Path(npz_dir).expanduser().glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No saved validation npz files found in {npz_dir}")

    samples = [load_saved_npz_sample(path, label_source=label_source) for path in npz_paths]
    labeled = sum(1 for sample in samples if int(sample.get("label", -1)) >= 0)
    if labeled == 0:
        raise ValueError(
            f"Saved validation npz files in {npz_dir} have no labels. "
            "Use --saved-val-label-source npz or stem with labeled files."
        )
    return samples


def load_split_suffix_label_sample(npz_path: str | Path) -> Dict[str, Any]:
    sample = load_saved_npz_sample(npz_path, label_source="none")
    label = infer_split_suffix_label(Path(npz_path).expanduser())
    if label is not None:
        sample["label"] = label
    return sample


def load_class_test_val_samples(npz_dir: str | Path) -> List[Dict[str, Any]]:
    npz_paths = sorted(Path(npz_dir).expanduser().glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No class-test npz files found in {npz_dir}")
    return [load_split_suffix_label_sample(path) for path in npz_paths]


def predict_scores_and_logits(
    model: torch.nn.Module,
    sample: Dict[str, Any],
    cfg,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    scores = model.predict(inputs)
    logits = model(inputs)
    return scores, logits


def run_saved_npz_eval_group(
    model: torch.nn.Module,
    samples: Sequence[Dict[str, Any]],
    cfg,
    device: torch.device,
) -> dict:
    model.eval()
    total = 0
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for sample in samples:
            label = int(sample.get("label", -1))
            if label < 0:
                continue

            scores, logits = predict_scores_and_logits(model, sample, cfg, device)
            labels = torch.tensor([label], dtype=torch.long, device=device)
            loss = criterion(logits, labels)
            pred_ids = scores[0].topk(min(5, scores.shape[1])).indices.tolist()
            total += 1
            loss_sum += loss.item()
            top1_correct += int(pred_ids[0] == label)
            top5_correct += int(label in pred_ids)

    if total == 0:
        raise ValueError("Saved validation samples have no usable labels.")
    return {
        "loss": loss_sum / total,
        "top1": top1_correct / total,
        "top5": top5_correct / total,
        "num_labeled": total,
        "num_samples": len(samples),
    }


def combine_saved_npz_metrics(metrics_list: Sequence[Dict[str, Any]]) -> dict:
    usable_metrics = [metrics for metrics in metrics_list if int(metrics["num_labeled"]) > 0]
    if not usable_metrics:
        raise ValueError("Saved validation metrics have no labeled samples.")

    total_labeled = sum(int(metrics["num_labeled"]) for metrics in usable_metrics)
    total_samples = sum(int(metrics.get("num_samples", metrics["num_labeled"])) for metrics in usable_metrics)
    num_groups = len(usable_metrics)

    return {
        "loss": sum(float(metrics["loss"]) for metrics in usable_metrics) / num_groups,
        "top1": sum(float(metrics["top1"]) for metrics in usable_metrics) / num_groups,
        "top5": sum(float(metrics["top5"]) for metrics in usable_metrics) / num_groups,
        "num_labeled": total_labeled,
        "num_samples": total_samples,
        "combine_method": "unweighted_dataset_mean",
    }


def run_saved_npz_eval(
    model: torch.nn.Module,
    saved_samples: Sequence[Dict[str, Any]],
    class_test_samples: Sequence[Dict[str, Any]],
    cfg,
    device: torch.device,
) -> dict:
    saved_metrics = run_saved_npz_eval_group(model, saved_samples, cfg, device)
    class_test_metrics = run_saved_npz_eval_group(model, class_test_samples, cfg, device)
    combined_metrics = combine_saved_npz_metrics([saved_metrics, class_test_metrics])
    combined_metrics["metrics_by_dataset"] = {
        "saved_inference_npz": saved_metrics,
        "class_test_one_per_class": class_test_metrics,
    }
    return combined_metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_top1: float,
    best_epoch: Optional[int],
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_top1": best_top1,
            "best_epoch": best_epoch,
            "config": config,
        },
        path,
    )


def save_run_config(cfg, cfg_dict: dict, args: argparse.Namespace, work_dir: Path) -> None:
    source = Path(getattr(cfg, "CONFIG_PATH", args.config))
    if source.exists():
        shutil.copy2(source, work_dir / source.name)

    saved_config = dict(cfg_dict)
    saved_config.update(
        {
            "EPOCHS": args.epochs,
            "BATCH_SIZE": args.batch_size,
            "LR": args.lr,
            "WEIGHT_DECAY": args.weight_decay,
            "ANN_FILE": args.ann_file,
            "VAL_MODE": args.val_mode,
            "SAVED_VAL_NPZ_DIR": args.saved_val_npz_dir,
            "SAVED_VAL_CLASS_TEST_DIR": args.saved_val_class_test_dir,
            "SAVED_VAL_LABEL_SOURCE": args.saved_val_label_source,
            "SOURCE_CONFIG": str(source),
        }
    )
    with open(work_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(saved_config, f, indent=2, ensure_ascii=False)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def init_wandb(args: argparse.Namespace, run_dir: Path, run_name: str):
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging is enabled, but wandb is not installed. "
            "Install requirements.txt or pass --no-wandb."
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        dir=str(run_dir),
        config={
            "ann_file": args.ann_file,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "val_mode": args.val_mode,
            "saved_val_npz_dir": args.saved_val_npz_dir,
            "saved_val_class_test_dir": args.saved_val_class_test_dir,
            "config_file": args.config,
        },
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/loss", step_metric="epoch")
    wandb.define_metric("train/top1_acc", step_metric="epoch")
    wandb.define_metric("val/loss", step_metric="epoch")
    wandb.define_metric("val/top1_acc", step_metric="epoch")
    wandb.define_metric("val/top5_acc", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    return run


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg_dict = config_to_dict(cfg)
    model_type = getattr(cfg, "MODEL_TYPE", "cnn1d")
    args.epochs = args.epochs if args.epochs is not None else cfg.EPOCHS
    args.batch_size = args.batch_size if args.batch_size is not None else cfg.BATCH_SIZE
    args.lr = args.lr if args.lr is not None else cfg.LR
    args.weight_decay = args.weight_decay if args.weight_decay is not None else cfg.WEIGHT_DECAY
    args.ann_file = resolve_config_path(cfg, args.ann_file or getattr(cfg, "ANN_FILE", str(DEFAULT_ANN_FILE)))

    set_seed(args.seed)
    device = resolve_device(args.device)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    work_base = Path(args.work_dir) if args.work_dir is not None else DEFAULT_WORK_ROOT / f"{model_type}_standalone"
    work_dir = work_base / run_name
    work_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, cfg_dict, args, work_dir)
    logger = setup_logger(work_dir / "train.log")
    wandb_run = init_wandb(args, work_dir, run_name)

    logger.info("Run directory: %s", work_dir)
    logger.info("Device: %s", device)
    logger.info("Config file: %s", args.config)
    logger.info("Saved config copy: %s", work_dir / Path(getattr(cfg, "CONFIG_PATH", args.config)).name)
    logger.info("Saved config json: %s", work_dir / "config.json")
    logger.info("Model type: %s", model_type)
    logger.info("Annotation file: %s", args.ann_file)
    logger.info(
        "Epochs: %d | Batch size: %d | LR: %.6g | Weight decay: %.6g",
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
    )

    train_set = MediapipeSignDataset(
        args.ann_file,
        split="train",
        clip_len=cfg.CLIP_LEN,
        num_clips=1,
        test_mode=False,
        repeat=cfg.TRAIN_REPEAT,
        zero_pad_short=getattr(cfg, "ZERO_PAD_SHORT", False),
        input_mode=getattr(cfg, "INPUT_MODE", "xy"),
        keypoint_normalize=getattr(cfg, "KEYPOINT_NORMALIZE", None),
        random_horizontal_flip=getattr(cfg, "RANDOM_HORIZONTAL_FLIP", None),
        short_sample_interpolation=getattr(cfg, "SHORT_SAMPLE_INTERPOLATION", None),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    saved_val_samples = None
    class_test_val_samples = None
    if args.val_mode == "ann":
        val_set = MediapipeSignDataset(
            args.ann_file,
            split="val",
            clip_len=cfg.CLIP_LEN,
            num_clips=cfg.TEST_NUM_CLIPS,
            test_mode=True,
            zero_pad_short=getattr(cfg, "ZERO_PAD_SHORT", False),
            input_mode=getattr(cfg, "INPUT_MODE", "xy"),
            keypoint_normalize=getattr(cfg, "KEYPOINT_NORMALIZE", None),
            random_horizontal_flip=getattr(cfg, "RANDOM_HORIZONTAL_FLIP", None),
            short_sample_interpolation=getattr(cfg, "SHORT_SAMPLE_INTERPOLATION", None),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        logger.info("Train samples: %d | Annotation validation enabled", len(train_set))
    else:
        saved_val_samples = load_saved_val_samples(
            args.saved_val_npz_dir,
            label_source=args.saved_val_label_source,
        )
        class_test_val_samples = load_class_test_val_samples(args.saved_val_class_test_dir)
        logger.info("Train samples: %d | Saved-NPZ val dir=%s", len(train_set), args.saved_val_npz_dir)
        logger.info(
            "Class-test val dir=%s selection=all_files_suffix_label",
            args.saved_val_class_test_dir,
        )

    model = build_model(cfg).to(device)
    if args.resume:
        info = load_checkpoint(model, args.resume, map_location=device, strict=False)
        logger.info(
            "Loaded checkpoint: %s | missing=%s | unexpected=%s",
            args.resume,
            info["missing_keys"],
            info["unexpected_keys"],
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    criterion = torch.nn.CrossEntropyLoss()
    best_top1 = 0.0
    best_epoch: Optional[int] = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        top1_sum = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy(logits.detach(), labels)
            batch = labels.numel()
            total += batch
            loss_sum += loss.item() * batch
            top1_sum += acc["top1"] * batch

        scheduler.step()
        train_metrics = {"loss": loss_sum / total, "top1": top1_sum / total}
        current_lr = scheduler.get_last_lr()[0]
        val_metrics = None
        best_saved = False

        should_val = epoch >= cfg.VAL_BEGIN and epoch % cfg.VAL_INTERVAL == 0
        if should_val:
            if args.val_mode == "ann":
                val_metrics = run_eval(model, val_loader, device)
            else:
                val_metrics = run_saved_npz_eval(
                    model,
                    saved_val_samples,
                    class_test_val_samples,
                    cfg,
                    device,
                )
            if val_metrics["top1"] >= best_top1:
                best_top1 = val_metrics["top1"]
                best_epoch = epoch
                best_saved = True
                save_checkpoint(
                    work_dir / "best.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_top1,
                    best_epoch,
                    cfg_dict,
                )
        save_checkpoint(
            work_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_top1,
            best_epoch,
            cfg_dict,
        )
        periodic_checkpoint_path = None
        if epoch % 50 == 0:
            periodic_checkpoint_path = work_dir / f"epoch_{epoch:03d}.pth"
            save_checkpoint(
                periodic_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_top1,
                best_epoch,
                cfg_dict,
            )

        log_payload = {
            "epoch": epoch,
            "lr": current_lr,
            "train/loss": train_metrics["loss"],
            "train/top1_acc": train_metrics["top1"],
        }
        logger.info("Epoch %03d/%03d | lr=%.6g", epoch, args.epochs, current_lr)
        logger.info(
            "  train | loss=%.4f | top1_acc=%.4f",
            train_metrics["loss"],
            train_metrics["top1"],
        )
        if val_metrics is not None:
            log_payload.update(
                {
                    "val/loss": val_metrics["loss"],
                    "val/top1_acc": val_metrics["top1"],
                    "val/top5_acc": val_metrics["top5"],
                }
            )
            logger.info(
                "  val   | loss=%.4f | top1_acc=%.4f | top5_acc=%.4f",
                val_metrics["loss"],
                val_metrics["top1"],
                val_metrics["top5"],
            )
            if "metrics_by_dataset" in val_metrics:
                for dataset_name, dataset_metrics in val_metrics["metrics_by_dataset"].items():
                    log_prefix = f"val/{dataset_name}"
                    log_payload.update(
                        {
                            f"{log_prefix}/loss": dataset_metrics["loss"],
                            f"{log_prefix}/top1_acc": dataset_metrics["top1"],
                            f"{log_prefix}/top5_acc": dataset_metrics["top5"],
                        }
                    )
                    logger.info(
                        "  val/%s | loss=%.4f | top1_acc=%.4f | top5_acc=%.4f",
                        dataset_name,
                        dataset_metrics["loss"],
                        dataset_metrics["top1"],
                        dataset_metrics["top5"],
                    )
        if best_saved:
            logger.info(
                "  checkpoint | best.pth saved at epoch %03d (val_top1_acc=%.4f)",
                epoch,
                best_top1,
            )
        if periodic_checkpoint_path is not None:
            logger.info(
                "  checkpoint | %s saved",
                periodic_checkpoint_path.name,
            )
        if wandb_run is not None:
            wandb_run.log(log_payload, step=epoch)

    with open(work_dir / "train_args.json", "w", encoding="utf-8") as f:
        saved_args = vars(args).copy()
        saved_args["run_dir"] = str(work_dir)
        saved_args["model_type"] = model_type
        saved_args["config_values"] = cfg_dict
        json.dump(saved_args, f, indent=2, ensure_ascii=False)

    if best_epoch is None:
        logger.info("Training finished. No best checkpoint was saved because validation did not run.")
    else:
        logger.info(
            "Training finished. Best checkpoint: epoch %03d (val_top1_acc=%.4f)",
            best_epoch,
            best_top1,
        )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
