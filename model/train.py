from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

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

from model.builder import build_model
from model.config_utils import config_to_dict, load_config, resolve_config_path
from model.model import load_checkpoint
from model.data import MediapipeSignDataset


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
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    logger.info("Train samples: %d | Val samples: %d", len(train_set), len(val_set))

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
            val_metrics = run_eval(model, val_loader, device)
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
        if best_saved:
            logger.info(
                "  checkpoint | best.pth saved at epoch %03d (val_top1_acc=%.4f)",
                epoch,
                best_top1,
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
