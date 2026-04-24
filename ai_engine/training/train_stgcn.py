"""
ST-GCN 고립단어 분류 베이스라인 학습

실행 (로컬):
  python ai_engine/training/train_stgcn.py \
      --keypoints_dir ai_engine/data/keypoints \
      --split_csv     split_result.csv \
      --label_map     ai_engine/data/keypoints/label_map.txt \
      --class_pickle  class_label.p \
      --out_dir       weights/stgcn_baseline

Colab T4 권장 옵션
  --batch_size 64 --num_workers 2 --device cuda --epochs 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 프로젝트 루트의 ai_engine 패키지 가져오기
THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from training.dataset import KSLKeypointDataset, load_class_words  # noqa: E402
from models.stgcn import STGCN  # noqa: E402


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keypoints_dir", required=True)
    p.add_argument("--split_csv",     required=True)
    p.add_argument("--label_map",     required=True)
    p.add_argument("--class_pickle",  required=True)
    p.add_argument("--out_dir",       default="weights/stgcn_baseline")
    p.add_argument("--t_fixed",       type=int, default=105)
    p.add_argument("--epochs",        type=int, default=80)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--step_size",     type=int, default=20)
    p.add_argument("--gamma",         type=float, default=0.5)
    p.add_argument("--num_workers",   type=int, default=2)
    p.add_argument("--device",        default="auto")
    p.add_argument("--seed",          type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"[device] {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Datasets / Loaders ───────────────────────────────────────────────
    ds_args = dict(
        split_csv=args.split_csv,
        keypoints_dir=args.keypoints_dir,
        label_map_path=args.label_map,
        t_fixed=args.t_fixed,
    )
    train_ds = KSLKeypointDataset(split="train", **ds_args)
    val_ds   = KSLKeypointDataset(split="val",   **ds_args)
    test_ds  = KSLKeypointDataset(split="test",  **ds_args)
    print(f"[data] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, "
          f"classes={train_ds.num_classes}")
    print(f"[data] dropped(zero-hand) train={train_ds._dropped_zero_hand}, "
          f"val={val_ds._dropped_zero_hand}, test={test_ds._dropped_zero_hand}")

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    # ── Model / Optim ────────────────────────────────────────────────────
    model = STGCN(num_classes=train_ds.num_classes, mode="classify").to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] STGCN classify, params={n_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                                gamma=args.gamma)

    # ── Train ────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_path = out_dir / "stgcn_best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_correct += (logits.argmax(-1) == y).sum().item()
            tr_total += x.size(0)
        scheduler.step()
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:>3d}  lr={lr_now:.2e}  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}  ({dt:.1f}s)")
        history.append({"epoch": epoch, "lr": lr_now,
                        "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ↑ saved best (val_acc={val_acc:.3f}) → {best_path}")

    # ── Test (best checkpoint) ───────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"\n[test] best val_acc={best_val_acc:.3f}  →  test_acc={test_acc:.3f} (loss={test_loss:.4f})")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "test_result.json", "w") as f:
        json.dump({"best_val_acc": best_val_acc,
                   "test_acc": test_acc, "test_loss": test_loss}, f, indent=2)

    # 영어 라벨 저장 (추론용)
    words = load_class_words(args.class_pickle, train_ds.num_classes)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
