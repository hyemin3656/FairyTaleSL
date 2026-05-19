"""
holistic_results_pose_align_{train/val/test} 포맷 →  main ST-GCN 포맷 변환기

입력 구조 (다운로드한 데이터):
  <src_root>/
    holistic_results_pose_align_train/
      {subject}_{class}/
        left_hand_21.npy   (T, 21, 4)
        right_hand_21.npy  (T, 21, 4)
        pose_33.npy        (T, 33, 4)
    holistic_results_pose_align_val/
    holistic_results_pose_align_test/

출력 구조 (main ST-GCN 학습용):
  <out_dir>/
    {label_idx:03d}/
      {subject_id:02d}.npy   (T, 225)  ← left63 | right63 | pose99
    label_map.txt             ← label_idx TAB class_id

실행:
  python ai_engine/pipelines/convert_holistic_to_flat.py \
      --src_root /path/to/downloaded_data \
      --out_dir  ai_engine/data/keypoints
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np

SPLIT_DIRS = {
    "train": "holistic_results",
    "val":   "holistic_results_val",
    "test":  "holistic_results_test",
}

POSE_N      = 33
HAND_N      = 21
COORD_DIM   = 3   # x, y, z만 사용 (4번째 visibility 제외)

FOLDER_RE = re.compile(r"^(\d+)_(\d+)$")


def load_npy_safe(path: Path, expected_v: int) -> np.ndarray | None:
    """(T, V, ≥3) → (T, V, 3) float32. 파일 없거나 shape 이상이면 None."""
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[1] < expected_v or arr.shape[2] < COORD_DIM:
        print(f"  [WARN] 예상치 못한 shape {arr.shape}: {path}")
        return None
    return arr[:, :expected_v, :COORD_DIM].astype(np.float32)


def collect_class_ids(src_root: Path) -> list[int]:
    """모든 split 폴더를 훑어 class_id 목록을 수집."""
    class_ids: set[int] = set()
    for split_name, dir_name in SPLIT_DIRS.items():
        split_dir = src_root / dir_name
        if not split_dir.exists():
            continue
        for folder in split_dir.iterdir():
            if not folder.is_dir():
                continue
            m = FOLDER_RE.match(folder.name)
            if m:
                class_ids.add(int(m.group(2)))
    return sorted(class_ids)


def build_label_map(class_ids: list[int]) -> dict[int, int]:
    """class_id → label_idx (0-based, 연속)"""
    return {cid: idx for idx, cid in enumerate(class_ids)}


def convert(src_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # label_map 생성
    class_ids = collect_class_ids(src_root)
    if not class_ids:
        raise RuntimeError(f"샘플 폴더를 찾을 수 없습니다: {src_root}")
    cid_to_idx = build_label_map(class_ids)

    label_map_path = out_dir / "label_map.txt"
    with open(label_map_path, "w", encoding="utf-8") as f:
        for cid, idx in cid_to_idx.items():
            f.write(f"{idx}\t{cid}\n")
    print(f"[label_map] {len(cid_to_idx)}개 클래스 → {label_map_path}")

    # 변환
    ok = skip = err = 0
    for split_name, dir_name in SPLIT_DIRS.items():
        split_dir = src_root / dir_name
        if not split_dir.exists():
            print(f"[SKIP] {split_dir} 없음")
            continue

        folders = sorted(d for d in split_dir.iterdir() if d.is_dir())
        print(f"\n[{split_name}] {len(folders)}개 샘플 변환 중...")

        for folder in folders:
            m = FOLDER_RE.match(folder.name)
            if not m:
                continue

            subject_id = int(m.group(1))
            class_id   = int(m.group(2))

            if class_id not in cid_to_idx:
                print(f"  [WARN] class_id={class_id} label_map에 없음, 건너뜀")
                err += 1
                continue

            label_idx = cid_to_idx[class_id]
            out_path  = out_dir / f"{label_idx:03d}" / f"{subject_id:02d}.npy"

            if out_path.exists():
                skip += 1
                continue

            # 키포인트 로드
            left  = load_npy_safe(folder / "left_hand_21.npy",  HAND_N)
            right = load_npy_safe(folder / "right_hand_21.npy", HAND_N)
            pose  = load_npy_safe(folder / "pose_33.npy",       POSE_N)

            if pose is None:
                print(f"  [ERR] pose_33.npy 없음: {folder}")
                err += 1
                continue

            T = pose.shape[0]

            def pad_or_trim(arr, target_t):
                if arr is None:
                    return np.zeros((target_t, HAND_N, COORD_DIM), np.float32)
                t = arr.shape[0]
                if t > target_t:
                    return arr[:target_t]
                if t < target_t:
                    pad = np.zeros((target_t - t, arr.shape[1], COORD_DIM), np.float32)
                    return np.concatenate([arr, pad], axis=0)
                return arr

            left  = pad_or_trim(left,  T)   # (T, 21, 3)
            right = pad_or_trim(right, T)   # (T, 21, 3)

            # NaN → 0
            left  = np.nan_to_num(left,  nan=0.0)
            right = np.nan_to_num(right, nan=0.0)
            pose  = np.nan_to_num(pose,  nan=0.0)

            # (T, 225) flat: left63 | right63 | pose99
            flat = np.concatenate([
                left.reshape(T, -1),   # 63
                right.reshape(T, -1),  # 63
                pose.reshape(T, -1),   # 99
            ], axis=1).astype(np.float32)

            assert flat.shape == (T, 225), flat.shape

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, flat)
            ok += 1

    print(f"\n완료: ok={ok}, skip(이미 존재)={skip}, err={err}")
    print(f"출력 경로: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True,
                        help="holistic_results_pose_align_* 폴더들이 있는 상위 디렉토리")
    parser.add_argument("--out_dir",  default="ai_engine/data/keypoints",
                        help="변환 결과 저장 경로 (default: ai_engine/data/keypoints)")
    args = parser.parse_args()

    convert(Path(args.src_root), Path(args.out_dir))


if __name__ == "__main__":
    main()
