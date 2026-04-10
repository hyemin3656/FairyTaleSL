"""
추출된 키포인트 데이터셋 통계 확인 및 시각화

실행:
  python check_dataset.py --data_dir ./data/keypoints
"""
import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    label_map_path = os.path.join(data_dir, "label_map.txt")

    # 라벨 맵 로드
    label_map = {}  # idx → gloss_id
    if os.path.exists(label_map_path):
        with open(label_map_path) as f:
            for line in f:
                idx, gid = line.strip().split("\t")
                label_map[int(idx)] = gid

    class_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    print(f"{'클래스':>5}  {'글로스ID':>8}  {'샘플수':>6}  {'평균T':>6}  {'minT':>5}  {'maxT':>5}")
    print("-" * 50)

    all_lengths = []
    total_samples = 0
    zero_hand_count = 0

    for cls_dir in class_dirs:
        cls_idx = int(cls_dir)
        gloss_id = label_map.get(cls_idx, "?")
        npy_files = sorted(
            f for f in os.listdir(os.path.join(data_dir, cls_dir))
            if f.endswith(".npy")
        )
        if not npy_files:
            continue

        lengths = []
        for nf in npy_files:
            arr = np.load(os.path.join(data_dir, cls_dir, nf))  # (T, 225)
            lengths.append(arr.shape[0])

            # 손 랜드마크가 없는 프레임 비율 확인
            left_zero  = (arr[:, :63].sum(axis=1) == 0).sum()
            right_zero = (arr[:, 63:126].sum(axis=1) == 0).sum()
            if left_zero == arr.shape[0] and right_zero == arr.shape[0]:
                zero_hand_count += 1

        all_lengths.extend(lengths)
        total_samples += len(npy_files)
        print(
            f"{cls_idx:>5}  {gloss_id:>8}  {len(npy_files):>6}  "
            f"{np.mean(lengths):>6.1f}  {min(lengths):>5}  {max(lengths):>5}"
        )

    print("-" * 50)
    print(f"\n총 샘플: {total_samples}")
    print(f"총 클래스: {len(class_dirs)}")
    if all_lengths:
        print(f"프레임 길이 — min:{min(all_lengths)}, max:{max(all_lengths)}, "
              f"avg:{np.mean(all_lengths):.1f}, median:{np.median(all_lengths):.1f}")
    print(f"손 감지 실패 시퀀스: {zero_hand_count}개 (양손 모두 0인 경우)")
    print(f"\n권장 고정 길이 (T_fixed): {int(np.percentile(all_lengths, 75))} 프레임 "
          f"(75th percentile)")


if __name__ == "__main__":
    main()
