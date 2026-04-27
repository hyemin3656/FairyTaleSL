"""
KSL 키포인트 시퀀스 → ST-GCN 입력 텐서 변환 Dataset

데이터 소스
  data/keypoints/{label_idx:03d}/{subject_id:02d}.npy   shape (T, 225)
    [0:63]    left_hand  (21 × 3)
    [63:126]  right_hand (21 × 3)
    [126:225] pose       (33 × 3)   ← 본 베이스라인은 미사용

분할 정의
  split_result.csv: sample_id, subject_id, class_id, split
    class_id 는 원본 gloss 코드(1~77, 불연속) → label_map.txt 로 label_idx(0~66) 변환

라벨 의미
  class_label.p: {0: ' ', 1: 'hi', ...}  → cl[label_idx + 1] 가 영어 단어
"""
from __future__ import annotations

import csv
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

HAND_N = 21
NUM_HANDS = 2
HAND_DIMS = HAND_N * 3 * NUM_HANDS  # 126


@dataclass
class SampleRef:
    sample_id: int
    subject_id: int
    label_idx: int
    npy_path: str


def load_label_map(path: str) -> dict[int, str]:
    """label_idx(int) → gloss_id(str, '01' 등)"""
    m: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            idx, gid = line.strip().split("\t")
            m[int(idx)] = gid
    return m


def load_class_words(pickle_path: str, num_classes: int) -> list[str]:
    """label_idx 순서대로 영어 단어 리스트. cl[label_idx + 1] 매핑."""
    with open(pickle_path, "rb") as f:
        cl = pickle.load(f)
    return [cl[i + 1] for i in range(num_classes)]


def _build_refs(
    split_csv: str,
    keypoints_dir: str,
    label_map: dict[int, str],
    split: str,
) -> list[SampleRef]:
    gloss_to_idx = {int(v): k for k, v in label_map.items()}
    refs: list[SampleRef] = []
    with open(split_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            cid = int(row["class_id"])
            if cid not in gloss_to_idx:
                continue
            label_idx = gloss_to_idx[cid]
            sid = int(row["subject_id"])
            npy = os.path.join(keypoints_dir, f"{label_idx:03d}", f"{sid:02d}.npy")
            if not os.path.exists(npy):
                continue
            refs.append(SampleRef(int(row["sample_id"]), sid, label_idx, npy))
    return refs


def _filter_zero_hand(refs: list[SampleRef]) -> tuple[list[SampleRef], int]:
    """양손 키포인트가 시퀀스 내내 0인 샘플 제외."""
    kept: list[SampleRef] = []
    dropped = 0
    for r in refs:
        arr = np.load(r.npy_path, mmap_mode="r")
        left_zero = (arr[:, :63].sum(axis=1) == 0).all()
        right_zero = (arr[:, 63:126].sum(axis=1) == 0).all()
        if left_zero and right_zero:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def _normalize_hands(hands: np.ndarray) -> np.ndarray:
    """
    손목 기준 평행이동 + 손 크기로 스케일.
    hands: (T, 2, 21, 3)  →  same shape, normalized.
    """
    out = hands.copy()
    # 손목 = 노드 0
    wrist = out[:, :, 0:1, :]                        # (T, 2, 1, 3)
    out = out - wrist                                # 평행이동
    # 스케일: 손목→중지 시작점(노드 9)까지 거리, 손별·프레임별
    span = np.linalg.norm(out[:, :, 9, :], axis=-1, keepdims=True)  # (T, 2, 1)
    span = np.where(span < 1e-6, 1.0, span)[..., None]              # (T, 2, 1, 1)
    out = out / span
    # 정규화 후에도 감지 실패(원래 0 벡터)는 0 유지
    zero_mask = (hands.sum(axis=-1, keepdims=True) == 0)             # (T, 2, 21, 1)
    out = np.where(zero_mask, 0.0, out)
    return out.astype(np.float32)


def _fit_length(seq: np.ndarray, t_fixed: int) -> np.ndarray:
    """T 차원을 t_fixed로 맞춤: center crop 또는 zero-pad."""
    T = seq.shape[0]
    if T == t_fixed:
        return seq
    if T > t_fixed:
        start = (T - t_fixed) // 2
        return seq[start:start + t_fixed]
    pad = np.zeros((t_fixed - T,) + seq.shape[1:], dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)


def _random_crop_or_pad(seq: np.ndarray, t_fixed: int) -> np.ndarray:
    """학습용: T 차원을 t_fixed로 맞추되 시작점을 랜덤화."""
    T = seq.shape[0]
    if T == t_fixed:
        return seq
    if T > t_fixed:
        start = np.random.randint(0, T - t_fixed + 1)
        return seq[start:start + t_fixed]
    pad_total = t_fixed - T
    pad_left = np.random.randint(0, pad_total + 1)
    pad_right = pad_total - pad_left
    pad_l = np.zeros((pad_left,)  + seq.shape[1:], dtype=seq.dtype)
    pad_r = np.zeros((pad_right,) + seq.shape[1:], dtype=seq.dtype)
    return np.concatenate([pad_l, seq, pad_r], axis=0)


class KSLKeypointDataset(Dataset):
    """
    반환 텐서: (3, T_fixed, 21, 2) float32   +  label_idx int
      C=3 (x,y,z), T=t_fixed, V=21, M=2

    augment=True 일 때 적용 (학습 split만):
      - 시간축 random crop / random-position pad
      - 좌표 가우시안 노이즈 (σ=aug_noise_std), 0벡터(미검출) 프레임은 제외
      - 좌우 손 swap (확률 aug_hand_swap_prob, 기본 0 — handedness가 의미있는 단어가 다수)
    """

    def __init__(
        self,
        split_csv: str,
        keypoints_dir: str,
        label_map_path: str,
        split: str,
        t_fixed: int = 105,
        normalize: bool = True,
        drop_zero_hand: bool = False,
        augment: bool = False,
        aug_noise_std: float = 0.02,
        aug_hand_swap_prob: float = 0.0,
    ):
        assert split in {"train", "val", "test"}
        self.t_fixed = t_fixed
        self.normalize = normalize
        self.augment = augment
        self.aug_noise_std = aug_noise_std
        self.aug_hand_swap_prob = aug_hand_swap_prob
        self.label_map = load_label_map(label_map_path)
        self.num_classes = len(self.label_map)

        refs = _build_refs(split_csv, keypoints_dir, self.label_map, split)
        if drop_zero_hand:
            refs, dropped = _filter_zero_hand(refs)
            self._dropped_zero_hand = dropped
        else:
            self._dropped_zero_hand = 0
        self.refs = refs

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ref = self.refs[idx]
        arr = np.load(ref.npy_path)                       # (T, 225)
        hands = arr[:, :HAND_DIMS].reshape(-1, NUM_HANDS, HAND_N, 3)  # (T, 2, 21, 3)
        if self.normalize:
            hands = _normalize_hands(hands)

        if self.augment:
            hands = _random_crop_or_pad(hands, self.t_fixed)
            if self.aug_noise_std > 0:
                noise = (np.random.randn(*hands.shape).astype(np.float32)
                         * self.aug_noise_std)
                # 0벡터(미검출) 프레임에는 노이즈 추가 X
                zero_mask = (hands.sum(axis=-1, keepdims=True) == 0)
                hands = np.where(zero_mask, hands, hands + noise)
            if self.aug_hand_swap_prob > 0 and np.random.rand() < self.aug_hand_swap_prob:
                hands = hands[:, ::-1, :, :].copy()
                hands[..., 0] = -hands[..., 0]            # x도 미러링
        else:
            hands = _fit_length(hands, self.t_fixed)

        # → (C, T, V, M)
        x = np.transpose(hands, (3, 0, 2, 1))             # (3, T, 21, 2)
        return torch.from_numpy(x).float(), ref.label_idx
