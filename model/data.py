from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_LEFT_SHOULDER = 11
DEFAULT_RIGHT_SHOULDER = 12
DEFAULT_LEFT_HIP = 23
DEFAULT_RIGHT_HIP = 24


def load_annotation(path: Union[str, Path]) -> dict:
    with open(Path(path).expanduser(), "rb") as f:
        return pickle.load(f)



def _get_joint_pair(
    cfg: Dict[str, Any],
    pair_name: str,
    num_joints: int,
) -> Tuple[int, int]:
    if pair_name == "shoulder":
        left = int(cfg.get("left_shoulder_index", DEFAULT_LEFT_SHOULDER))
        right = int(cfg.get("right_shoulder_index", DEFAULT_RIGHT_SHOULDER))
    elif pair_name == "hip":
        left = int(cfg.get("left_hip_index", DEFAULT_LEFT_HIP))
        right = int(cfg.get("right_hip_index", DEFAULT_RIGHT_HIP))
    else:
        raise ValueError(f"Unsupported joint pair: {pair_name}")

    if not (0 <= left < num_joints and 0 <= right < num_joints):
        raise ValueError(
            f"{pair_name} indices ({left}, {right}) are out of range for V={num_joints}."
        )
    return left, right


def _joint_pair_center(
    keypoint: np.ndarray,
    left_index: int,
    right_index: int,
    dims: int,
) -> np.ndarray:
    return (
        keypoint[:, :, left_index, :dims] + keypoint[:, :, right_index, :dims]
    ) * 0.5


def normalize_keypoints(
    keypoint: np.ndarray,
    normalize_cfg: Union[Dict[str, Any], bool, None],
) -> np.ndarray:
    if not normalize_cfg:
        return keypoint
    if normalize_cfg is True:
        normalize_cfg = {}
    if not isinstance(normalize_cfg, dict):
        raise TypeError("keypoint normalization config must be a dict, bool, or None.")
    if not normalize_cfg.get("enabled", True):
        return keypoint
    if keypoint.ndim != 4:
        raise ValueError(f"keypoint must have shape (M, T, V, C), got {keypoint.shape}.")
    if keypoint.shape[-1] < 2:
        raise ValueError("keypoint normalization requires at least x/y channels.")

    num_joints = keypoint.shape[2]
    eps = float(normalize_cfg.get("eps", 1e-6))
    default_coord_dims = min(3, keypoint.shape[-1])
    coord_dims = min(int(normalize_cfg.get("coord_dims", default_coord_dims)), keypoint.shape[-1])
    coord_dims = max(2, coord_dims)
    out = keypoint.astype(np.float32, copy=True)

    center_mode = normalize_cfg.get("center", "shoulder")
    if center_mode not in (None, False, "none"):
        if center_mode == "torso":
            shoulder_l, shoulder_r = _get_joint_pair(normalize_cfg, "shoulder", num_joints)
            hip_l, hip_r = _get_joint_pair(normalize_cfg, "hip", num_joints)
            shoulder_center = _joint_pair_center(out, shoulder_l, shoulder_r, coord_dims)
            hip_center = _joint_pair_center(out, hip_l, hip_r, coord_dims)
            center = (shoulder_center + hip_center) * 0.5
        else:
            left, right = _get_joint_pair(normalize_cfg, str(center_mode), num_joints)
            center = _joint_pair_center(out, left, right, coord_dims)
        out[..., :coord_dims] -= center[:, :, None, :]

    scale_mode = normalize_cfg.get("scale", "shoulder")
    if scale_mode not in (None, False, "none"):
        scale_dims = normalize_cfg.get("scale_dims", "xy")
        if scale_dims == "xy":
            scale_dims = 2
        elif scale_dims == "xyz":
            scale_dims = min(3, keypoint.shape[-1])
        else:
            scale_dims = int(scale_dims)
        scale_dims = max(2, min(scale_dims, keypoint.shape[-1]))

        if scale_mode == "torso":
            shoulder_l, shoulder_r = _get_joint_pair(normalize_cfg, "shoulder", num_joints)
            hip_l, hip_r = _get_joint_pair(normalize_cfg, "hip", num_joints)
            shoulder_center = _joint_pair_center(out, shoulder_l, shoulder_r, scale_dims)
            hip_center = _joint_pair_center(out, hip_l, hip_r, scale_dims)
            delta = shoulder_center - hip_center
        else:
            left, right = _get_joint_pair(normalize_cfg, str(scale_mode), num_joints)
            delta = out[:, :, right, :scale_dims] - out[:, :, left, :scale_dims]
        scale = np.linalg.norm(delta, axis=-1)
        scale = np.where(scale > eps, scale, 1.0).astype(np.float32)
        out[..., :coord_dims] /= scale[:, :, None, None]

    if normalize_cfg.get("rotate", False):
        left, right = _get_joint_pair(
            normalize_cfg,
            str(normalize_cfg.get("rotate_pair", "shoulder")),
            num_joints,
        )
        delta = out[:, :, right, :2] - out[:, :, left, :2]
        angle = np.arctan2(delta[..., 1], delta[..., 0])
        cos = np.cos(angle).astype(np.float32)
        sin = np.sin(angle).astype(np.float32)
        x = out[..., 0].copy()
        y = out[..., 1].copy()
        out[..., 0] = x * cos[:, :, None] + y * sin[:, :, None]
        out[..., 1] = -x * sin[:, :, None] + y * cos[:, :, None]

    return out


def interpolate_short_sample(
    keypoint: np.ndarray,
    keypoint_score: Union[np.ndarray, None],
    target_frames: int,
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    current_frames = keypoint.shape[1]
    if current_frames >= target_frames:
        return keypoint, keypoint_score
    if current_frames <= 0:
        raise ValueError("Cannot interpolate a sample with no frames.")
    if current_frames == 1:
        keypoint = np.repeat(keypoint, target_frames, axis=1)
        if keypoint_score is not None:
            keypoint_score = np.repeat(keypoint_score, target_frames, axis=1)
        return keypoint.astype(np.float32, copy=False), keypoint_score

    num_insert = target_frames - current_frames
    num_intervals = current_frames - 1
    insert_counts = np.diff(
        np.round(np.linspace(0, num_insert, num_intervals + 1)).astype(np.int64)
    )

    keypoint_frames = []
    score_frames = [] if keypoint_score is not None else None
    for frame_idx, insert_count in enumerate(insert_counts):
        keypoint_frames.append(keypoint[:, frame_idx : frame_idx + 1])
        if score_frames is not None:
            score_frames.append(keypoint_score[:, frame_idx : frame_idx + 1])
        for insert_idx in range(1, int(insert_count) + 1):
            alpha = insert_idx / (int(insert_count) + 1)
            left = keypoint[:, frame_idx : frame_idx + 1]
            right = keypoint[:, frame_idx + 1 : frame_idx + 2]
            keypoint_frames.append((1.0 - alpha) * left + alpha * right)
            if score_frames is not None:
                score_left = keypoint_score[:, frame_idx : frame_idx + 1]
                score_right = keypoint_score[:, frame_idx + 1 : frame_idx + 2]
                score_frames.append((1.0 - alpha) * score_left + alpha * score_right)

    keypoint_frames.append(keypoint[:, -1:])
    out_keypoint = np.concatenate(keypoint_frames, axis=1).astype(np.float32, copy=False)
    if score_frames is None:
        return out_keypoint, None
    score_frames.append(keypoint_score[:, -1:])
    out_score = np.concatenate(score_frames, axis=1).astype(np.float32, copy=False)
    return out_keypoint, out_score


def resolve_short_interpolation_target(
    interpolation_cfg: Union[Dict[str, Any], bool, None],
    clip_len: int,
    num_clips: int,
) -> Union[int, None]:
    if not interpolation_cfg:
        return None
    if interpolation_cfg is True:
        interpolation_cfg = {}
    if not isinstance(interpolation_cfg, dict):
        raise TypeError("short sample interpolation config must be a dict, bool, or None.")
    if not interpolation_cfg.get("enabled", True):
        return None

    target = interpolation_cfg.get("target", "clip_len")
    if target == "clip_len":
        return int(clip_len)
    if target == "sampled_frames":
        return int(clip_len * num_clips)
    return int(target)


def input_mode_to_channels(input_mode: str) -> int:
    if input_mode == "xy":
        return 2
    if input_mode == "xyz":
        return 3
    if input_mode == "xyscore":
        return 3
    if input_mode == "xyzscore":
        return 4
    raise ValueError("input_mode must be one of: xy, xyz, xyscore, xyzscore")


def build_keypoint_features(
    keypoint: np.ndarray,
    keypoint_score: Union[np.ndarray, None],
    input_mode: str,
) -> np.ndarray:
    if input_mode == "xy":
        if keypoint.shape[-1] < 2:
            raise ValueError(f"xy input requires keypoint C>=2, got {keypoint.shape}.")
        return keypoint[..., :2].astype(np.float32)
    if input_mode == "xyz":
        if keypoint.shape[-1] < 3:
            raise ValueError(f"xyz input requires keypoint C>=3, got {keypoint.shape}.")
        return keypoint[..., :3].astype(np.float32)
    if input_mode == "xyscore":
        if keypoint.shape[-1] < 2:
            raise ValueError(f"xyscore input requires keypoint C>=2, got {keypoint.shape}.")
        if keypoint_score is None:
            raise ValueError("xyscore input requires keypoint_score in the sample.")
        score = np.asarray(keypoint_score, dtype=np.float32)[..., None]
        return np.concatenate([keypoint[..., :2].astype(np.float32), score], axis=-1)
    if input_mode == "xyzscore":
        if keypoint.shape[-1] < 3:
            raise ValueError(f"xyzscore input requires keypoint C>=3, got {keypoint.shape}.")
        if keypoint_score is None:
            raise ValueError("xyzscore input requires keypoint_score in the sample.")
        score = np.asarray(keypoint_score, dtype=np.float32)[..., None]
        return np.concatenate([keypoint[..., :3].astype(np.float32), score], axis=-1)
    raise ValueError("input_mode must be one of: xy, xyz, xyscore, xyzscore")


def uniform_sample_frames(
    num_frames: int,
    clip_len: int,
    num_clips: int = 1,
    test_mode: bool = False,
    seed: int = 255,
    zero_pad_short: bool = False,
) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    if test_mode:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    all_inds = []
    for i in range(num_clips):
        if num_frames < clip_len:
            if zero_pad_short:
                inds = np.full(clip_len, -1, dtype=np.int64)
                inds[:num_frames] = np.arange(num_frames, dtype=np.int64)
            else:
                start = i if test_mode and num_frames < num_clips else rng.randint(0, num_frames)
                if test_mode and num_frames >= num_clips:
                    start = i * num_frames // num_clips
                inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            chosen = rng.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[chosen] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array([j * num_frames // clip_len for j in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = rng.randint(bsize)
            inds = bst + offset
        all_inds.append(inds)

    inds = np.concatenate(all_inds).astype(np.int64)
    valid = inds >= 0
    inds[valid] = np.mod(inds[valid], num_frames)
    return inds


def preprocess_keypoint_sample(
    sample: Dict[str, Any],
    clip_len: int,
    num_clips: int = 1,
    test_mode: bool = False,
    num_person: int = 1,
    seed: int = 255,
    zero_pad_short: bool = False,
    input_mode: str = "xy",
    keypoint_normalize: Union[Dict[str, Any], bool, None] = None,
    short_sample_interpolation: Union[Dict[str, Any], bool, None] = None,
) -> np.ndarray:
    keypoint = np.asarray(sample["keypoint"], dtype=np.float32)
    keypoint_score = sample.get("keypoint_score")
    if keypoint_score is not None:
        keypoint_score = np.asarray(keypoint_score, dtype=np.float32)
    total_frames = int(sample.get("total_frames", keypoint.shape[1]))
    if keypoint.shape[1] != total_frames:
        raise ValueError(
            f"total_frames={total_frames} does not match keypoint T={keypoint.shape[1]}."
        )

    interpolation_target = resolve_short_interpolation_target(
        short_sample_interpolation,
        clip_len=clip_len,
        num_clips=num_clips,
    )
    if interpolation_target is not None and total_frames < interpolation_target:
        keypoint, keypoint_score = interpolate_short_sample(
            keypoint,
            keypoint_score,
            target_frames=interpolation_target,
        )
        total_frames = keypoint.shape[1]

    keypoint = normalize_keypoints(keypoint, keypoint_normalize)

    frame_inds = uniform_sample_frames(
        total_frames,
        clip_len=clip_len,
        num_clips=num_clips,
        test_mode=test_mode,
        seed=seed,
        zero_pad_short=zero_pad_short,
    )

    valid = frame_inds >= 0
    feature_dim = input_mode_to_channels(input_mode)
    sampled_keypoint = np.zeros(
        (keypoint.shape[0], frame_inds.shape[0], keypoint.shape[2], feature_dim),
        dtype=np.float32,
    )
    if valid.any():
        sampled_score = None
        if keypoint_score is not None:
            sampled_score = keypoint_score[:, frame_inds[valid], :]
        sampled_keypoint[:, valid] = build_keypoint_features(
            keypoint[:, frame_inds[valid], :, :],
            sampled_score,
            input_mode,
        )
    keypoint = sampled_keypoint

    cur_num_person = keypoint.shape[0]
    if cur_num_person < num_person:
        pad = np.zeros((num_person - cur_num_person,) + keypoint.shape[1:], dtype=np.float32)
        keypoint = np.concatenate([keypoint, pad], axis=0)
    elif cur_num_person > num_person:
        keypoint = keypoint[:num_person]

    m, t, v, c = keypoint.shape
    if t % num_clips != 0:
        raise ValueError(f"Sampled frames {t} must be divisible by num_clips {num_clips}.")
    keypoint = keypoint.reshape(m, num_clips, t // num_clips, v, c)
    keypoint = keypoint.transpose(1, 0, 2, 3, 4)
    return np.ascontiguousarray(keypoint)


class MediapipeSignDataset(Dataset):
    def __init__(
        self,
        ann_file: Union[str, Path],
        split: str,
        clip_len: int = 100,
        num_clips: int = 1,
        test_mode: bool = False,
        repeat: int = 1,
        seed: int = 255,
        zero_pad_short: bool = False,
        input_mode: str = "xy",
        keypoint_normalize: Union[Dict[str, Any], bool, None] = None,
        short_sample_interpolation: Union[Dict[str, Any], bool, None] = None,
    ) -> None:
        ann = load_annotation(ann_file)
        split_names = ann["split"][split]
        annotations = {item["frame_dir"]: item for item in ann["annotations"]}
        self.samples: List[dict] = [annotations[name] for name in split_names]
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.repeat = repeat
        self.seed = seed
        self.zero_pad_short = zero_pad_short
        self.input_mode = input_mode
        self.keypoint_normalize = keypoint_normalize
        self.short_sample_interpolation = short_sample_interpolation

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index % len(self.samples)]
        keypoint = preprocess_keypoint_sample(
            sample,
            clip_len=self.clip_len,
            num_clips=self.num_clips,
            test_mode=self.test_mode,
            seed=self.seed,
            zero_pad_short=self.zero_pad_short,
            input_mode=self.input_mode,
            keypoint_normalize=self.keypoint_normalize,
            short_sample_interpolation=self.short_sample_interpolation,
        )
        if keypoint.shape[0] == 1:
            keypoint = keypoint[0]
        label = int(sample["label"])
        return torch.from_numpy(keypoint), torch.tensor(label, dtype=torch.long)


def build_npz_sample(npz_path: Union[str, Path]) -> dict:
    data = np.load(Path(npz_path).expanduser(), allow_pickle=True)
    required = ["pose", "left_hand", "right_hand"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing arrays in {npz_path}: {missing}")

    pose = np.asarray(data["pose"], dtype=np.float32)[:, :23]
    left = np.asarray(data["left_hand"], dtype=np.float32)[:, :21]
    right = np.asarray(data["right_hand"], dtype=np.float32)[:, :21]

    arrays = {"pose": pose, "left_hand": left, "right_hand": right}
    t_values = {name: arr.shape[0] for name, arr in arrays.items()}
    if len(set(t_values.values())) != 1:
        raise ValueError("pose, left_hand, and right_hand must have the same frame length.")
    for name, arr in arrays.items():
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError(f"{name} must have shape (T, V, C>=3), got {arr.shape}.")
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN.")

    keypoint = np.concatenate([pose[..., :3], left[..., :3], right[..., :3]], axis=1)
    pose_score = np.ones(pose.shape[:2], dtype=np.float32)
    left_score = left[..., 3] if left.shape[-1] > 3 else np.ones(left.shape[:2], dtype=np.float32)
    right_score = right[..., 3] if right.shape[-1] > 3 else np.ones(right.shape[:2], dtype=np.float32)
    keypoint_score = np.concatenate([pose_score, left_score, right_score], axis=1)
    return {
        "frame_dir": Path(npz_path).stem,
        "label": -1,
        "total_frames": keypoint.shape[0],
        "keypoint": keypoint[None].astype(np.float32),
        "keypoint_score": keypoint_score[None].astype(np.float32),
    }

