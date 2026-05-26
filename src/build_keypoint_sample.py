from pathlib import Path
import numpy as np
from tool.create_mediapipe_sign_ann import (
    COORD_DIM,
    NUM_HAND,
    NUM_NODE,
    NUM_POSE,
    ensure_tvc,
    load_npy,
    nan_to_zero_with_score,
    pad_or_trim_time,
    make_zero_hand,
)

POSE_FILE = "pose_33.npy"
LEFT_HAND_FILE = "left_hand_21.npy"
RIGHT_HAND_FILE = "right_hand_21.npy"

def build_keypoint_sample_from_dir_or_arr(keypoint_dir=None, arrs=None, allow_missing_hands=True):
    if keypoint_dir:
        keypoint_dir = Path(keypoint_dir).expanduser().resolve()

        pose = load_npy(keypoint_dir / POSE_FILE)
        left = load_npy(keypoint_dir / LEFT_HAND_FILE)
        right = load_npy(keypoint_dir / RIGHT_HAND_FILE)
    sample_name = keypoint_dir.name if keypoint_dir else "arrs"
    if arrs is not None:
        pose = arrs[0]
        left = arrs[1]
        right = arrs[2]

    if pose is None:
        raise FileNotFoundError(f"Missing {POSE_FILE}: {keypoint_dir}")
    pose = ensure_tvc(pose, NUM_POSE, POSE_FILE)
    total_frames = pose.shape[0]

    if left is None:
        if not allow_missing_hands:
            raise ValueError("left hand keypoints are required")
        left = make_zero_hand(total_frames)
    else:
        left = ensure_tvc(left, NUM_HAND, LEFT_HAND_FILE)
        left = pad_or_trim_time(left, total_frames)

    if right is None:
        if not allow_missing_hands:
            raise ValueError("right hand keypoints are required")
        right = make_zero_hand(total_frames)
    else:
        right = ensure_tvc(right, NUM_HAND, RIGHT_HAND_FILE)
        right = pad_or_trim_time(right, total_frames)

    pose, pose_score = nan_to_zero_with_score(pose)
    left, left_score = nan_to_zero_with_score(left)
    right, right_score = nan_to_zero_with_score(right)

    keypoint = np.concatenate([pose, left, right], axis=1)
    keypoint_score = np.concatenate([pose_score, left_score, right_score], axis=1)

    return build_keypoint_sample(
        keypoint=keypoint,
        keypoint_score=keypoint_score,
        sample_name=sample_name)

def build_keypoint_sample_from_total_npy(keypoint_npy):
    keypoint_npy = Path(keypoint_npy).expanduser().resolve()
    keypoint = np.asarray(np.load(keypoint_npy))

    if keypoint.ndim == 4:
        if keypoint.shape[0] != 1:
            raise ValueError(
                "A 4D --keypoint-npy must have shape [1, T, 65, 3], "
                f"got {keypoint.shape}")
        keypoint = keypoint[0]

    keypoint = ensure_tvc(keypoint, NUM_NODE, keypoint_npy.name)
    keypoint, keypoint_score = nan_to_zero_with_score(keypoint)

    return build_keypoint_sample(
        keypoint=keypoint,
        keypoint_score=keypoint_score,
        sample_name=keypoint_npy.stem)


def build_keypoint_sample(keypoint, keypoint_score, sample_name):
    total_frames = keypoint.shape[0]

    assert keypoint.shape == (total_frames, NUM_NODE, COORD_DIM), keypoint.shape
    assert keypoint_score.shape == (total_frames, NUM_NODE), keypoint_score.shape

    return {
        "frame_dir": sample_name,
        "total_frames": total_frames,
        "keypoint": keypoint[None, ...].astype(np.float32),
        "keypoint_score": keypoint_score[None, ...].astype(np.float32),
    }
