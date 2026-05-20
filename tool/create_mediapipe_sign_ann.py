import os
import csv
import ast
import json
import pickle
from pathlib import Path

import numpy as np

KSL_DIR = Path("/home/ubuntu/runyourai/ksl")
ROOT_DIR = KSL_DIR / "content/gloss_sequences_splited"
OUT_PKL = KSL_DIR / "mediapipe_sign_3d.pkl"
TEMPLATE_CSV = KSL_DIR / "gloss_sequence_templates.csv"

RANDOM_SEED = 42

POSE_FILE = "pose_33.npy"
LEFT_HAND_FILE = "left_hand_21.npy"
RIGHT_HAND_FILE = "right_hand_21.npy"

# 설정
NUM_POSE = 23       # pose 0~22
NUM_HAND = 21
NUM_NODE = 65       # 23 + 21 + 21
NUM_PERSON = 1
COORD_DIM = 3       # x, y, z



def load_npy(path: Path):
    if not path.exists():
        return None
    arr = np.load(path)
    return arr


def ensure_tvc(arr, expected_v=None, name="array"):
    """
    annotation의 keypoint의 key에 들어갈 array로 슬라이싱
    arr.shape :  [T, V, C] (일반적으로 MediaPipe 저장 결과가 [T, V, C]라고 가정).
    T: number of frames 
    V: number of keypoints 
    C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint, C=4 for adding visibility ).
    """
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape [T, V, C], but got {arr.shape}")

    T, V, C = arr.shape

    if expected_v is not None and V < expected_v:
        raise ValueError(f"{name} has too few keypoints: expected >= {expected_v}, got {V}")

    if C < COORD_DIM:
        raise ValueError(f"{name} must have at least {COORD_DIM} coordinates, got C={C}")

    return arr[:, :expected_v, :COORD_DIM].astype(np.float32)


def make_zero_hand(T):
    return np.zeros((T, NUM_HAND, COORD_DIM), dtype=np.float32)


def pad_or_trim_time(arr, target_T):
    """
    같은 sample내의 각 npy의 T가 다를 경우 target_T에 맞춘다.
    짧으면 0 padding, 길면 자른다.
    """
    T, V, C = arr.shape

    if T == target_T:
        return arr

    if T > target_T:
        return arr[:target_T]

    padded = np.zeros((target_T, V, C), dtype=arr.dtype)
    padded[:T] = arr
    return padded


def nan_to_zero_with_score(arr):
    """
    NaN이 있는 keypoint는 좌표 0, score 0 으로 설정
    정상 keypoint는 score 1.
    arr: #(T, NUM_POSE or NUM_HAND, COORD_DIM)
    """
    invalid = np.isnan(arr).any(axis=-1)  # [T, NUM_POSE] (keypoint-level)
    score = (~invalid).astype(np.float32)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[invalid] = 0.0

    return arr.astype(np.float32), score.astype(np.float32)

def remap_label(label):
    #비어있는 클래스 제거한 클래스 label 반환
    #label : 1~77
    missing = [12, 19, 28, 33, 35, 45, 46, 53, 73, 75]

    # label보다 작은 missing 개수 세기
    shift = sum(1 for m in missing if m < label)

    return label - shift #1~67

def parse_subject_and_class(folder_name):
    """
    폴더명이 '피험자_클래스' 형태라고 가정.
    ex. 19(00~19)_07(01~77)
    """
    if "_" not in folder_name:
        raise ValueError(
            f"Folder name '{folder_name}' does not contain '_'. "
            "Expected format: subject_class"
        )

    subject, class_name = folder_name.rsplit("_", 1)
    subject = subject.lstrip("0") or "0"
    class_name = class_name.lstrip("0") or "0"
    #remapped_class_label = remap_label(int(class_name))
    return subject, class_name  #0~

def load_gloss_sequence_templates(csv_path=TEMPLATE_CSV):
    """Load template_id -> class_id_sequence mapping from CSV."""
    template_map = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            template_id = row["template_id"]
            class_id_sequence = ast.literal_eval(row["class_id_sequence"])
            if not isinstance(class_id_sequence, list):
                raise ValueError(
                    f"class_id_sequence must be list, got {type(class_id_sequence)} "
                    f"for {template_id}"
                )
            template_map[template_id] = [int(x) for x in class_id_sequence]
    return template_map


def map_sequence_label(class_id, template_map=None):
    """Return gloss id sequence for a sentence template class id.

    Args:
        class_id (int | str): Class id from annotation folder name. For
            example, 12 maps to template_id ``tpl_0012``.
        template_map (dict, optional): Preloaded template mapping. If omitted,
            the CSV is read inside this function.

    Returns:
        list[int]: CTC target gloss id sequence, e.g. ``[0, 12]``.
    """
    class_id = int(class_id)
    template_id = f"tpl_{class_id:04d}"
    if template_map is None:
        template_map = load_gloss_sequence_templates()

    if template_id not in template_map:
        raise KeyError(
            f"Cannot find template_id={template_id} in {TEMPLATE_CSV}"
        )
    return template_map[template_id]


def build_annotations():
    template_map = load_gloss_sequence_templates()
    split_dirs = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()]) #train/val/test

    annotations = []
    split = {}
    for split_dir in split_dirs:
        split_name = split_dir.name
        split[split_name] = []
        sample_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        for sample_dir in sample_dirs: #each sample directory
            frame_dir = sample_dir.name #{subject_id}_{class_id}
            subject, class_name = parse_subject_and_class(frame_dir)
            class_id = int(class_name)
            gt_gloss = map_sequence_label(class_id, template_map)

            pose = load_npy(sample_dir / POSE_FILE) 
            left = load_npy(sample_dir / LEFT_HAND_FILE)
            right = load_npy(sample_dir / RIGHT_HAND_FILE)

            if pose is None:
                print(f"[SKIP] pose.npy missing: {sample_dir}")
                continue

            pose = ensure_tvc(pose, expected_v=NUM_POSE, name=f"{frame_dir}/pose") #(T, NUM_POSE, COORD_DIM)

            T = pose.shape[0]

            if left is None:
                left = make_zero_hand(T)
            else:
                left = ensure_tvc(left, expected_v=NUM_HAND, name=f"{frame_dir}/left hand") #(T, NUM_HAND, COORD_DIM)
                left = pad_or_trim_time(left, T) #pose의 프레임 개수 기준 (동적 crop 시 hand 기준으로 수정 필요)

            if right is None:
                right = make_zero_hand(T)
            else:
                right = ensure_tvc(right, expected_v=NUM_HAND, name=f"{frame_dir}/right hand") #(T, NUM_HAND, COORD_DIM)
                right = pad_or_trim_time(right, T) #pose의 프레임 개수 기준

            pose, pose_score = nan_to_zero_with_score(pose)
            left, left_score = nan_to_zero_with_score(left)
            right, right_score = nan_to_zero_with_score(right)

            # [T, 65, 3]
            keypoint = np.concatenate([pose, left, right], axis=1)
            input_keypoint = keypoint[None, ...].astype(np.float32) # [M, T, V, C]

            # [T, 65]
            keypoint_score = np.concatenate([pose_score, left_score, right_score], axis=1)
            input_keypoint_score = keypoint_score[None, ...].astype(np.float32)  # [M, T, V]

            assert keypoint.shape == (T, NUM_NODE, COORD_DIM), keypoint.shape
            assert keypoint_score.shape == (T, NUM_NODE), keypoint_score.shape

            annotations.append(
                {
                    "frame_dir": frame_dir,
                    "class_id": class_id,
                    "gt_gloss": gt_gloss,
                    "label": gt_gloss,  # CTC target sequence, e.g. [0, 12]
                    "total_frames": T,
                    "keypoint": input_keypoint, 
                    "keypoint_score": input_keypoint_score
                    # 3D skeleton에서는 필수는 아니지만, missing 정보를 보존하고 싶으면 같이 넣어둘 수 있음
                    # 단, 모델이 기본적으로 score를 feature로 쓰지는 않음
                }
            )

            split[split_name].append(frame_dir)
        print(f"{split_name} 완료")

    return annotations, split


def main():
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)

    annotations, split = build_annotations()

    ann_file = {
        "split": split,
        "annotations": annotations,
    }

    with open(OUT_PKL, "wb") as f:
        pickle.dump(ann_file, f)

    print(f"Saved annotation file to: {OUT_PKL}")
    print(f"Num samples: {len(annotations)}")

    # shape 확인용
    first = annotations[0]
    print("Example:")
    print(" frame_dir:", first["frame_dir"])
    print(" class_id:", first["class_id"])
    print(" gt_gloss:", first["gt_gloss"])
    print(" total_frames:", first["total_frames"])
    print(" keypoint shape:", first["keypoint"].shape)
    print(" keypoint_score shape:", first["keypoint_score"].shape)


if __name__ == "__main__":
    main()
