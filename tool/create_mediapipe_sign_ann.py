import os
import json
import pickle
from pathlib import Path
import csv
import ast
import numpy as np

ROOT_DIR = Path("../dataset/gloss_sequences")
OUT_PKL = Path("../dataset/gloss_sequences/mediapipe_sign_3d_without_face_pose_score_1.pkl")
TEMPLATE_CSV =  Path("../dataset/gloss_sequences/gloss_sequence_templates.csv")

RANDOM_SEED = 42

# 설정
NUM_POSE = 23       # pose 0~22
NUM_FACE = 468
NUM_HAND = 21
NUM_NODE = 65 #533       # 23 + 21 + 21 + 468
NUM_PERSON = 1
COORD_DIM = 3       # x, y, z

def load_gloss_sequence_templates(csv_path=TEMPLATE_CSV):
    """Load template_id -> class_id_sequence mapping from CSV."""
    last_error = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        template_map = {}
        try:
            with open(csv_path, "r", encoding=encoding, newline="") as f:
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
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"Could not decode {csv_path} with utf-8-sig, cp949, or euc-kr",
    )

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

    return arr[:, :expected_v,].astype(np.float32)


# def make_zero_hand(T):
#     return np.zeros((T, NUM_HAND, COORD_DIM), dtype=np.float32)


# def pad_or_trim_time(arr, target_T):
#     """
#     같은 sample내의 각 npy의 T가 다를 경우 target_T에 맞춘다.
#     짧으면 0 padding, 길면 자른다.
#     """
#     T, V, C = arr.shape

#     if T == target_T:
#         return arr

#     if T > target_T:
#         return arr[:target_T]

#     padded = np.zeros((target_T, V, C), dtype=arr.dtype)
#     padded[:T] = arr
#     return padded


# def nan_to_zero_with_score(arr):
#     """
#     NaN이 있는 keypoint는 좌표 0, score 0 으로 설정
#     정상 keypoint는 score 1.
#     arr: #(T, NUM_POSE or NUM_HAND, COORD_DIM)
#     """
#     invalid = np.isnan(arr).any(axis=-1)  # [T, NUM_POSE] (keypoint-level)
#     score = (~invalid).astype(np.float32)

#     arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
#     arr[invalid] = 0.0

#     return arr.astype(np.float32), score.astype(np.float32)

# def remap_label(label):
#     #비어있는 클래스 제거한 클래스 label 반환
#     #label : 1~77
#     missing = [12, 19, 28, 33, 35, 45, 46, 53, 73, 75]

#     # label보다 작은 missing 개수 세기
#     shift = sum(1 for m in missing if m < label)

#     return label - shift #1~67

def parse_subject_and_class(folder_name):
    """
    폴더명이 '피험자_클래스' 형태라고 가정.
    ex. 19(00~19)_07(00~66)
    """
    if "_" not in folder_name:
        raise ValueError(
            f"Folder name '{folder_name}' does not contain '_'. "
            "Expected format: subject_class"
        )

    subject, class_name = folder_name.rsplit("_", 1)
    subject = subject.lstrip("0") or "0"
    class_name = class_name.lstrip("0") or "0"
    return subject, int(class_name)  #0~66


def build_annotations():
    template_map = load_gloss_sequence_templates()
    split_dirs = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()]) #train/val/test

    annotations = []
    split = {}
    frame_lengths = []   

    for split_dir in split_dirs:
        split_name = split_dir.name
        split[split_name] = []
        for npz_path in sorted(split_dir.rglob("*.npz")):
            file_name = npz_path.stem      # 00_00
            subject, class_name = parse_subject_and_class(file_name)
            class_id = int(class_name)
            gt_gloss = map_sequence_label(class_id, template_map)

            npz = np.load(npz_path)
            pose = npz["pose"]
            face = npz["face"]
            left_hand = npz["left_hand"]
            right_hand = npz["right_hand"]

            pose = ensure_tvc(pose, expected_v=NUM_POSE, name=f"{file_name}[pose]") #(T, NUM_POSE, 4)
            face = ensure_tvc(face, expected_v=NUM_FACE, name=f"{file_name}[face]") #(T, NUM_FACE, 4)
            left = ensure_tvc(left_hand, expected_v=NUM_HAND, name=f"{file_name}[left hand]") #(T, NUM_HAND, 4)
            right = ensure_tvc(right_hand, expected_v=NUM_HAND, name=f"{file_name}/[right hand]") #(T, NUM_HAND, 4)
            T = pose.shape[0]
            frame_lengths.append(T)   # 추가

            arrays = {
                "pose": pose,
                "face": face,
                "left_hand": left,
                "right_hand": right,
            }
            # 1. NaN 있으면 에러
            for name, arr in arrays.items():
                if np.isnan(arr).any():
                    raise ValueError(f"{file_name}[{name}] contains NaN")

            # 2. T가 모두 같지 않으면 에러
            T_values = {name: arr.shape[0] for name, arr in arrays.items()}

            if len(set(T_values.values())) != 1:
                raise ValueError(
                    f"{file_name} has inconsistent T: "
                    + ", ".join([f"{name}={T}" for name, T in T_values.items()]))

            keypoints = {}
            scores = {}

            for name, arr in arrays.items():
                keypoints[name] = arr[..., :-1]  # (T, V, C-1)
                scores[name] = arr[..., -1]      # (T, V)
            
            scores["pose"] = np.ones(scores["pose"].shape, dtype=np.float32)

            # [T, 533, 3]
            keypoint = np.concatenate([keypoints['pose'], keypoints['left_hand'], keypoints['right_hand']], axis=1) # keypoints['face']
            input_keypoint = keypoint[None, ...].astype(np.float32) # [M, T, V, C]

            # [T, 533]
            keypoint_score = np.concatenate([scores['pose'], scores['left_hand'], scores['right_hand']], axis=1) # scores['face']
            input_keypoint_score = keypoint_score[None, ...].astype(np.float32)  # [M, T, V]

            assert keypoint.shape == (T, NUM_NODE, COORD_DIM), keypoint.shape
            assert keypoint_score.shape == (T, NUM_NODE), keypoint_score.shape

            annotations.append(
                {
                    "frame_dir": file_name,
                    "template_id": class_name, 
                    "label": gt_gloss,
                    "total_frames": T,
                    "keypoint": input_keypoint, 
                    "keypoint_score": input_keypoint_score
                }
            )

            split[split_name].append(file_name)
        print(f"{split_name} 완료")

    return annotations, split, frame_lengths


def main():
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)

    annotations, split, frame_lengths = build_annotations()

    ann_file = {
        "split": split,
        "annotations": annotations,
    }

    with open(OUT_PKL, "wb") as f:
        pickle.dump(ann_file, f)

    print(f"Saved annotation file to: {OUT_PKL}")
    print(f"Num samples: {len(annotations)}")
    # frame 통계
    print(f"Average frames: {np.mean(frame_lengths):.2f}")
    print(f"Min frames: {np.min(frame_lengths)}")
    print(f"Max frames: {np.max(frame_lengths)}")
    print(f"Median frames: {np.median(frame_lengths):.2f}")
    # shape 확인용
    first = annotations[6]
    print("Example:")
    print(" frame_dir:", first["frame_dir"])
    print(" template_id:", first["template_id"])
    print(" label:", first["label"])
    print(" total_frames:", first["total_frames"])
    print(" keypoint shape:", first["keypoint"].shape)
    print(" keypoint_score shape:", first["keypoint_score"].shape)


if __name__ == "__main__":
    main()
