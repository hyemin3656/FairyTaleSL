
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

INPUT_NPZ_DIR = "/home/ubuntu/dataset/holistic_result_comp2"
OUTPUT_AUG_DIR = "/home/ubuntu/dataset/holistic_result_comp2_augmented_20_vari_4"
AUGMENTATIONS_PER_SAMPLE = 18
AUGMENTATIONS_PER_SAMPLE_CHOICES = [17, 18, 19]
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42

# 이 데이터셋에서는 video id/file name이 class id 역할을 합니다.
CLASS_COLUMN = "source_video"

# 아래 후보들로 가능한 조합을 만든 뒤 intensity가 낮은 것부터 높은 것까지
# 골고루 뽑아 source마다 17/18/19개 중 하나의 augmentation을 만듭니다.
GAU_NOISE = [0.0, 0.0005, 0.001, 0.002, 0.003]
PERSON_SCALE = [0.75, 0.85, 1.0, 1.15, 1.3]
DUMMY_THRESHOLD = [0.9, 0.8]
SPEED = [0.65, 0.8, 1.2, 1.5]

AUGMENTATION_ENABLE = {
    "front_offset": True,
    "dummy_threshold": True,
    "temporal_speed": True,
    "person_scale": True,
    "xy_scale": True,
    "rotation": True,
    "hand_motion_scale": True,
    "position_shift": True,
    "hand_dropout": True,
    "gaussian_noise": True,
}

AUGMENTATION_SPEEDS = [1.0, 0.65, 0.8, 1.2, 1.5]
AUGMENTATION_SHIFTS = [
    (0.0, 0.0),
    (-0.05, 0.0),
    (0.05, 0.0),
    (0.0, -0.08),
    (0.0, 0.08),
    (-0.08, -0.08),
    (0.08, -0.08),
    (-0.08, 0.12),
    (0.08, 0.12),
    (0.0, -0.16),
    (0.0, 0.16),
]
AUGMENTATION_XY_SCALES = [
    (1.0, 1.0),
    (0.9, 1.0),
    (1.1, 1.0),
    (1.0, 0.9),
    (1.0, 1.1),
    (0.9, 1.1),
    (1.1, 0.9),
]
AUGMENTATION_ROTATIONS = [0.0, -5.0, 5.0, -8.0, 8.0]
AUGMENTATION_HAND_DROPOUT_RATIOS = [0.0, 0.03, 0.06, 0.1]
AUGMENTATION_HAND_MOTION_SCALES = [0.8, 0.9, 1.0, 1.1, 1.25]
BALANCED_CANDIDATE_TRIALS = 5000


def shift_name_suffix(shift):
    shift_x, shift_y = shift
    return f"shift_x_{shift_x:+.2f}_y_{shift_y:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")



def xy_scale_name_suffix(xy_scale):
    scale_x, scale_y = xy_scale
    return f"xy_scale_x_{scale_x:.2f}_y_{scale_y:.2f}".replace(".", "_")


def rotation_name_suffix(rotation_degrees):
    return f"rot_{rotation_degrees:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")


def hand_dropout_name_suffix(ratio):
    return f"hand_drop_{ratio:.2f}".replace(".", "_")


def hand_motion_name_suffix(scale):
    return f"hand_motion_{scale:.2f}".replace(".", "_")

BASE_AUGMENTATION_SPEC = {
    "name": "base",
    "ratio": 1,
    "params": {
        "front": True,
        "dummy_threshold": 0.9,
        "window_ratio": 0.5,
        "start_ratio": 0.8,
        "end_ratio": 0.8,
        "max_gap": 10,
        "max_appear": 5,
        "person_scale": 1.0,
        "xy_scale_x": 1.0,
        "xy_scale_y": 1.0,
        "rotation_degrees": 0.0,
        "hand_dropout_ratio": 0.0,
        "hand_motion_scale": 1.0,
        "position_shift_x": 0.0,
        "position_shift_y": 0.0,
        "clip_position_shift": False,
        "gaussian_scale": 0.0,
        "speed": 1.0,
    },
}


def _rank(value, candidates):
    if len(candidates) <= 1:
        return 0.0
    return candidates.index(value) / (len(candidates) - 1)


def _max_abs_delta(candidates, center):
    return max(abs(item - center) for item in candidates) or 1.0


def _candidate_values(name, values, identity):
    return values if AUGMENTATION_ENABLE.get(name, True) else [identity]


def _shift_strength(shift):
    max_strength = max((abs(x) + abs(y)) for x, y in AUGMENTATION_SHIFTS)
    if max_strength <= 0:
        return 0.0
    return (abs(shift[0]) + abs(shift[1])) / max_strength


def _xy_scale_strength(xy_scale):
    max_strength = max((abs(x - 1.0) + abs(y - 1.0)) for x, y in AUGMENTATION_XY_SCALES)
    if max_strength <= 0:
        return 0.0
    return (abs(xy_scale[0] - 1.0) + abs(xy_scale[1] - 1.0)) / max_strength


def _candidate_intensity(params):
    components = [
        _rank(params["gaussian_scale"], GAU_NOISE),
        abs(params["person_scale"] - 1.0) / _max_abs_delta(PERSON_SCALE, 1.0),
        _rank(params["dummy_threshold"], DUMMY_THRESHOLD),
        abs(params["speed"] - 1.0) / _max_abs_delta(AUGMENTATION_SPEEDS, 1.0),
        _shift_strength((params["position_shift_x"], params["position_shift_y"])),
        _xy_scale_strength((params["xy_scale_x"], params["xy_scale_y"])),
        abs(params["rotation_degrees"]) / max(abs(item) for item in AUGMENTATION_ROTATIONS),
        params["hand_dropout_ratio"] / max(AUGMENTATION_HAND_DROPOUT_RATIOS),
        abs(params["hand_motion_scale"] - 1.0) / _max_abs_delta(AUGMENTATION_HAND_MOTION_SCALES, 1.0),
    ]
    intensity = sum(components) / len(components)
    high_count = sum(component >= 0.85 for component in components)
    low_count = sum(component <= 0.15 for component in components)
    extreme_penalty = max(0, high_count - 3) * 0.08 + max(0, low_count - 5) * 0.02
    return intensity, extreme_penalty


def _balanced_spec_name(index, params, intensity):
    return (
        f"balanced_{index:02d}_intensity_{intensity:.2f}_"
        f"gau_{params['gaussian_scale']:.4f}_"
        f"scale_{params['person_scale']:.2f}_"
        f"dummy_{params['dummy_threshold']:.1f}_"
        f"speed_{str(params['speed']).replace('.', '_')}_"
        f"{shift_name_suffix((params['position_shift_x'], params['position_shift_y']))}_"
        f"{xy_scale_name_suffix((params['xy_scale_x'], params['xy_scale_y']))}_"
        f"{rotation_name_suffix(params['rotation_degrees'])}_"
        f"{hand_dropout_name_suffix(params['hand_dropout_ratio'])}_"
        f"{hand_motion_name_suffix(params['hand_motion_scale'])}"
    )


def _sample_candidate_params(rng):
    shift_values = _candidate_values("position_shift", AUGMENTATION_SHIFTS, (0.0, 0.0))
    xy_scale_values = _candidate_values("xy_scale", AUGMENTATION_XY_SCALES, (1.0, 1.0))
    shift = shift_values[int(rng.integers(len(shift_values)))]
    xy_scale = xy_scale_values[int(rng.integers(len(xy_scale_values)))]

    front_values = [True, False] if AUGMENTATION_ENABLE.get("front_offset", True) else [True]
    gaussian_values = _candidate_values("gaussian_noise", GAU_NOISE, 0.0)
    person_scale_values = _candidate_values("person_scale", PERSON_SCALE, 1.0)
    dummy_values = _candidate_values("dummy_threshold", DUMMY_THRESHOLD, 0.9)
    speed_values = _candidate_values("temporal_speed", AUGMENTATION_SPEEDS, 1.0)
    rotation_values = _candidate_values("rotation", AUGMENTATION_ROTATIONS, 0.0)
    hand_dropout_values = _candidate_values("hand_dropout", AUGMENTATION_HAND_DROPOUT_RATIOS, 0.0)
    hand_motion_values = _candidate_values("hand_motion_scale", AUGMENTATION_HAND_MOTION_SCALES, 1.0)

    return {
        "front": front_values[int(rng.integers(len(front_values)))],
        "gaussian_scale": gaussian_values[int(rng.integers(len(gaussian_values)))],
        "person_scale": person_scale_values[int(rng.integers(len(person_scale_values)))],
        "dummy_threshold": dummy_values[int(rng.integers(len(dummy_values)))],
        "speed": speed_values[int(rng.integers(len(speed_values)))],
        "position_shift_x": float(shift[0]),
        "position_shift_y": float(shift[1]),
        "xy_scale_x": float(xy_scale[0]),
        "xy_scale_y": float(xy_scale[1]),
        "rotation_degrees": float(rotation_values[int(rng.integers(len(rotation_values)))]),
        "hand_dropout_ratio": float(hand_dropout_values[int(rng.integers(len(hand_dropout_values)))]),
        "hand_motion_scale": float(hand_motion_values[int(rng.integers(len(hand_motion_values)))]),
    }


def build_balanced_augmentation_specs(total):
    if total < 1:
        raise ValueError("total must be at least 1.")

    specs = [BASE_AUGMENTATION_SPEC]
    if total == 1:
        return specs

    selected_keys = set()
    targets = np.linspace(0.06, 0.9, total - 1)
    rng = np.random.default_rng(RANDOM_SEED)
    for index, target in enumerate(targets, start=1):
        best = None
        best_intensity = None
        best_cost = None
        for _ in range(BALANCED_CANDIDATE_TRIALS):
            params = _sample_candidate_params(rng)
            key = tuple(sorted(params.items()))
            if key in selected_keys:
                continue
            intensity, extreme_penalty = _candidate_intensity(params)
            cost = abs(intensity - float(target)) + extreme_penalty
            cost += 0.001 * ((index + int(params["front"])) % 3)
            if best_cost is None or cost < best_cost:
                best = params
                best_intensity = intensity
                best_cost = cost

        if best is None:
            break

        selected_keys.add(tuple(sorted(best.items())))
        specs.append(
            {
                "name": _balanced_spec_name(index, best, best_intensity),
                "ratio": 1,
                "params": dict(best),
            }
        )

    return specs


AUGMENTATION_SPECS = build_balanced_augmentation_specs(max(AUGMENTATIONS_PER_SAMPLE_CHOICES))


def augmentation_count_for_source(source_video):
    rng = np.random.default_rng(RANDOM_SEED + int(source_video))
    return int(rng.choice(AUGMENTATIONS_PER_SAMPLE_CHOICES))


def _allocate_counts(total, weights):
    weights = np.array(weights, dtype=np.float64)
    if total <= 0:
        raise ValueError("total must be positive.")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and at least one weight must be positive.")

    raw = weights / weights.sum() * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts.tolist()


def build_augmentation_plan(specs, total):
    counts = _allocate_counts(total, [spec.get("ratio", 1) for spec in specs])
    plan = []
    for spec, count in zip(specs, counts):
        for local_idx in range(count):
            plan.append(
                {
                    "name": spec["name"],
                    "params": dict(spec.get("params", {})),
                    "local_idx": local_idx,
                    "target_count": count,
                }
            )
    return plan


def assign_aug_splits_for_source(num_augs, split_ratios, source_order):
    """
    한 클래스/source 안에서 증강본을 train/val/test로 나눕니다.

    예: num_augs=20, split_ratios=8:1:1이면 클래스마다 16/2/2로 분할합니다.
    val/test에 들어가는 augmentation index는 source_order에 따라 회전시켜서
    전체 데이터셋 기준 파라미터 조합이 특정 split에 몰리지 않게 합니다.
    """
    split_counts = _split_counts(num_augs, split_ratios)
    split_names = list(split_ratios.keys())
    train_split = split_names[0]
    assignments = [train_split] * num_augs

    holdout_splits = split_names[1:]
    holdout_total = sum(split_counts[name] for name in holdout_splits)
    if holdout_total == 0:
        return assignments

    cursor = (source_order * holdout_total) % num_augs
    used = set()
    for split_name in holdout_splits:
        for _ in range(split_counts[split_name]):
            while cursor % num_augs in used:
                cursor += 1
            aug_idx = cursor % num_augs
            assignments[aug_idx] = split_name
            used.add(aug_idx)
            cursor += 1

    return assignments


def _split_counts(n, split_ratios):
    names = list(split_ratios.keys())
    ratios = np.array([split_ratios[name] for name in names], dtype=np.float64)
    ratios = ratios / ratios.sum()

    if n >= 10 and len(names) == 3:
        raw = ratios * n
        counts = np.floor(raw).astype(int)
        remainder = n - int(counts.sum())
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    elif n >= 3 and len(names) == 3:
        counts = np.array([n - 2, 1, 1], dtype=int)
    elif n == 2 and len(names) == 3:
        counts = np.array([1, 1, 0], dtype=int)
    else:
        counts = np.zeros(len(names), dtype=int)
        counts[0] = n

    return dict(zip(names, counts.tolist()))


def split_source_videos(video_meta_df, split_ratios, class_column=None, seed=42):
    rng = np.random.default_rng(seed)
    video_meta_df = video_meta_df.copy().reset_index(drop=True)
    video_meta_df["split"] = None

    if class_column is None or class_column not in video_meta_df.columns:
        shuffled = video_meta_df.index.to_numpy()
        rng.shuffle(shuffled)
        counts = _split_counts(len(shuffled), split_ratios)
        start = 0
        for split_name, count in counts.items():
            chosen = shuffled[start:start + count]
            video_meta_df.loc[chosen, "split"] = split_name
            start += count
        return video_meta_df

    for _, group in video_meta_df.groupby(class_column, sort=False):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        counts = _split_counts(len(indices), split_ratios)
        start = 0
        for split_name, count in counts.items():
            chosen = indices[start:start + count]
            video_meta_df.loc[chosen, "split"] = split_name
            start += count

    return video_meta_df


def merge_params(default_params, override_params):
    params = dict(default_params)
    params.update(override_params)
    return params


def augment_one_video(data, video_df, params):
    npz_30, df_30 = to_30fps(data, video_df, front=params["front"])
    npz_dropped, df_dropped = drop_dummy_frames(
        npz_30,
        df_30,
        threshold=params["dummy_threshold"],
    )
    npz_cropped, df_cropped = crop_video_by_hand_detection(
        npz_dropped,
        df_dropped,
        window_ratio=params["window_ratio"],
        start_ratio=params["start_ratio"],
        end_ratio=params["end_ratio"],
    )
    if npz_cropped is None or df_cropped is None:
        return None, None

    npz_interp, df_interp = interpolate_short_gaps(
        npz_cropped,
        df_cropped,
        max_gap=params["max_gap"],
    )
    npz_cleaned, df_cleaned = remove_short_hand_appearances(
        npz_interp,
        df_interp,
        max_appear=params["max_appear"],
    )
    npz_speed, df_speed = temporal_speed(
        npz_cleaned,
        df_cleaned,
        speed=params.get("speed", 1.0),
    )
    npz_scaled, df_scaled = person_center_scale(
        npz_speed,
        df_speed,
        scale=params["person_scale"],
        clip=params.get("clip_scale", False),
    )
    npz_xy_scaled, df_xy_scaled = person_xy_scale(
        npz_scaled,
        df_scaled,
        scale_x=params.get("xy_scale_x", 1.0),
        scale_y=params.get("xy_scale_y", 1.0),
        clip=params.get("clip_scale", False),
    )
    npz_rotated, df_rotated = rotate_keypoints(
        npz_xy_scaled,
        df_xy_scaled,
        degrees=params.get("rotation_degrees", 0.0),
        clip=params.get("clip_rotation", False),
    )
    npz_hand_scaled, df_hand_scaled = hand_motion_scale(
        npz_rotated,
        df_rotated,
        scale=params.get("hand_motion_scale", 1.0),
        clip=params.get("clip_hand_motion", False),
    )
    npz_shifted, df_shifted = position_shift(
        npz_hand_scaled,
        df_hand_scaled,
        shift_x=params.get("position_shift_x", 0.0),
        shift_y=params.get("position_shift_y", 0.0),
        clip=params.get("clip_position_shift", False),
    )
    npz_hand_dropout, df_hand_dropout = random_zero_hand_frames(
        npz_shifted,
        df_shifted,
        ratio=params.get("hand_dropout_ratio", 0.0),
    )
    npz_gaussian, df_gaussian = gaussian_noise(
        npz_hand_dropout,
        df_hand_dropout,
        scale=params["gaussian_scale"],
    )

    return npz_gaussian, df_gaussian


def save_augmented_npz(output_path, data_npz):
    np.savez_compressed(
        output_path,
        pose=data_npz["pose"],
        face=data_npz["face"],
        left_hand=data_npz["left_hand"],
        right_hand=data_npz["right_hand"],
        fps=data_npz["fps"],
    )


def to_30fps(data, video_df, front=True):
    original_fps = float(data['fps'])

    processed_data = {}

    # frame_idx 기준 정렬 권장
    video_df = video_df.sort_values('frame_idx').reset_index(drop=True)

    if np.isclose(original_fps, 59.94) or np.isclose(original_fps, 60.0):
        if front:
            # NPZ downsampling
            processed_data['pose'] = data['pose'][::2]
            processed_data['face'] = data['face'][::2]
            processed_data['left_hand'] = data['left_hand'][::2]
            processed_data['right_hand'] = data['right_hand'][::2]
            processed_data['fps'] = np.array(original_fps / 2)

            # merged_df도 같은 방식으로 downsampling
            videos_downsampled = video_df.iloc[::2].copy()
        else:
            processed_data['pose'] = data['pose'][1::2]
            processed_data['face'] = data['face'][1::2]
            processed_data['left_hand'] = data['left_hand'][1::2]
            processed_data['right_hand'] = data['right_hand'][1::2]
            processed_data['fps'] = np.array(original_fps / 2)

            # merged_df도 같은 방식으로 downsampling
            videos_downsampled = video_df.iloc[1::2].copy()

        # frame_idx를 0, 1, 2, ... 로 다시 부여
        videos_downsampled['frame_idx'] = np.arange(len(videos_downsampled))

    else:
        # 이미 30fps인 경우 그대로 저장
        processed_data['pose'] = data['pose']
        processed_data['face'] = data['face']
        processed_data['left_hand'] = data['left_hand']
        processed_data['right_hand'] = data['right_hand']
        processed_data['fps'] = np.array(original_fps)

        videos_downsampled = video_df.copy()
        videos_downsampled['frame_idx'] = np.arange(len(videos_downsampled))

    videos_downsampled = videos_downsampled.reset_index(drop=True)

    return processed_data, videos_downsampled

def drop_dummy_frames(data, video_df, threshold = 0.9):
    pose_arr = data["pose"]
    face_arr = data["face"]
    left_hand_arr = data["left_hand"]
    right_hand_arr = data["right_hand"]

    processed_data = {}
    frame_idxs = []

    for index, row in video_df.iterrows():
        frame_idx = int(row["frame_idx"])

        left_hand = row["left_hand_detected"]
        right_hand = row["right_hand_detected"]

        # 두 손 중 하나 이상 detection된 프레임만 검사
        if left_hand or right_hand:
            min_y = 10.0

            if left_hand:
                left_y = left_hand_arr[frame_idx][0][1]   # left wrist y
                if left_y < min_y:
                    min_y = left_y

            if right_hand:
                right_y = right_hand_arr[frame_idx][0][1] # right wrist y
                if right_y < min_y:
                    min_y = right_y

            # 더 위에 있는 손목의 y좌표가 0.9 이상이면 삭제
            if min_y >= threshold:
                frame_idxs.append(frame_idx)

    # df에서 해당 frame_idx 행 삭제
    filtered_video_df = video_df[
        ~video_df["frame_idx"].isin(frame_idxs)
    ].copy()

    # npz 배열에서 해당 프레임 삭제
    pose_arr = np.delete(pose_arr, frame_idxs, axis=0)
    face_arr = np.delete(face_arr, frame_idxs, axis=0)
    left_hand_arr = np.delete(left_hand_arr, frame_idxs, axis=0)
    right_hand_arr = np.delete(right_hand_arr, frame_idxs, axis=0)

    processed_data['pose'] = pose_arr
    processed_data['face'] = face_arr
    processed_data['left_hand'] = left_hand_arr
    processed_data['right_hand'] = right_hand_arr
    processed_data['fps'] = data['fps']

    # 삭제 후 frame_idx를 npz 배열 위치에 맞게 다시 0부터 재정렬
    filtered_video_df = filtered_video_df.reset_index(drop=True)
    filtered_video_df["frame_idx"] = np.arange(len(filtered_video_df))

    # 확인용 출력
    # if len(frame_idxs) > 0:
    #     frame_idxs_arr = np.array(frame_idxs)

    #     if ((frame_idxs_arr > 40) & (frame_idxs_arr < 60)).any():
    #         print(frame_idxs)

    return processed_data, filtered_video_df


def crop_video_by_hand_detection(
    data_npz,
    video_df,
    window_ratio= 0.5, #길수록 종료판단 늦음
    start_ratio = 0.8, #높을수록 더 많은 프레임에서 손이 검출되어야 시작
    end_ratio = 0.8 #높을수록 더 확실하게 손이 검출 안 되어야 종료


):
    proccessed_npz = {}
    fps = data_npz['fps']
    window = int(fps*window_ratio)
    video_name = video_df["video"].iloc[0]
    cropped_list = []
    crop_info_list = []
    video_df = video_df.sort_values('frame_idx').reset_index(drop=True)

    # left 또는 right 중 하나라도 detection 되면 True
    video_df["any_hand_detected"] = (
        video_df["left_hand_detected"].eq(True) |
        video_df["right_hand_detected"].eq(True)
    )

    detected = video_df["any_hand_detected"].to_numpy()
    frames = video_df["frame_idx"].to_numpy()

    n = len(video_df)

    start_frame = None
    start_idx = None
    end_frame = None

    # ===== 시작 시점 찾기 =====
    for i in range(0, n - window + 1):
        if detected[i:i + window].mean() >= start_ratio:
            detected_indices = np.where(detected[i:i+window])[0] + i
            start_idx = int(detected_indices[0])
            start_frame = frames[start_idx]
            break

    # ===== 종료 시점 찾기 =====
    #start idx부터
    if start_idx is not None:
        last_detected_frame = start_frame

        for i in range(start_idx, n):
            if detected[i]:
                last_detected_frame = frames[i]

            # i부터 i+window-1까지 연속 미검출이면 종료
            if i + window <= n and (~detected[i:i + window]).mean() >= end_ratio:
                detected_until_window = np.where(detected[start_idx:i + window])[0] + start_idx
                end_idx = int(detected_until_window[-1])
                end_frame = frames[end_idx]
                break

        # 끝까지 10프레임 연속 미검출이 없으면 마지막 detection 프레임까지 사용
        if end_frame is None:
            end_frame = last_detected_frame

    # 유효한 crop 구간이 있는 경우만 자르기
    if start_idx is not None:
        cropped_video_df = video_df[
            (video_df["frame_idx"] >= start_frame) &
            (video_df["frame_idx"] <= end_frame)
        ].copy()

        # crop 이후 frame_idx를 0부터 다시 부여
        df_cropped = cropped_video_df.reset_index(drop=True)
        df_cropped["frame_idx"] = np.arange(len(df_cropped))
        #npy 자르기
        pose_cropped = data_npz['pose'][start_frame:end_frame+1]
        face_cropped = data_npz['face'][start_frame:end_frame+1]
        left_hand_cropped = data_npz['left_hand'][start_frame:end_frame+1]
        right_hand_cropped = data_npz['right_hand'][start_frame:end_frame+1]

        proccessed_npz['pose'] = pose_cropped
        proccessed_npz['face'] = face_cropped
        proccessed_npz['left_hand'] = left_hand_cropped
        proccessed_npz['right_hand'] = right_hand_cropped
        proccessed_npz['fps'] = data_npz['fps']
    else:
        # 조건 만족 구간이 없는 비디오
        print(f"조건 만족 구간이 없는 비디오:", video_name)
        proccessed_npz = None
        df_cropped = None

    return proccessed_npz, df_cropped

def interpolate_short_gaps(data_npz, video_df, max_gap=10):
    """
    arr: (T, N, 4)
         마지막 차원 = x, y, z, mask_or_visibility
    frame_level_detection_series: pandas Series of booleans, length T. True if the body part was detected in the frame, False otherwise.

    mask > 0 이면 검출된 landmark
    mask == 0 이면 미검출
    """
    processed_npz = {}
    processed_video_df = video_df.copy()

    for key in ["pose", "face", "left_hand", "right_hand"]:
        arr = data_npz[key].copy()
        frame_level_detection_series = processed_video_df[f"{key}_detected"]
        T, N, C = arr.shape
        frame_detected_mask = frame_level_detection_series.to_numpy(dtype=bool)
        frame_detected_mask_ = frame_detected_mask.copy()

        assert T == len(frame_detected_mask)

        for j in range(N):  # point-level
            # Use the frame-level detection for this landmark point
            detected_for_val = arr[:, j, 3] > 0  # (T)
            detected = frame_detected_mask
            assert (detected_for_val == detected).all()

            # 모든 프레임에서 검출됐으면 보간 필요 없음
            if detected.all():
                continue

            # 모든 프레임에서 검출 안 됐으면 보간 불가능
            if not detected.any():
                continue

            for c in range(3):  # x, y, z만 보간
                s = pd.Series(arr[:, j, c])  # (T)
                s[~detected] = np.nan  # 검출 안 된 프레임의 좌표를 NaN으로 만듦

                interp = s.interpolate(
                    method="linear",
                    limit=max_gap,  # 연속된 NaN이 limit 이하일 때만 해당 구간 보간
                    limit_direction="both",  # 앞쪽/뒤쪽 양방향으로 보간을 허용
                    limit_area="inside"  # 앞뒤에 정상값이 모두 있는 내부 구간만 보간
                )

                arr[:, j, c] = interp.fillna(0).values  # 긴 구간 검출 안 된 구간은 0

            interpolated = (~detected) & ((arr[:, j, :3] != 0).any(axis=1))
            frame_detected_mask_[interpolated] = True
            arr[interpolated, j, 3] = 0.5
        processed_npz[key] = arr
        processed_video_df[f"{key}_detected"] = frame_detected_mask_

    if "fps" in data_npz:
        processed_npz["fps"] = data_npz["fps"]

    if {"left_hand_detected", "right_hand_detected"}.issubset(processed_video_df.columns):
        processed_video_df["any_hand_detected"] = (
            processed_video_df["left_hand_detected"].eq(True) |
            processed_video_df["right_hand_detected"].eq(True)
        )

    return processed_npz, processed_video_df


def remove_short_hand_appearances(data_npz, video_df, max_appear=10):
    """
    손이 max_appear 프레임 이하로만 짧게 검출된 구간을 false positive로 보고 제거합니다.

    left_hand/right_hand 각각에 대해 연속된 detected=True 구간을 찾고,
    해당 구간 길이가 max_appear 이하이면 npz 값을 0으로 만들고
    video_df의 {key}_detected 컬럼도 False로 바꿉니다.
    """
    processed_npz = {}
    processed_video_df = video_df.copy()

    for key in ["pose", "face", "left_hand", "right_hand"]:
        processed_npz[key] = data_npz[key].copy()

    for key in ["left_hand", "right_hand"]:
        detected_col = f"{key}_detected"
        arr = processed_npz[key]
        detected = processed_video_df[detected_col].to_numpy(dtype=bool)
        T = arr.shape[0]

        assert T == len(detected)

        padded = np.r_[False, detected, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        starts = changes[::2]
        ends = changes[1::2]

        remove_indices = []
        for start, end in zip(starts, ends):
            if end - start <= max_appear:
                remove_indices.extend(range(start, end))

        if remove_indices:
            remove_indices = np.array(remove_indices, dtype=int)
            arr[remove_indices] = 0
            detected_col_idx = processed_video_df.columns.get_loc(detected_col)
            processed_video_df.iloc[remove_indices, detected_col_idx] = False

        processed_npz[key] = arr

    if "fps" in data_npz:
        processed_npz["fps"] = data_npz["fps"]

    if {"left_hand_detected", "right_hand_detected"}.issubset(processed_video_df.columns):
        processed_video_df["any_hand_detected"] = (
            processed_video_df["left_hand_detected"].eq(True) |
            processed_video_df["right_hand_detected"].eq(True)
        )

    return processed_npz, processed_video_df



def _resample_array_linear(arr, new_t):
    old_t = arr.shape[0]
    if old_t == new_t:
        return arr.copy()
    if old_t == 1:
        return np.repeat(arr, new_t, axis=0).astype(arr.dtype, copy=False)

    old_x = np.arange(old_t, dtype=np.float32)
    new_x = np.linspace(0, old_t - 1, new_t, dtype=np.float32)
    out = np.empty((new_t,) + arr.shape[1:], dtype=np.float32)

    flat = arr.reshape(old_t, -1).astype(np.float32)
    out_flat = out.reshape(new_t, -1)
    for col in range(flat.shape[1]):
        out_flat[:, col] = np.interp(new_x, old_x, flat[:, col])

    return out.astype(arr.dtype, copy=False)


def _resample_video_df(video_df, source_positions, new_t):
    source_indices = np.rint(source_positions).astype(int)
    source_indices = np.clip(source_indices, 0, len(video_df) - 1)
    out_df = video_df.iloc[source_indices].copy().reset_index(drop=True)
    out_df["frame_idx"] = np.arange(new_t)
    return out_df


def _sync_detection_columns_from_npz(video_df, data_npz):
    out_df = video_df.copy()
    for key in ["pose", "face", "left_hand", "right_hand"]:
        col = f"{key}_detected"
        if col in out_df.columns:
            out_df[col] = (data_npz[key][:, :, 3] > 0).any(axis=1)

    if {"left_hand_detected", "right_hand_detected"}.issubset(out_df.columns):
        out_df["any_hand_detected"] = (
            out_df["left_hand_detected"].eq(True) |
            out_df["right_hand_detected"].eq(True)
        )
    return out_df


def temporal_speed(data_npz, video_df, speed=1.0):
    """
    speed < 1.0: 느리게. speed 비율에 따라 프레임 수를 늘리고 좌표를 선형보간합니다.
    speed > 1.0: 빠르게. speed 비율에 따라 일정 간격으로 프레임을 건너뜁니다.
    speed == 1.0: 그대로 반환합니다.
    """
    speed = float(speed)
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}.")

    processed_data = {}
    video_df = video_df.sort_values("frame_idx").reset_index(drop=True)
    old_t = int(data_npz["pose"].shape[0])
    if old_t != len(video_df):
        raise ValueError(f"data_npz T={old_t} does not match video_df length={len(video_df)}.")

    if np.isclose(speed, 1.0):
        for key in ["pose", "face", "left_hand", "right_hand"]:
            processed_data[key] = data_npz[key].copy()
        if "fps" in data_npz:
            processed_data["fps"] = data_npz["fps"]
        return processed_data, video_df.copy().reset_index(drop=True)

    new_t = max(1, int(round(old_t / speed)))

    if speed < 1.0:
        source_positions = np.linspace(0, old_t - 1, new_t, dtype=np.float32)
        for key in ["pose", "face", "left_hand", "right_hand"]:
            arr = _resample_array_linear(data_npz[key], new_t)
            if arr.shape[-1] > 3:
                arr[:, :, 3] = (arr[:, :, 3] > 0.5).astype(arr.dtype)
                arr[:, :, :3] *= arr[:, :, 3:4]
            processed_data[key] = arr
    else:
        source_indices = np.rint(np.arange(new_t, dtype=np.float32) * speed).astype(int)
        source_indices = np.clip(source_indices, 0, old_t - 1)
        source_positions = source_indices
        for key in ["pose", "face", "left_hand", "right_hand"]:
            processed_data[key] = data_npz[key][source_indices].copy()

    if "fps" in data_npz:
        processed_data["fps"] = data_npz["fps"]

    processed_video_df = _resample_video_df(video_df, source_positions, new_t)
    processed_video_df = _sync_detection_columns_from_npz(processed_video_df, processed_data)

    return processed_data, processed_video_df

def person_center_scale(
    video_npz,
    video_df,
    scale=1.1,
    left_shoulder_idx=11,
    right_shoulder_idx=12,
    default_center=(0.5, 0.5),
    clip=False,
):
    """
    사람 중심 기준으로 keypoint 좌표를 확대/축소합니다.

    기본 중심점은 pose의 양 어깨 중점입니다. 어깨가 검출되지 않은 프레임은
    해당 프레임의 valid pose keypoint 평균을 사용하고, 그것도 없으면
    default_center를 사용합니다.
    """
    augmented_data = {}
    processed_video_df = video_df.copy()

    pose = video_npz["pose"]
    T = pose.shape[0]
    centers = np.zeros((T, 2), dtype=np.float32)
    default_center_arr = np.array(default_center, dtype=np.float32)

    for t in range(T):
        pose_frame = pose[t]
        shoulder_indices_valid = (
            left_shoulder_idx < pose_frame.shape[0]
            and right_shoulder_idx < pose_frame.shape[0]
        )
        shoulder_visible = (
            shoulder_indices_valid
            and pose_frame[left_shoulder_idx, 3] > 0
            and pose_frame[right_shoulder_idx, 3] > 0
        )

        if shoulder_visible:
            centers[t] = (
                pose_frame[left_shoulder_idx, :2]
                + pose_frame[right_shoulder_idx, :2]
            ) * 0.5
            continue

        valid_pose = pose_frame[pose_frame[:, 3] > 0]
        if valid_pose.size > 0:
            centers[t] = valid_pose[:, :2].mean(axis=0)
        else:
            centers[t] = default_center_arr

    for key in ["pose", "face", "left_hand", "right_hand"]:
        arr = video_npz[key].astype(np.float32).copy()
        valid_mask = arr[:, :, 3] > 0

        arr[:, :, :2] = centers[:, None, :] + (arr[:, :, :2] - centers[:, None, :]) * scale

        if clip:
            arr[:, :, 0] = np.clip(arr[:, :, 0], 0.0, 1.0)
            arr[:, :, 1] = np.clip(arr[:, :, 1], 0.0, 1.2)

        arr[:, :, :2] *= valid_mask[:, :, None]
        augmented_data[key] = arr

    if "fps" in video_npz:
        augmented_data["fps"] = video_npz["fps"]

    return augmented_data, processed_video_df


def _pose_frame_centers(
    video_npz,
    left_shoulder_idx=11,
    right_shoulder_idx=12,
    default_center=(0.5, 0.5),
):
    pose = video_npz["pose"]
    centers = np.zeros((pose.shape[0], 2), dtype=np.float32)
    default_center_arr = np.array(default_center, dtype=np.float32)

    for t, pose_frame in enumerate(pose):
        shoulder_visible = (
            left_shoulder_idx < pose_frame.shape[0]
            and right_shoulder_idx < pose_frame.shape[0]
            and pose_frame[left_shoulder_idx, 3] > 0
            and pose_frame[right_shoulder_idx, 3] > 0
        )
        if shoulder_visible:
            centers[t] = (
                pose_frame[left_shoulder_idx, :2]
                + pose_frame[right_shoulder_idx, :2]
            ) * 0.5
            continue

        valid_pose = pose_frame[pose_frame[:, 3] > 0]
        centers[t] = valid_pose[:, :2].mean(axis=0) if valid_pose.size > 0 else default_center_arr

    return centers


def person_xy_scale(video_npz, video_df, scale_x=1.0, scale_y=1.0, clip=False):
    scale = np.array([float(scale_x), float(scale_y)], dtype=np.float32)
    centers = _pose_frame_centers(video_npz)
    augmented_data = {}

    for key in ["pose", "face", "left_hand", "right_hand"]:
        arr = video_npz[key].astype(np.float32).copy()
        valid_mask = arr[:, :, 3] > 0
        arr[:, :, :2] = centers[:, None, :] + (arr[:, :, :2] - centers[:, None, :]) * scale
        if clip:
            arr[:, :, 0] = np.clip(arr[:, :, 0], 0.0, 1.0)
            arr[:, :, 1] = np.clip(arr[:, :, 1], 0.0, 1.0)
        arr[:, :, :2] *= valid_mask[:, :, None]
        augmented_data[key] = arr

    if "fps" in video_npz:
        augmented_data["fps"] = video_npz["fps"]
    return augmented_data, video_df


def rotate_keypoints(video_npz, video_df, degrees=0.0, clip=False):
    degrees = float(degrees)
    if np.isclose(degrees, 0.0):
        out = {key: video_npz[key].copy() for key in ["pose", "face", "left_hand", "right_hand"]}
        if "fps" in video_npz:
            out["fps"] = video_npz["fps"]
        return out, video_df

    centers = _pose_frame_centers(video_npz)
    rad = np.deg2rad(degrees)
    rot = np.array(
        [[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]],
        dtype=np.float32,
    )
    augmented_data = {}

    for key in ["pose", "face", "left_hand", "right_hand"]:
        arr = video_npz[key].astype(np.float32).copy()
        valid_mask = arr[:, :, 3] > 0
        centered = arr[:, :, :2] - centers[:, None, :]
        arr[:, :, :2] = centers[:, None, :] + centered @ rot.T
        if clip:
            arr[:, :, 0] = np.clip(arr[:, :, 0], 0.0, 1.0)
            arr[:, :, 1] = np.clip(arr[:, :, 1], 0.0, 1.0)
        arr[:, :, :2] *= valid_mask[:, :, None]
        augmented_data[key] = arr

    if "fps" in video_npz:
        augmented_data["fps"] = video_npz["fps"]
    return augmented_data, video_df


def hand_motion_scale(video_npz, video_df, scale=1.0, clip=False):
    scale = float(scale)
    centers = _pose_frame_centers(video_npz)
    augmented_data = {}

    for key in ["pose", "face", "left_hand", "right_hand"]:
        arr = video_npz[key].astype(np.float32).copy()
        if key in ("left_hand", "right_hand"):
            valid_mask = arr[:, :, 3] > 0
            arr[:, :, :2] = centers[:, None, :] + (arr[:, :, :2] - centers[:, None, :]) * scale
            if clip:
                arr[:, :, 0] = np.clip(arr[:, :, 0], 0.0, 1.0)
                arr[:, :, 1] = np.clip(arr[:, :, 1], 0.0, 1.0)
            arr[:, :, :2] *= valid_mask[:, :, None]
        augmented_data[key] = arr

    if "fps" in video_npz:
        augmented_data["fps"] = video_npz["fps"]
    return augmented_data, video_df


def random_zero_hand_frames(video_npz, video_df, ratio=0.0, min_keep_frames=1):
    ratio = float(ratio)
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"ratio must be in [0.0, 1.0), got {ratio}.")

    augmented_data = {key: video_npz[key].astype(np.float32).copy() for key in ["pose", "face", "left_hand", "right_hand"]}
    total_frames = int(video_npz["pose"].shape[0])
    zero_count = int(round(total_frames * ratio))
    max_zero_count = max(0, total_frames - int(min_keep_frames))
    zero_count = min(zero_count, max_zero_count)

    if zero_count > 0:
        for key in ["left_hand", "right_hand"]:
            zero_indices = np.random.choice(total_frames, size=zero_count, replace=False)
            augmented_data[key][zero_indices] = 0.0

    if "fps" in video_npz:
        augmented_data["fps"] = video_npz["fps"]

    processed_video_df = _sync_detection_columns_from_npz(video_df, augmented_data)
    return augmented_data, processed_video_df


def position_shift(video_npz, video_df, shift_x=0.0, shift_y=0.0, clip=False):
    """
    모든 유효 keypoint의 x/y 좌표를 지정한 값만큼 이동합니다.

    좌표가 0~1 스케일인 데이터를 기준으로 shift_x=0.05는 화면 너비의 5%,
    shift_y=-0.03은 화면 높이의 3%만큼 위로 이동하는 의미입니다.
    """
    shift_x = float(shift_x)
    shift_y = float(shift_y)
    augmented_data = {}

    for key in ["pose", "face", "left_hand", "right_hand"]:
        arr = video_npz[key].astype(np.float32).copy()
        valid_mask = arr[:, :, 3] > 0

        arr[:, :, 0] += shift_x * valid_mask
        arr[:, :, 1] += shift_y * valid_mask

        if clip:
            arr[:, :, 0] = np.clip(arr[:, :, 0], 0.0, 1.0)
            arr[:, :, 1] = np.clip(arr[:, :, 1], 0.0, 1.0)

        arr[:, :, :2] *= valid_mask[:, :, None]
        augmented_data[key] = arr

    if "fps" in video_npz:
        augmented_data["fps"] = video_npz["fps"]

    return augmented_data, video_df



def gaussian_noise(video_npz, video_df, scale=0.001):
    augmented_data = {}

    for key in ['pose', 'face', 'left_hand', 'right_hand']:
        original_array = video_npz[key].astype(np.float32)
        augmented_array = original_array.copy()

        # x, y에만 noise 추가. 같은 keypoint는 모든 프레임에서 같은 noise를 사용합니다.
        noise = np.random.normal(
            loc=0.0,
            scale=scale,
            size=augmented_array[:, :, :2].shape[1:]
        ).astype(np.float32)

        # visibility/mask가 있는 경우: 마지막 차원이 0보다 큰 keypoint에만 noise 적용
        valid_mask = augmented_array[:, :, 3] > 0

        augmented_array[:, :, :2] += noise[None, :, :] * valid_mask[:, :, None]

        augmented_data[key] = augmented_array
    
    augmented_data['fps'] = video_npz['fps']

    return augmented_data, video_df



if __name__ == "__main__":
    input_dir = Path(INPUT_NPZ_DIR)
    output_dir = Path(OUTPUT_AUG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_RATIOS:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    npz_files = sorted([f for f in os.listdir(INPUT_NPZ_DIR) if f.endswith(".npz")])
    merged_df = pd.read_csv(input_dir / "all_summary.csv")

    default_params = {}
    for spec in AUGMENTATION_SPECS:
        if spec["name"] == "base":
            default_params = dict(spec["params"])
            break
    if not default_params:
        raise ValueError("AUGMENTATION_SPECS must include a 'base' spec with default params.")

    source_rows = []
    for npz_file in npz_files:
        video_name = os.path.splitext(npz_file)[0]
        video_id = int(video_name)
        video_df = merged_df[merged_df["video"] == video_id]
        if video_df.empty:
            print(f"Skip {npz_file}: no rows in all_summary.csv")
            continue
        source_rows.append(
            {
                "source_video": video_id,
                "class_id": video_id,
                "npz_file": npz_file,
            }
        )

    source_meta_df = pd.DataFrame(source_rows)
    if source_meta_df.empty:
        raise ValueError("No source videos found to augment.")
    source_meta_df.to_csv(output_dir / "source_videos.csv", index=False)

    aug_count_by_source = {
        int(row["source_video"]): augmentation_count_for_source(int(row["source_video"]))
        for _, row in source_meta_df.iterrows()
    }
    split_count_examples = {
        count: _split_counts(count, SPLIT_RATIOS)
        for count in sorted(set(aug_count_by_source.values()))
    }
    manifest_rows = []
    summary_rows = []
    split_summary_rows = {split_name: [] for split_name in SPLIT_RATIOS}
    skipped = []

    print(f"Classes/source videos: {len(source_meta_df)}")
    print(f"Augmentations per class choices: {AUGMENTATIONS_PER_SAMPLE_CHOICES}")
    print(f"Augmentation count distribution: {pd.Series(aug_count_by_source).value_counts().sort_index().to_dict()}")
    print(f"Split count examples: {split_count_examples}")

    for source_order, source_row in tqdm(
        list(source_meta_df.iterrows()),
        total=len(source_meta_df),
    ):
        source_video = int(source_row["source_video"])
        class_id = int(source_row["class_id"])
        npz_file = source_row["npz_file"]
        aug_count = aug_count_by_source[source_video]
        aug_plan = build_augmentation_plan(AUGMENTATION_SPECS, aug_count)
        aug_splits = assign_aug_splits_for_source(
            len(aug_plan),
            SPLIT_RATIOS,
            source_order,
        )

        input_npz_path = input_dir / npz_file
        data = np.load(input_npz_path, allow_pickle=True)
        video_df = merged_df[merged_df["video"] == source_video].copy()

        for aug_idx, aug_spec in enumerate(aug_plan):
            split_name = aug_splits[aug_idx]
            params = merge_params(default_params, aug_spec["params"])
            np.random.seed(RANDOM_SEED + source_video * 1000 + aug_idx)

            try:
                aug_npz, aug_df = augment_one_video(data, video_df, params)
            except Exception as exc:
                skipped.append(
                    {
                        "source_video": source_video,
                        "class_id": class_id,
                        "npz_file": npz_file,
                        "split": split_name,
                        "aug_idx": aug_idx,
                        "aug_name": aug_spec["name"],
                        "error": repr(exc),
                    }
                )
                continue

            if aug_npz is None or aug_df is None:
                skipped.append(
                    {
                        "source_video": source_video,
                        "class_id": class_id,
                        "npz_file": npz_file,
                        "split": split_name,
                        "aug_idx": aug_idx,
                        "aug_name": aug_spec["name"],
                        "error": "no valid crop interval",
                    }
                )
                continue

            subject_id = aug_idx
            sample_id = f"{subject_id}_{class_id}"
            output_npz_path = output_dir / split_name / f"{sample_id}.npz"
            save_augmented_npz(output_npz_path, aug_npz)

            aug_df = aug_df.copy()
            aug_df["video"] = sample_id
            aug_df["subject_id"] = subject_id
            aug_df["source_video"] = source_video
            aug_df["class_id"] = class_id
            aug_df["split"] = split_name
            aug_df["aug_idx"] = aug_idx
            aug_df["aug_name"] = aug_spec["name"]

            summary_rows.append(aug_df)
            split_summary_rows[split_name].append(aug_df)

            manifest_rows.append(
                {
                    "video": sample_id,
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "file": str(output_npz_path.relative_to(output_dir)),
                    "split": split_name,
                    "source_video": source_video,
                    "class_id": class_id,
                    "source_file": npz_file,
                    "aug_idx": aug_idx,
                    "aug_name": aug_spec["name"],
                    "aug_count_for_class": aug_count,
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "num_frames": int(aug_npz["pose"].shape[0]),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)

    if summary_rows:
        all_summary_df = pd.concat(summary_rows, ignore_index=True)
        all_summary_df.to_csv(output_dir / "all_summary.csv", index=False)

    for split_name, rows in split_summary_rows.items():
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(
                output_dir / split_name / "all_summary.csv",
                index=False,
            )

    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped.csv", index=False)
        print(f"Skipped augmentations: {len(skipped)}. See {output_dir / 'skipped.csv'}")

    print(f"Saved augmented dataset: {output_dir}")
    print("Augmented split counts:")
    if not manifest_df.empty:
        print(manifest_df["split"].value_counts().reindex(SPLIT_RATIOS.keys(), fill_value=0))
        print("Class counts by split:")
        print(manifest_df.groupby("split")["class_id"].nunique().reindex(SPLIT_RATIOS.keys(), fill_value=0))
        print("Augmentation counts by split:")
        print(pd.crosstab(manifest_df["aug_name"], manifest_df["split"]))
