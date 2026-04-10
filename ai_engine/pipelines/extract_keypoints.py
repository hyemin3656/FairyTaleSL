"""
KSL 데이터셋 키포인트 추출 스크립트 (MediaPipe Tasks API 0.10+)

입력:  KSL_rgb/{person}_{gloss}/*.jpg
출력:  data/keypoints/{label_idx:03d}/{person_id}.npy  — shape (T, 225)

키포인트 구성 (1 프레임 = 225 값):
  left_hand  : 21 × 3 = 63   (x, y, z)
  right_hand : 21 × 3 = 63
  pose       : 33 × 3 = 99
  합계       : 225

실행:
  python pipelines/extract_keypoints.py \
      --rgb_dir /path/to/KSL_rgb \
      --action_dir /path/to/KSL_ACTION_VIDEO \
      --out_dir ./data/keypoints \
      --hand_model ./models/mediapipe_tasks/hand_landmarker.task \
      --pose_model ./models/mediapipe_tasks/pose_landmarker_full.task
"""
import argparse
import os
import re
import numpy as np
import cv2
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HAND_N = 21
POSE_N = 33
FRAME_DIM = (HAND_N * 2 + POSE_N) * 3  # 225


def make_hand_landmarker(model_path: str) -> mp_vision.HandLandmarker:
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def make_pose_landmarker(model_path: str) -> mp_vision.PoseLandmarker:
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_frame_keypoints(
    frame_bgr: np.ndarray,
    hand_det: mp_vision.HandLandmarker,
    pose_det: mp_vision.PoseLandmarker,
) -> np.ndarray:
    """
    단일 BGR 프레임 → 키포인트 벡터 (225,)
    순서: left_hand(63) | right_hand(63) | pose(99)
    감지 실패 시 0으로 채움.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # ── 손 ─────────────────────────────────────────────────────────────────────
    left_hand  = np.zeros(HAND_N * 3, np.float32)
    right_hand = np.zeros(HAND_N * 3, np.float32)

    hand_result = hand_det.detect(mp_img)
    if hand_result.hand_landmarks:
        for lm_list, handedness_list in zip(
            hand_result.hand_landmarks,
            hand_result.handedness,
        ):
            label = handedness_list[0].category_name  # "Left" or "Right"
            pts = np.array([[p.x, p.y, p.z] for p in lm_list], np.float32).flatten()
            if label == "Left":
                left_hand = pts
            else:
                right_hand = pts

    # ── 자세 ───────────────────────────────────────────────────────────────────
    pose_kp = np.zeros(POSE_N * 3, np.float32)
    pose_result = pose_det.detect(mp_img)
    if pose_result.pose_landmarks:
        pose_kp = np.array(
            [[p.x, p.y, p.z] for p in pose_result.pose_landmarks[0]],
            np.float32,
        ).flatten()

    return np.concatenate([left_hand, right_hand, pose_kp])  # (225,)


def process_sequence(
    folder_path: str,
    out_path: str,
    hand_model_path: str,
    pose_model_path: str,
) -> tuple[str, int, str]:
    """단일 시퀀스 폴더 처리. (folder_name, n_frames, status) 반환"""
    folder_name = os.path.basename(folder_path)

    if os.path.exists(out_path):
        n = np.load(out_path).shape[0]
        return folder_name, n, "skip"

    jpg_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not jpg_files:
        return folder_name, 0, "empty"

    hand_det = make_hand_landmarker(hand_model_path)
    pose_det = make_pose_landmarker(pose_model_path)

    sequence = []
    for jpg in jpg_files:
        frame = cv2.imread(os.path.join(folder_path, jpg))
        if frame is None:
            continue
        kp = extract_frame_keypoints(frame, hand_det, pose_det)
        sequence.append(kp)

    hand_det.close()
    pose_det.close()

    if not sequence:
        return folder_name, 0, "no_frames"

    arr = np.array(sequence, np.float32)  # (T, 225)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)
    return folder_name, len(sequence), "ok"


def build_label_map(action_video_dir: str) -> dict[str, int]:
    gloss_ids = sorted(
        d for d in os.listdir(action_video_dir)
        if os.path.isdir(os.path.join(action_video_dir, d))
    )
    return {gid: idx for idx, gid in enumerate(gloss_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_dir",    required=True)
    parser.add_argument("--action_dir", required=False)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--hand_model", default="models/mediapipe_tasks/hand_landmarker.task")
    parser.add_argument("--pose_model", default="models/mediapipe_tasks/pose_landmarker_full.task")
    args = parser.parse_args()

    action_dir = args.action_dir or os.path.join(os.path.dirname(args.rgb_dir), "KSL_ACTION_VIDEO")

    # 라벨 맵 저장
    label_map = build_label_map(action_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    label_map_path = os.path.join(args.out_dir, "label_map.txt")
    with open(label_map_path, "w", encoding="utf-8") as f:
        for gid, idx in label_map.items():
            f.write(f"{idx}\t{gid}\n")
    print(f"[라벨 맵] {len(label_map)}개 클래스 → {label_map_path}")

    # 처리 폴더 목록
    pattern = re.compile(r"^(\d{2})_(\d+)$")
    folders = []
    for d in sorted(os.listdir(args.rgb_dir)):
        full = os.path.join(args.rgb_dir, d)
        if not os.path.isdir(full):
            continue
        m = pattern.match(d)
        if not m:
            continue
        person_id, gloss_id = m.group(1), m.group(2)
        if gloss_id not in label_map:
            continue
        label_idx = label_map[gloss_id]
        out_path = os.path.join(args.out_dir, f"{label_idx:03d}", f"{person_id}.npy")
        folders.append((full, out_path))

    print(f"[대상] {len(folders)}개 시퀀스 (총 {len(label_map)}클래스)")

    ok = skip = err = 0
    with tqdm(folders, desc="추출 중") as bar:
        for fp, op in bar:
            _, n_frames, status = process_sequence(fp, op, args.hand_model, args.pose_model)
            if status == "ok":     ok   += 1
            elif status == "skip": skip += 1
            else:                  err  += 1
            bar.set_postfix(ok=ok, skip=skip, err=err)

    print(f"\n완료: ok={ok}, skip={skip}, err={err}")
    print(f"출력: {args.out_dir}/")


if __name__ == "__main__":
    main()
