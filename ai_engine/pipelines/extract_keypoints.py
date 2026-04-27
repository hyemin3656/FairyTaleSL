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

# BlazePose 33-landmark indices
LEFT_WRIST_IDX  = 15
RIGHT_WRIST_IDX = 16


def make_hand_landmarker(model_path: str) -> mp_vision.HandLandmarker:
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def make_hand_landmarker_roi(model_path: str) -> mp_vision.HandLandmarker:
    """ROI crop용 단일-손 IMAGE 모드 detector. 더 관대한 임계값."""
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def make_pose_landmarker(model_path: str) -> mp_vision.PoseLandmarker:
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def detect_hand_in_roi(
    frame_bgr: np.ndarray,
    hand_det_roi: mp_vision.HandLandmarker,
    wrist_x_norm: float,
    wrist_y_norm: float,
    crop_size: int,
) -> np.ndarray | None:
    """손목 좌표 주변 crop_size × crop_size crop으로 손 재검출.
    성공 시 full-image 정규화 좌표 (HAND_N*3,) 반환, 실패 시 None.
    """
    H, W = frame_bgr.shape[:2]
    cx = int(round(wrist_x_norm * W))
    cy = int(round(wrist_y_norm * H))
    half = crop_size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(W, cx + half)
    y1 = min(H, cy + half)
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return None

    crop_bgr = frame_bgr[y0:y1, x0:x1]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
    res = hand_det_roi.detect(mp_img)
    if not res.hand_landmarks:
        return None

    cw, ch = (x1 - x0), (y1 - y0)
    pts = np.array(
        [[(p.x * cw + x0) / W, (p.y * ch + y0) / H, p.z] for p in res.hand_landmarks[0]],
        np.float32,
    ).flatten()
    return pts


def extract_frame_keypoints(
    frame_bgr: np.ndarray,
    hand_det: mp_vision.HandLandmarker,
    pose_det: mp_vision.PoseLandmarker,
    hand_det_roi: mp_vision.HandLandmarker | None,
    timestamp_ms: int,
    roi_crop_size: int = 256,
) -> tuple[np.ndarray, int]:
    """
    단일 BGR 프레임 → (키포인트 (225,), ROI로 구한 손 개수)
    순서: left_hand(63) | right_hand(63) | pose(99)
    감지 실패 시 0으로 채움.
    timestamp_ms는 시퀀스 내에서 단조 증가해야 함 (VIDEO 모드).
    hand_det_roi가 주어지면 1차 검출 실패 손에 대해 pose 손목 ROI crop 재검출.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # ── 자세 (먼저: ROI fallback에 손목 좌표 필요) ────────────────────────────
    pose_kp = np.zeros(POSE_N * 3, np.float32)
    pose_landmarks = None
    pose_result = pose_det.detect_for_video(mp_img, timestamp_ms)
    if pose_result.pose_landmarks:
        pose_landmarks = pose_result.pose_landmarks[0]
        pose_kp = np.array(
            [[p.x, p.y, p.z] for p in pose_landmarks], np.float32
        ).flatten()

    # ── 1차 손 검출 (전체 이미지) ─────────────────────────────────────────────
    left_hand  = np.zeros(HAND_N * 3, np.float32)
    right_hand = np.zeros(HAND_N * 3, np.float32)
    left_found = right_found = False

    hand_result = hand_det.detect_for_video(mp_img, timestamp_ms)
    if hand_result.hand_landmarks:
        for lm_list, handedness_list in zip(
            hand_result.hand_landmarks,
            hand_result.handedness,
        ):
            label = handedness_list[0].category_name  # "Left" or "Right"
            pts = np.array([[p.x, p.y, p.z] for p in lm_list], np.float32).flatten()
            if label == "Left":
                left_hand = pts
                left_found = True
            else:
                right_hand = pts
                right_found = True

    # ── 2차 손 검출 (Pose 손목 ROI fallback) ─────────────────────────────────
    rescued = 0
    if hand_det_roi is not None and pose_landmarks is not None:
        if not left_found:
            wrist = pose_landmarks[LEFT_WRIST_IDX]
            pts = detect_hand_in_roi(frame_bgr, hand_det_roi, wrist.x, wrist.y, roi_crop_size)
            if pts is not None:
                left_hand = pts
                rescued += 1
        if not right_found:
            wrist = pose_landmarks[RIGHT_WRIST_IDX]
            pts = detect_hand_in_roi(frame_bgr, hand_det_roi, wrist.x, wrist.y, roi_crop_size)
            if pts is not None:
                right_hand = pts
                rescued += 1

    return np.concatenate([left_hand, right_hand, pose_kp]), rescued


def process_sequence(
    folder_path: str,
    out_path: str,
    hand_model_path: str,
    pose_model_path: str,
    fps: float = 30.0,
    use_roi: bool = True,
    roi_crop_size: int = 256,
) -> tuple[str, int, str, int]:
    """단일 시퀀스 폴더 처리. (folder_name, n_frames, status, rescued_total) 반환"""
    folder_name = os.path.basename(folder_path)

    if os.path.exists(out_path):
        n = np.load(out_path).shape[0]
        return folder_name, n, "skip", 0

    jpg_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not jpg_files:
        return folder_name, 0, "empty", 0

    hand_det     = make_hand_landmarker(hand_model_path)
    pose_det     = make_pose_landmarker(pose_model_path)
    hand_det_roi = make_hand_landmarker_roi(hand_model_path) if use_roi else None

    frame_dt_ms = max(1, int(round(1000.0 / fps)))
    sequence = []
    rescued_total = 0
    for idx, jpg in enumerate(jpg_files):
        frame = cv2.imread(os.path.join(folder_path, jpg))
        if frame is None:
            continue
        timestamp_ms = idx * frame_dt_ms
        kp, rescued = extract_frame_keypoints(
            frame, hand_det, pose_det, hand_det_roi, timestamp_ms, roi_crop_size
        )
        sequence.append(kp)
        rescued_total += rescued

    hand_det.close()
    pose_det.close()
    if hand_det_roi is not None:
        hand_det_roi.close()

    if not sequence:
        return folder_name, 0, "no_frames", 0

    arr = np.array(sequence, np.float32)  # (T, 225)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)
    return folder_name, len(sequence), "ok", rescued_total


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
    parser.add_argument("--hand_model", default="ai_engine/models/mediapipe_tasks/hand_landmarker.task")
    parser.add_argument("--pose_model", default="ai_engine/models/mediapipe_tasks/pose_landmarker_full.task")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="원본 영상 FPS (VIDEO 모드 timestamp 계산용)")
    parser.add_argument("--no_roi", action="store_true",
                        help="Pose 손목 ROI fallback 비활성화")
    parser.add_argument("--roi_crop_size", type=int, default=256,
                        help="ROI crop 한 변 픽셀 (기본 256)")
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
    rescued_grand = 0
    with tqdm(folders, desc="추출 중") as bar:
        for fp, op in bar:
            _, n_frames, status, rescued = process_sequence(
                fp, op, args.hand_model, args.pose_model,
                args.fps, use_roi=(not args.no_roi), roi_crop_size=args.roi_crop_size,
            )
            if status == "ok":     ok   += 1
            elif status == "skip": skip += 1
            else:                  err  += 1
            rescued_grand += rescued
            bar.set_postfix(ok=ok, skip=skip, err=err, rescued=rescued_grand)

    print(f"\n완료: ok={ok}, skip={skip}, err={err}")
    print(f"ROI fallback으로 추가 검출된 손 인스턴스: {rescued_grand}")
    print(f"출력: {args.out_dir}/")


if __name__ == "__main__":
    main()
