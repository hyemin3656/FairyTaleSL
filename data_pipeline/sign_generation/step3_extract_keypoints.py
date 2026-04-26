"""
KCISA 수어 영상에서 MediaPipe Tasks API로 keypoint 추출.

출력 포맷: 225차원 npy (좌손63 + 우손63 + 포즈99) — AI Hub 포맷과 동일
  - 좌손: 21점 × 3(x,y,z) = 63
  - 우손: 21점 × 3       = 63
  - 포즈: 33점 × 3       = 99
프레임 평균으로 글로스당 대표 벡터 1개 저장.
"""
import json
import urllib.request
import numpy as np
import cv2
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "data/videos"
KEYPOINT_DIR = BASE_DIR / "data/keypoints"
KEYPOINT_DIR.mkdir(exist_ok=True)
GLOSS_LIST_PATH = BASE_DIR / "data/gloss_list.json"

POSE_MODEL_PATH  = BASE_DIR / "pose_landmarker_lite.task"
HAND_MODEL_PATH  = BASE_DIR / "hand_landmarker.task"

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def _download_model(path: Path, url: str):
    if not path.exists():
        print(f"모델 다운로드: {path.name} ...", flush=True)
        urllib.request.urlretrieve(url, str(path))
        print(f"  → 완료")


def _build_landmarkers():
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    pose_lm  = vision.PoseLandmarker.create_from_options(pose_opts)
    hand_lm  = vision.HandLandmarker.create_from_options(hand_opts)
    return pose_lm, hand_lm


def _lm_to_arr(landmarks, n: int) -> np.ndarray:
    if landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
    return np.zeros(n * 3, dtype=np.float32)


def extract_frame_vector(rgb_frame, pose_lm, hand_lm) -> np.ndarray | None:
    """프레임 → 225차원 벡터 (좌손63 + 우손63 + 포즈99)."""
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    pose_res = pose_lm.detect(mp_img)
    hand_res = hand_lm.detect(mp_img)

    pose = _lm_to_arr(
        pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None, 33
    )

    lhand = np.zeros(63, dtype=np.float32)
    rhand = np.zeros(63, dtype=np.float32)
    for i, cat in enumerate(hand_res.handedness):
        label = cat[0].category_name  # "Left" | "Right"
        lms = hand_res.hand_landmarks[i]
        arr = _lm_to_arr(lms, 21)
        if label == "Left":
            lhand = arr
        else:
            rhand = arr

    vec = np.concatenate([lhand, rhand, pose])
    return None if np.all(vec == 0) else vec


def process_video(video_path: Path, pose_lm, hand_lm) -> np.ndarray | None:
    """영상 → 프레임 평균 keypoint 벡터."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vec = extract_frame_vector(rgb, pose_lm, hand_lm)
        if vec is not None:
            frames.append(vec)

    cap.release()
    return np.mean(frames, axis=0) if frames else None


def main():
    _download_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    _download_model(HAND_MODEL_PATH, HAND_MODEL_URL)

    with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
        gloss_list = json.load(f)

    targets = [g for g in gloss_list if g.get("video_path") and not g.get("keypoint_path")]
    print(f"\nkeypoint 추출 대상: {len(targets)}개\n")

    pose_lm, hand_lm = _build_landmarkers()
    success, fail = 0, 0

    try:
        for g in targets:
            gloss = g["gloss"]
            video_path = Path(g["video_path"])
            save_path = KEYPOINT_DIR / f"{gloss}.npy"

            if not video_path.exists():
                print(f"❌ {gloss}: 영상 없음")
                fail += 1
                continue

            print(f"추출 중: {gloss}", end=" ... ", flush=True)
            avg_vec = process_video(video_path, pose_lm, hand_lm)

            if avg_vec is not None:
                np.save(str(save_path), avg_vec)
                g["keypoint_path"] = str(save_path)
                g["registered"] = True
                success += 1
                print("✅")
            else:
                print("❌ (감지 실패)")
                fail += 1
    finally:
        pose_lm.close()
        hand_lm.close()

    with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(gloss_list, f, ensure_ascii=False, indent=2)

    total_kp = sum(1 for g in gloss_list if g.get("keypoint_path"))
    print(f"\n========== 결과 ==========")
    print(f"이번 추출: 성공 {success}개 / 실패 {fail}개")
    print(f"전체 keypoint 등록: {total_kp}/220개")


if __name__ == "__main__":
    main()
