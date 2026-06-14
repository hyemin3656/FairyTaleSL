"""
step3_extract_from_existing.py
data/videos/ 에 이미 있는 영상에서 keypoint 추출 → motion_db 업서트 → 영상 삭제

대상:
  - keypoint 없는 영상
  - 합성 30프레임짜리 (step3b_synthetic_sequences 산출물) → 실제 영상으로 교체

실행:
  venv/bin/python step3_extract_from_existing.py
"""

import sqlite3
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request

BASE_DIR      = Path(__file__).parent
VIDEO_DIR     = BASE_DIR / "data/videos"
DB_PATH       = BASE_DIR / "data/motion_db.sqlite"
POSE_MODEL    = BASE_DIR / "pose_landmarker_lite.task"
HAND_MODEL    = BASE_DIR / "hand_landmarker.task"
POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
TARGET_FPS    = 15


def ensure_models():
    for path, url in [(POSE_MODEL, POSE_URL), (HAND_MODEL, HAND_URL)]:
        if not path.exists():
            print(f"모델 다운로드: {path.name} ...")
            urllib.request.urlretrieve(url, str(path))


def build_landmarkers():
    pose_lm = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
    )
    hand_lm = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
    )
    return pose_lm, hand_lm


def lm_to_arr(lms, n):
    if lms:
        return np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32).flatten()
    return np.zeros(n * 3, dtype=np.float32)


def extract(video_path: Path, pose_lm, hand_lm) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / TARGET_FPS))
    frames, idx = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pr = pose_lm.detect(img)
            hr = hand_lm.detect(img)
            pose  = lm_to_arr(pr.pose_landmarks[0] if pr.pose_landmarks else None, 33)
            lhand = np.zeros(63, dtype=np.float32)
            rhand = np.zeros(63, dtype=np.float32)
            for i, cat in enumerate(hr.handedness):
                arr = lm_to_arr(hr.hand_landmarks[i], 21)
                if cat[0].category_name == "Left":
                    lhand = arr
                else:
                    rhand = arr
            vec = np.concatenate([lhand, rhand, pose])
            if not np.all(vec == 0):
                frames.append(vec)
        idx += 1
    cap.release()
    return np.array(frames, dtype=np.float32) if frames else None


def upsert(gloss: str, kp: np.ndarray, conn: sqlite3.Connection):
    blob = kp.astype(np.float32).tobytes()
    conn.execute("""
        INSERT INTO motion_db (gloss, keypoint_data, emotion_label)
        VALUES (?, ?, 'neutral')
        ON CONFLICT(gloss) DO UPDATE SET keypoint_data = excluded.keypoint_data
    """, (gloss, blob))
    conn.commit()


def needs_extract(gloss: str, conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT keypoint_data FROM motion_db WHERE gloss=?", (gloss,)
    ).fetchone()
    if not row or not row[0]:
        return True
    arr = np.frombuffer(row[0], dtype=np.float32)
    return len(arr) // 225 == 30   # 합성 시퀀스


def main():
    ensure_models()
    pose_lm, hand_lm = build_landmarkers()
    conn = sqlite3.connect(str(DB_PATH))

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    targets = [v for v in videos if needs_extract(v.stem, conn)]
    total = len(targets)
    print(f"추출 대상: {total}개 / 전체 영상 {len(videos)}개\n")

    success = fail = no_kp = 0
    t0 = time.time()

    for i, v in enumerate(targets, 1):
        gloss = v.stem
        kp = extract(v, pose_lm, hand_lm)

        if kp is None or len(kp) == 0:
            print(f"[{i}/{total}] {gloss}: ⚠️ keypoint 없음")
            no_kp += 1
        else:
            upsert(gloss, kp, conn)
            print(f"[{i}/{total}] {gloss}: ✅ {len(kp)}프레임")
            success += 1

        # 영상 삭제 (keypoint DB에 저장됨)
        v.unlink(missing_ok=True)

        if i % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (total - i)
            print(f"\n  === {i}/{total} | ✅{success} ⚠️{no_kp} | "
                  f"경과:{elapsed/60:.0f}분 ETA:{eta/60:.0f}분 ===\n")

    conn.close()
    print(f"\n{'='*50}")
    print(f"완료: {success}개 | keypoint 없음: {no_kp}개")
    print(f"소요 시간: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
