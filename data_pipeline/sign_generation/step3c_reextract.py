"""
step3c_reextract.py
generic placeholder BLOB을 갖는 글로스를 영상에서 재추출하여 motion_db.sqlite 업데이트.

사용법:
  python step3c_reextract.py            # 동화 사용 글로스 우선 처리
  python step3c_reextract.py --all      # 전체 5,696개 처리
"""
import argparse
import hashlib
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "data/motion_db.sqlite"
VIDEO_DIR  = BASE_DIR / "data/videos"
TALES_PATH = BASE_DIR / "data/fairy_tales_structured.json"

POSE_MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

TARGET_FPS = 15


def _download_model(path: Path, url: str):
    if not path.exists():
        print(f"모델 다운로드: {path.name} ...", flush=True)
        urllib.request.urlretrieve(url, str(path))
        print("  완료")


def _build_landmarkers():
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
    )
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
    )
    return (
        vision.PoseLandmarker.create_from_options(pose_opts),
        vision.HandLandmarker.create_from_options(hand_opts),
    )


def _extract_frame(rgb, pose_lm, hand_lm) -> np.ndarray | None:
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pose_res = pose_lm.detect(mp_img)
    hand_res = hand_lm.detect(mp_img)

    pose = np.zeros(99, dtype=np.float32)
    if pose_res.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_res.pose_landmarks[0]],
            dtype=np.float32,
        ).flatten()

    lhand = np.zeros(63, dtype=np.float32)
    rhand = np.zeros(63, dtype=np.float32)
    for i, cat in enumerate(hand_res.handedness):
        arr = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_res.hand_landmarks[i]],
            dtype=np.float32,
        ).flatten()
        if cat[0].category_name == "Left":
            lhand = arr
        else:
            rhand = arr

    vec = np.concatenate([lhand, rhand, pose])
    return None if np.all(vec == 0) else vec


def process_video(path: Path, pose_lm, hand_lm) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / TARGET_FPS))

    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vec = _extract_frame(rgb, pose_lm, hand_lm)
            if vec is not None:
                frames.append(vec)
        idx += 1
    cap.release()
    return np.array(frames, dtype=np.float32) if frames else None


def get_generic_hash(conn) -> str:
    """DB에서 generic placeholder BLOB의 MD5 해시 반환."""
    row = conn.execute(
        "SELECT keypoint_data FROM motion_db ORDER BY id LIMIT 1"
    ).fetchone()
    # 가장 많이 공유되는 BLOB을 generic으로 판별
    rows = conn.execute("SELECT keypoint_data FROM motion_db WHERE keypoint_data IS NOT NULL").fetchall()
    from collections import Counter
    counter = Counter(hashlib.md5(r[0]).hexdigest() for r in rows)
    return counter.most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="전체 5,696개 처리 (기본: 동화 사용 글로스만)")
    args = parser.parse_args()

    _download_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    _download_model(HAND_MODEL_PATH, HAND_MODEL_URL)

    conn = sqlite3.connect(str(DB_PATH))
    generic_hash = get_generic_hash(conn)
    print(f"Generic BLOB 해시: {generic_hash[:8]}...")

    # 대상 글로스 선정
    all_rows = conn.execute("SELECT gloss, keypoint_data FROM motion_db").fetchall()
    generic_glosses = [
        g for g, blob in all_rows
        if blob and hashlib.md5(blob).hexdigest() == generic_hash
        and (VIDEO_DIR / f"{g}.mp4").exists()
    ]

    if not args.all:
        with open(TALES_PATH, encoding="utf-8") as f:
            tales = json.load(f)
        used = set()
        for t in tales:
            for s in t.get("segments", []):
                for sent in s.get("sentences", []):
                    used.update(sent.get("ksl_glosses", []))
        targets = [g for g in generic_glosses if g in used]
        print(f"동화 사용 글로스 우선 처리: {len(targets)}개 (--all 로 전체 {len(generic_glosses)}개 처리 가능)")
    else:
        targets = generic_glosses
        print(f"전체 처리: {len(targets)}개")

    pose_lm, hand_lm = _build_landmarkers()
    success, fail = 0, 0

    try:
        for i, gloss in enumerate(targets, 1):
            video = VIDEO_DIR / f"{gloss}.mp4"
            print(f"[{i}/{len(targets)}] {gloss}", end=" ... ", flush=True)
            seq = process_video(video, pose_lm, hand_lm)
            if seq is not None and len(seq) >= 5:
                conn.execute(
                    "UPDATE motion_db SET keypoint_data=?, frame_count=?, is_registered=1 WHERE gloss=?",
                    (seq.tobytes(), seq.shape[0], gloss),
                )
                conn.commit()
                print(f"✅ {seq.shape[0]}프레임")
                success += 1
            else:
                n = len(seq) if seq is not None else 0
                print(f"❌ 감지 실패 ({n}프레임)")
                fail += 1
    finally:
        pose_lm.close()
        hand_lm.close()
        conn.close()

    print(f"\n완료: 성공 {success}개 / 실패 {fail}개")


if __name__ == "__main__":
    main()
