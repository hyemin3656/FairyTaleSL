"""
step3_stream_extract.py
KCISA API에서 영상 URL을 받아 스트리밍으로 keypoint 추출.
영상 파일을 디스크에 저장하지 않고 tmpfs 경유 즉시 처리.

흐름: KCISA API 스캔 → 영상 URL 취득 → 임시 다운로드 → keypoint 추출 →
      motion_db.sqlite UPSERT → 임시 파일 삭제
"""
import json
import sqlite3
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests

BASE_DIR        = Path(__file__).parent
GLOSS_LIST_PATH = BASE_DIR / "data/gloss_list.json"
DB_PATH         = BASE_DIR / "data/motion_db.sqlite"

POSE_MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

API_KEY = "f12f5fa4-0583-4276-8226-62b3748ea583"
API_URL = "https://api.kcisa.kr/openapi/service/rest/meta13/getCTE01701"

TARGET_FPS = 15


# ── 모델 다운로드 ─────────────────────────────────────────────
def _download_model(path: Path, url: str):
    if not path.exists():
        print(f"모델 다운로드: {path.name} ...", flush=True)
        urllib.request.urlretrieve(url, str(path))
        print("  완료")


# ── landmarker 생성 ───────────────────────────────────────────
def _build_landmarkers():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    pose_lm = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
    )
    hand_lm = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
    )
    return pose_lm, hand_lm


# ── 프레임 → 225차원 벡터 ─────────────────────────────────────
def _lm_to_arr(landmarks, n: int) -> np.ndarray:
    if landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
    return np.zeros(n * 3, dtype=np.float32)


def extract_frame_vector(rgb_frame, pose_lm, hand_lm) -> np.ndarray | None:
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    pose_res = pose_lm.detect(mp_img)
    hand_res = hand_lm.detect(mp_img)

    pose  = _lm_to_arr(pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None, 33)
    lhand = np.zeros(63, dtype=np.float32)
    rhand = np.zeros(63, dtype=np.float32)
    for i, cat in enumerate(hand_res.handedness):
        arr = _lm_to_arr(hand_res.hand_landmarks[i], 21)
        if cat[0].category_name == "Left":
            lhand = arr
        else:
            rhand = arr

    vec = np.concatenate([lhand, rhand, pose])
    return None if np.all(vec == 0) else vec


def process_video(video_path: Path, pose_lm, hand_lm) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / TARGET_FPS))
    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vec = extract_frame_vector(rgb, pose_lm, hand_lm)
            if vec is not None:
                frames.append(vec)
        idx += 1
    cap.release()
    return np.array(frames, dtype=np.float32) if frames else None


# ── KCISA API 전체 스캔 → {글로스: url} ──────────────────────
def fetch_url_map(needed: set[str]) -> dict[str, str]:
    url_map: dict[str, str] = {}
    page = 1
    total_pages = None

    print("KCISA API 스캔 중...", flush=True)
    while True:
        try:
            res = requests.get(API_URL, params={"serviceKey": API_KEY, "numOfRows": 100, "pageNo": page}, timeout=15)
            root = ET.fromstring(res.text)
        except Exception as e:
            print(f"  [오류] page {page}: {e}")
            break

        if total_pages is None:
            try:
                total_pages = (int(root.findtext(".//totalCount") or 0) + 99) // 100
                print(f"  전체 {root.findtext('.//totalCount')}건 ({total_pages}페이지)")
            except Exception:
                pass

        items = root.findall(".//item")
        if not items:
            break

        for item in items:
            title     = item.findtext("title", "").strip()
            video_url = item.findtext("subDescription", "").strip().replace(
                "http://sldict.korean.go.kr", "https://sldict.korean.go.kr"
            )
            if not title or not video_url:
                continue
            for word in title.split(","):
                word = word.strip()
                if word in needed and word not in url_map:
                    url_map[word] = video_url

        if page % 5 == 0:
            print(f"  page {page}/{total_pages or '?'} | 매핑: {len(url_map)}/{len(needed)}")

        if url_map.keys() >= needed or (total_pages and page >= total_pages):
            break
        page += 1
        time.sleep(0.1)

    print(f"API 스캔 완료: {len(url_map)}/{len(needed)} 매핑")
    return url_map


# ── 영상 스트림 다운로드 → keypoint 추출 → 즉시 삭제 ─────────
def stream_extract(gloss: str, url: str, pose_lm, hand_lm) -> np.ndarray | None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        res = requests.get(url, stream=True, timeout=30)
        with open(tmp_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
        if tmp_path.stat().st_size < 1000:
            return None
        return process_video(tmp_path, pose_lm, hand_lm)
    except Exception as e:
        print(f"  [오류] {gloss}: {e}")
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


# ── motion_db UPSERT ─────────────────────────────────────────
def upsert_keypoint(conn: sqlite3.Connection, gloss: str, seq: np.ndarray):
    blob = seq.tobytes()
    conn.execute("""
        UPDATE motion_db
        SET keypoint_data = ?,
            is_registered = 1
        WHERE gloss = ?
    """, (blob, gloss))
    if conn.execute("SELECT changes()").fetchone()[0] == 0:
        conn.execute("""
            INSERT OR IGNORE INTO motion_db (gloss, keypoint_data, is_registered)
            VALUES (?, ?, 1)
        """, (gloss, blob))


# ── 진행 상태 저장/복구 ───────────────────────────────────────
PROGRESS_PATH = BASE_DIR / "data/stream_extract_progress.json"

def load_progress() -> set[str]:
    if PROGRESS_PATH.exists():
        return set(json.loads(PROGRESS_PATH.read_text()))
    return set()

def save_progress(done: set[str]):
    PROGRESS_PATH.write_text(json.dumps(sorted(done)))


# ── 메인 ─────────────────────────────────────────────────────
def main():
    _download_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    _download_model(HAND_MODEL_PATH, HAND_MODEL_URL)

    # 아직 실제 keypoint 없는 글로스 대상
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT gloss FROM motion_db WHERE is_registered = 0 OR is_registered IS NULL"
    ).fetchall()
    needed = {r[0] for r in rows}
    print(f"keypoint 추출 대상: {len(needed)}개")

    # 이전 진행분 복구
    done = load_progress()
    needed -= done
    if done:
        print(f"이전 진행분 복구: {len(done)}개 건너뜀")
    print(f"남은 대상: {len(needed)}개\n")

    # API 스캔
    url_map = fetch_url_map(needed)
    not_found = needed - url_map.keys()
    print(f"API 미발견: {len(not_found)}개\n")

    # 스트림 추출
    pose_lm, hand_lm = _build_landmarkers()
    success = fail = skipped = 0

    try:
        for i, (gloss, url) in enumerate(url_map.items(), 1):
            print(f"[{i}/{len(url_map)}] {gloss}", end=" ... ", flush=True)
            seq = stream_extract(gloss, url, pose_lm, hand_lm)
            if seq is not None:
                upsert_keypoint(conn, gloss, seq)
                conn.commit()
                done.add(gloss)
                save_progress(done)
                print(f"✅ {seq.shape[0]}프레임")
                success += 1
            else:
                print("❌")
                fail += 1

            if i % 50 == 0:
                print(f"  --- {i}개 처리 완료 (성공 {success} / 실패 {fail}) ---\n")

    except KeyboardInterrupt:
        print("\n\n중단됨. 진행 상태 저장됨.")
    finally:
        pose_lm.close()
        hand_lm.close()
        conn.close()

    print(f"\n완료: 성공 {success}개 / 실패 {fail}개 / 미발견 {len(not_found)}개")
    print(f"다음 단계: python step3b_synthetic_sequences.py && docker compose restart backend")


if __name__ == "__main__":
    main()
