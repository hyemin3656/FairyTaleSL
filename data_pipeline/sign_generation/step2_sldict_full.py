"""
step2_sldict_full.py
sldict.korean.go.kr 전체 수어 영상 수집 → keypoint 추출 → motion_db 적재

흐름:
  1단계: origin_no=1~MAX_ORIGIN_NO 순회로 sldict 전체 수집
         단어명: og:title "한국수어사전_단어명" 파싱
         영상URL: _105X105.jpg → _320X240.mp4 치환
  2단계: 영상 다운로드 (이미 있으면 스킵)
  3단계: MediaPipe keypoint 추출 (N×225 float32, 15fps)
  4단계: motion_db.sqlite 업서트

실행:
  cd data_pipeline/sign_generation
  venv/bin/python step2_sldict_full.py

주의:
  - 약 25,500개 항목 처리 → 수 시간 소요
  - 중간 중단 후 재실행 시 sldict_full_log.json 기준으로 이어받기
  - 네트워크 fetch/download는 FETCH_WORKERS 스레드로 병렬 처리
  - MediaPipe 추출은 단일 스레드 (thread-safety 보장)
"""

import json
import queue
import re
import sqlite3
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

# ── 경로 ──────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "data/videos"
DB_PATH   = BASE_DIR / "data/motion_db.sqlite"
LOG_PATH  = BASE_DIR / "data/sldict_full_log.json"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://sldict.korean.go.kr",
}

MAX_ORIGIN_NO = 25600
KEYPOINT_FPS  = 15
FETCH_WORKERS = 8   # 네트워크 병렬 스레드 수
LOG_INTERVAL  = 200  # 로그 저장 주기


# ── 1단계: sldict origin_no 조회 ─────────────────────────────

def fetch_entry(origin_no: int) -> tuple[str | None, str | None]:
    url = (
        f"https://sldict.korean.go.kr/front/sign/signContentsView.do"
        f"?origin_no={origin_no}&top_gubun=CTE"
    )
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None, None

    og = re.search(r'content="한국수어사전_([^"]+)"', html)
    word = og.group(1).strip() if og else None

    preview = re.search(r'(https?://[^\s"\']+_105X105\.jpg)', html)
    if not preview:
        return word, None
    video_url = (
        preview.group(1)
        .replace("_105X105.jpg", "_320X240.mp4")
        .replace("http://sldict.korean.go.kr", "https://sldict.korean.go.kr")
    )
    return word, video_url


# ── 2단계: 영상 다운로드 ──────────────────────────────────────

def download_video(url: str, save_path: Path) -> bool:
    if save_path.exists() and save_path.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) < 1000:
            return False
        save_path.write_bytes(data)
        return True
    except Exception as e:
        print(f"    [DL] {e}")
        return False


# ── 3단계: MediaPipe keypoint 추출 ───────────────────────────

POSE_MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def _ensure_models():
    for path, url in [(POSE_MODEL_PATH, POSE_MODEL_URL), (HAND_MODEL_PATH, HAND_MODEL_URL)]:
        if not path.exists():
            print(f"모델 다운로드: {path.name} ...")
            urllib.request.urlretrieve(url, str(path))
            print("  → 완료")


def _build_landmarkers():
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


def _lm_to_arr(landmarks, n: int) -> np.ndarray:
    if landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
    return np.zeros(n * 3, dtype=np.float32)


def extract_keypoints(video_path: Path, pose_lm, hand_lm) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / KEYPOINT_FPS))
    frames: list[np.ndarray] = []
    idx = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pose_res = pose_lm.detect(mp_img)
            hand_res = hand_lm.detect(mp_img)

            pose = _lm_to_arr(
                pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None, 33
            )
            lhand = np.zeros(63, dtype=np.float32)
            rhand = np.zeros(63, dtype=np.float32)
            for i, cat in enumerate(hand_res.handedness):
                arr = _lm_to_arr(hand_res.hand_landmarks[i], 21)
                if cat[0].category_name == "Left":
                    lhand = arr
                else:
                    rhand = arr

            vec = np.concatenate([lhand, rhand, pose])
            if not np.all(vec == 0):
                frames.append(vec)
        idx += 1

    cap.release()
    if not frames:
        return None
    return np.array(frames, dtype=np.float32)


# ── 4단계: motion_db 업서트 ──────────────────────────────────

_db_lock = threading.Lock()

def upsert(gloss: str, kp_seq: np.ndarray) -> None:
    blob = kp_seq.astype(np.float32).tobytes()
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""
            INSERT INTO motion_db (gloss, keypoint_data, emotion_label)
            VALUES (?, ?, 'neutral')
            ON CONFLICT(gloss) DO UPDATE SET keypoint_data = excluded.keypoint_data
        """, (gloss, blob))
        conn.commit()
        conn.close()


def has_keypoint(gloss: str) -> bool:
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT keypoint_data FROM motion_db WHERE gloss=?", (gloss,)
        ).fetchone()
        conn.close()
    return bool(row and row[0])


# ── 로그 ─────────────────────────────────────────────────────

_log_lock = threading.Lock()

def load_log() -> dict:
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"done_nos": [], "done_words": [], "no_video_nos": [], "failed_nos": []}


def save_log(log: dict) -> None:
    with _log_lock:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)


# ── 네트워크 fetch + download 작업 (병렬) ────────────────────

def fetch_and_download(origin_no: int) -> dict:
    """fetch_entry + download_video → 결과 dict 반환."""
    word, video_url = fetch_entry(origin_no)

    if not video_url:
        return {"origin_no": origin_no, "status": "no_video", "word": word}

    gloss = word or f"sign_{origin_no}"

    if word and has_keypoint(word):
        return {"origin_no": origin_no, "status": "already_done", "gloss": gloss}

    save_path = VIDEO_DIR / f"{gloss}.mp4"
    ok = download_video(video_url, save_path)
    if not ok:
        return {"origin_no": origin_no, "status": "dl_fail", "gloss": gloss}

    return {"origin_no": origin_no, "status": "ready", "gloss": gloss, "path": save_path}


# ── 메인 ─────────────────────────────────────────────────────

def main():
    _ensure_models()
    pose_lm, hand_lm = _build_landmarkers()

    log = load_log()
    done_nos     = set(log["done_nos"])
    no_video_nos = set(log.get("no_video_nos", []))
    failed_nos   = set(log.get("failed_nos", []))
    skip_nos     = done_nos | no_video_nos

    total    = MAX_ORIGIN_NO
    success  = len(done_nos)
    no_video = len(no_video_nos)
    dl_fail  = len(failed_nos)
    processed = 0

    pending = [n for n in range(1, MAX_ORIGIN_NO + 1) if n not in skip_nos]
    print(f"sldict 전체 수집 시작 (origin_no 1~{MAX_ORIGIN_NO})")
    print(f"이미 완료: {success}개 | 영상없음: {no_video}개 | 남은 수: {len(pending)}개")
    print(f"병렬 fetch 스레드: {FETCH_WORKERS}개\n")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {executor.submit(fetch_and_download, n): n for n in pending}

        for future in as_completed(futures):
            result = future.result()
            origin_no = result["origin_no"]
            status    = result["status"]
            processed += 1

            if status == "no_video":
                log["no_video_nos"].append(origin_no)
                no_video_nos.add(origin_no)
                no_video += 1

            elif status == "already_done":
                log["done_nos"].append(origin_no)
                done_nos.add(origin_no)
                success += 1

            elif status == "dl_fail":
                log["failed_nos"].append(origin_no)
                failed_nos.add(origin_no)
                dl_fail += 1
                print(f"[{origin_no}] {result.get('gloss','')} ❌ 다운로드 실패")

            elif status == "ready":
                gloss = result["gloss"]
                save_path = result["path"]
                kp = extract_keypoints(save_path, pose_lm, hand_lm)
                save_path.unlink(missing_ok=True)

                if kp is None or len(kp) == 0:
                    print(f"[{origin_no}] {gloss} ⚠️ keypoint 없음")
                    log["no_video_nos"].append(origin_no)
                    no_video_nos.add(origin_no)
                    no_video += 1
                else:
                    upsert(gloss, kp)
                    log["done_nos"].append(origin_no)
                    log["done_words"].append(gloss)
                    done_nos.add(origin_no)
                    success += 1
                    print(f"[{origin_no}] {gloss} ✅ {len(kp)}프레임")

            if processed % LOG_INTERVAL == 0:
                save_log(log)
                elapsed = time.time() - start_time
                rate = processed / elapsed * 3600
                remaining = (len(pending) - processed) / (processed / elapsed) / 3600
                print(
                    f"\n  === [{processed}/{len(pending)}] "
                    f"✅{success} 영상없음:{no_video} DL실패:{dl_fail} "
                    f"| {rate:.0f}개/시간 | 남은시간:{remaining:.1f}h ===\n"
                )

    save_log(log)
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"완료: {success}개 | 영상없음: {no_video}개 | DL실패: {dl_fail}개")
    print(f"소요시간: {elapsed/3600:.1f}시간")
    print(f"저장 경로: {DB_PATH}")


if __name__ == "__main__":
    main()
