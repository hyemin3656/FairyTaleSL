"""
step2_retarget_missing.py
동화 합성/폴백 글로스 58개 처리:
  1단계: sldict에서 해당 단어 검색 → keypoint 추출
  2단계: sldict에 없으면 motion_db 내 registered=1 글로스 중 유사어 keypoint 복사

실행:
  cd data_pipeline/sign_generation
  venv/bin/python step2_retarget_missing.py
"""

import difflib
import json
import re
import sqlite3
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

BASE_DIR  = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "data/videos"
DB_PATH   = BASE_DIR / "data/motion_db.sqlite"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://sldict.korean.go.kr",
}
KEYPOINT_FPS = 15

# 처리 대상 글로스
TARGET_GLOSSES = [
    '거느리','거북이','견우','결승선','고조선','교훈','그때','금도끼','금은보화',
    '까치','꼭대기','날개옷','놀부','눈멀','눈물','단군왕검','도끼','돌려보내',
    '돌아가시','동아줄','두레박','떠다니','먼저','밀가루','반드시','빠뜨리',
    '산신령','살아가','생계','쇠도끼','수탉','심청이','심학규','쏟아지',
    '연못가','열심히','오래도록','오물','용왕','우리나라','은하수','인당수',
    '잡아먹','장화','정성껏','중간','직녀','쫓기','찾아가','찾아오',
    '콩쥐','태백산','팥죽','팥쥐','홍련','환웅','효도','효심'
]

# 수동 유사어 힌트 (sldict에 없을 가능성 높은 고유명사/합성어)
MANUAL_SIMILAR = {
    # 고유명사 → 의미상 유사 일반어
    '심청이':   '아이',
    '심학규':   '아버지',
    '단군왕검': '왕',
    '환웅':     '하늘',
    '고조선':   '나라',
    '놀부':     '욕심',
    '콩쥐':     '아이',
    '팥쥐':     '아이',
    '장화':     '아이',
    '홍련':     '아이',
    '견우':     '사람',
    '직녀':     '사람',
    # 합성어 → 핵심 형태소
    '금은보화': '보물',
    '인당수':   '바다',
    '동아줄':   '줄',
    '두레박':   '물',
    '쇠도끼':   '도끼',
    '금도끼':   '도끼',
    '날개옷':   '날개',
    '태백산':   '산',
    '결승선':   '목표',
    '산신령':   '신',
    '은하수':   '별',
    # 동작형 → 기본형
    '거느리':   '이끌다',
    '거북이':   '동물',
    '까치':     '새',
    '꼭대기':   '산',
    '그때':     '때',
    '교훈':     '배우다',
    '눈멀':     '눈',
    '눈물':     '울다',
    '돌려보내': '보내다',
    '돌아가시': '돌아가다',
    '떠다니':   '떠나다',
    '먼저':     '앞',
    '밀가루':   '쌀',
    '반드시':   '꼭',
    '빠뜨리':   '떨어지다',
    '살아가':   '살다',
    '생계':     '생활',
    '수탉':     '닭',
    '쏟아지':   '쏟다',
    '연못가':   '연못',
    '열심히':   '열심',
    '오래도록': '오랫동안',
    '오물':     '더럽다',
    '용왕':     '왕',
    '우리나라': '나라',
    '잡아먹':   '먹다',
    '정성껏':   '정성',
    '중간':     '가운데',
    '쫓기':     '도망',
    '찾아가':   '가다',
    '찾아오':   '오다',
    '팥죽':     '죽',
    '효도':     '부모',
    '효심':     '마음',
}


# ── sldict 검색 ───────────────────────────────────────────────

def search_sldict(word: str) -> tuple[str | None, str | None]:
    """단어명으로 sldict 검색 → (단어명, 영상URL). 없으면 (None, None)."""
    search_url = (
        f"https://sldict.korean.go.kr/front/sign/signContentsSearchList.do"
        f"?search_keyword={urllib.parse.quote(word)}&top_gubun=&pageIndex=1"
    )
    try:
        req = urllib.request.Request(search_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None, None

    # origin_no 추출
    nos = re.findall(r"origin_no=(\d+)", html)
    if not nos:
        return None, None

    # 첫 번째 결과 상세 조회
    from step2_sldict_full import fetch_entry
    for no in nos[:3]:
        w, url = fetch_entry(int(no))
        if w and w == word and url:
            return w, url
        if w and url:
            return w, url
    return None, None


# ── MediaPipe ────────────────────────────────────────────────

POSE_MODEL = BASE_DIR / "pose_landmarker_lite.task"
HAND_MODEL = BASE_DIR / "hand_landmarker.task"


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


def _lm_to_arr(landmarks, n):
    if landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
    return np.zeros(n * 3, dtype=np.float32)


def extract_keypoints(video_path: Path, pose_lm, hand_lm) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / KEYPOINT_FPS))
    frames = []
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
            pose = _lm_to_arr(pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None, 33)
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
    return np.array(frames, dtype=np.float32) if frames else None


def download_video(url: str, save_path: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) < 1000:
            return False
        save_path.write_bytes(data)
        return True
    except Exception as e:
        print(f"    [DL오류] {e}")
        return False


# ── DB 조작 ──────────────────────────────────────────────────

def get_registered_glosses() -> dict[str, bytes]:
    """registered=1 글로스 → keypoint_data 전체 로드."""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT gloss, keypoint_data FROM motion_db WHERE is_registered=1 AND keypoint_data IS NOT NULL"
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def find_most_similar(word: str, candidates: list[str]) -> str:
    """difflib 기반 가장 유사한 글로스 반환."""
    if word in MANUAL_SIMILAR and MANUAL_SIMILAR[word] in candidates:
        return MANUAL_SIMILAR[word]
    matches = difflib.get_close_matches(word, candidates, n=1, cutoff=0.0)
    return matches[0] if matches else candidates[0]


def update_keypoint(gloss: str, kp_blob: bytes, source: str, similar_to: str | None = None):
    conn = sqlite3.connect(str(DB_PATH))
    if similar_to:
        conn.execute(
            "UPDATE motion_db SET keypoint_data=?, fallback_type=? WHERE gloss=?",
            (kp_blob, f"similar:{similar_to}", gloss)
        )
        print(f"  → 유사어 [{similar_to}] keypoint 복사")
    else:
        conn.execute(
            "UPDATE motion_db SET keypoint_data=?, is_registered=1, fallback_type=NULL WHERE gloss=?",
            (kp_blob, gloss)
        )
        print(f"  → sldict 실제 keypoint 저장")
    conn.commit()
    conn.close()


# ── 메인 ─────────────────────────────────────────────────────

import urllib.parse

def main():
    pose_lm, hand_lm = build_landmarkers()
    registered_map = get_registered_glosses()
    reg_list = list(registered_map.keys())

    print(f"처리 대상: {len(TARGET_GLOSSES)}개")
    print(f"registered 풀: {len(reg_list)}개\n")

    sldict_ok = 0
    similar_ok = 0
    fail = 0

    for i, gloss in enumerate(TARGET_GLOSSES, 1):
        print(f"[{i}/{len(TARGET_GLOSSES)}] {gloss}", end=" ")

        # 1단계: sldict 검색
        word, video_url = search_sldict(gloss)
        if video_url:
            save_path = VIDEO_DIR / f"{gloss}_tmp.mp4"
            if download_video(video_url, save_path):
                kp = extract_keypoints(save_path, pose_lm, hand_lm)
                save_path.unlink(missing_ok=True)
                if kp is not None and len(kp) > 0:
                    update_keypoint(gloss, kp.tobytes(), "sldict")
                    sldict_ok += 1
                    time.sleep(0.1)
                    continue

        # 2단계: 유사어 대체
        similar = find_most_similar(gloss, reg_list)
        update_keypoint(gloss, registered_map[similar], "similar", similar)
        similar_ok += 1
        time.sleep(0.05)

    print(f"\n{'='*50}")
    print(f"sldict 실제 수집: {sldict_ok}개")
    print(f"유사어 대체:      {similar_ok}개")
    print(f"실패:             {fail}개")


if __name__ == "__main__":
    main()
