"""
AI Hub 한국수어 데이터셋에서 keypoint를 추출해 motion DB에 저장.

입력:
  - [라벨]01_real_word_morpheme.zip : WORD번호 → 한국어 단어명 매핑
  - [라벨]02_syn_word_keypoint.zip  : WORD번호별 keypoint JSON (프레임 단위)
  - data/gloss_list.json            : 220개 글로스 목록

출력:
  - data/keypoints/{gloss}.npy      : 글로스별 평균 keypoint 벡터 (225차원)
  - data/gloss_list.json            : keypoint_path 업데이트
"""
import json
import zipfile
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KEYPOINT_DIR = DATA_DIR / "keypoints"
KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)

MORPHEME_ZIP = Path("/Users/SJ/PSYcho/수어 영상/2.Validation/[라벨]01_real_word_morpheme.zip")
KEYPOINT_ZIP = Path("/Users/SJ/PSYcho/수어 영상/2.Validation/[라벨]02_syn_word_keypoint.zip")
GLOSS_LIST_PATH = DATA_DIR / "gloss_list.json"

# OpenPose keypoint 인덱스 구성 (AI Hub SYN 포맷)
# pose: 25점×3 = 75, face: 70점×3 = 210, hand_left: 21점×3 = 63, hand_right: 21점×3 = 63 → 총 411
# 우리 Motion DB는 225차원(좌손63 + 우손63 + 포즈99)을 사용하므로 동일하게 맞춤
POSE_DIM = 75    # 25 keypoints × (x, y, confidence)
HAND_DIM = 63    # 21 keypoints × (x, y, confidence)


def build_word_dict(morpheme_zip_path: Path) -> dict[str, str]:
    """WORD번호 → 한국어 단어명 매핑 딕셔너리 생성."""
    word_dict: dict[str, str] = {}
    with zipfile.ZipFile(morpheme_zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith("_morpheme.json"):
                continue
            m = re.search(r"(WORD\d+)", name)
            if not m:
                continue
            word_key = m.group(1)
            if word_key in word_dict:
                continue
            with zf.open(name) as f:
                try:
                    data = json.load(f)
                    label = data["data"][0]["attributes"][0]["name"]
                    word_dict[word_key] = label.strip()
                except (KeyError, IndexError, json.JSONDecodeError):
                    pass
    return word_dict


def extract_keypoint_vector(kp_json: dict) -> np.ndarray | None:
    """keypoint JSON 1프레임 → 225차원 벡터 (좌손63 + 우손63 + 포즈99)."""
    people = kp_json.get("people", {})
    pose = people.get("pose_keypoints_2d", [])
    hand_l = people.get("hand_left_keypoints_2d", [])
    hand_r = people.get("hand_right_keypoints_2d", [])

    # 포즈: 25점×3=75에서 앞 33점(99차원) 사용 — 없으면 0 패딩
    pose_arr = np.array(pose[:75], dtype=np.float32) if len(pose) >= 75 else np.zeros(75, dtype=np.float32)
    hand_l_arr = np.array(hand_l[:63], dtype=np.float32) if len(hand_l) >= 63 else np.zeros(63, dtype=np.float32)
    hand_r_arr = np.array(hand_r[:63], dtype=np.float32) if len(hand_r) >= 63 else np.zeros(63, dtype=np.float32)

    vec = np.concatenate([hand_l_arr, hand_r_arr, pose_arr])  # 201차원
    if np.all(vec == 0):
        return None
    return vec


def collect_keypoints_by_word(keypoint_zip_path: Path) -> dict[str, list[np.ndarray]]:
    """keypoint zip에서 WORD번호별 프레임 벡터 리스트 수집."""
    word_frames: dict[str, list[np.ndarray]] = defaultdict(list)
    with zipfile.ZipFile(keypoint_zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith("_keypoints.json"):
                continue
            m = re.search(r"(WORD\d+)", name)
            if not m:
                continue
            word_key = m.group(1)
            with zf.open(name) as f:
                try:
                    data = json.load(f)
                    vec = extract_keypoint_vector(data)
                    if vec is not None:
                        word_frames[word_key].append(vec)
                except (json.JSONDecodeError, KeyError):
                    pass
    return word_frames


def main():
    with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
        gloss_list = json.load(f)

    needed_glosses = {g["gloss"] for g in gloss_list if not g.get("keypoint_path")}
    print(f"keypoint 미등록 글로스: {len(needed_glosses)}개")

    print("1단계: 형태소 라벨에서 단어 사전 구축 중...")
    word_dict = build_word_dict(MORPHEME_ZIP)
    print(f"   → {len(word_dict)}개 단어 매핑 완료")

    # 글로스 → WORD번호 역매핑
    gloss_to_word: dict[str, str] = {}
    for word_key, label in word_dict.items():
        if label in needed_glosses and label not in gloss_to_word:
            gloss_to_word[label] = word_key

    matched = set(gloss_to_word.keys())
    print(f"   → 220개 글로스 중 {len(matched)}개 매칭")
    print(f"   → 미매칭: {needed_glosses - matched}\n")

    if not matched:
        print("매칭된 글로스가 없습니다.")
        return

    print("2단계: keypoint zip에서 프레임 수집 중... (시간이 걸립니다)")
    word_frames = collect_keypoints_by_word(KEYPOINT_ZIP)
    print(f"   → {len(word_frames)}개 WORD 키포인트 수집 완료\n")

    print("3단계: 글로스별 평균 keypoint 저장 중...")
    updated = 0
    for g in gloss_list:
        gloss = g["gloss"]
        if g.get("keypoint_path") or gloss not in gloss_to_word:
            continue
        word_key = gloss_to_word[gloss]
        frames = word_frames.get(word_key, [])
        if not frames:
            print(f"   ⚠️  {gloss} ({word_key}): keypoint 프레임 없음")
            continue

        # 프레임 평균으로 대표 keypoint 생성
        avg_vec = np.mean(frames, axis=0)
        save_path = KEYPOINT_DIR / f"{gloss}.npy"
        np.save(str(save_path), avg_vec)

        g["keypoint_path"] = str(save_path)
        g["registered"] = True
        updated += 1
        print(f"   ✅ {gloss} ({word_key}): {len(frames)}프레임 → {save_path.name}")

    with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(gloss_list, f, ensure_ascii=False, indent=2)

    registered_total = sum(1 for g in gloss_list if g.get("keypoint_path"))
    print(f"\n========== 결과 ==========")
    print(f"이번 추가: {updated}개")
    print(f"전체 등록: {registered_total}/220개")


if __name__ == "__main__":
    main()
