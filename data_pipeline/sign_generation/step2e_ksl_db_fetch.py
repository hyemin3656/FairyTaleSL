"""
step2e_ksl_db_fetch.py
팀 ksl.mv.db (H2) → /tmp/ksl_signs.csv 를 활용해
- 기존 text-fallback 글로스 영상 수집
- DB에만 있는 신규 글로스도 gloss_list.json에 추가 + 영상 다운로드

실행 전제: H2 → CSV 추출 완료
  java -cp /tmp/h2.jar org.h2.tools.Shell \
    -url "jdbc:h2:/path/to/ksl;ACCESS_MODE_DATA=r" \
    -user sa -password "" \
    -sql "CALL CSVWRITE('/tmp/ksl_signs.csv','SELECT GLOSS,VIDEO_URL FROM SIGNS');"
"""
import csv
import json
import os
import requests
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
GLOSS_LIST_PATH = BASE_DIR / "data/gloss_list.json"
VIDEO_DIR = BASE_DIR / "data/videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = Path("/tmp/ksl_signs.csv")

if not CSV_PATH.exists():
    raise SystemExit(f"CSV 파일 없음: {CSV_PATH}")

# ── CSV 로드 ─────────────────────────────────────────────────
print("CSV 로드 중...")
db_entries: list[tuple[str, str]] = []  # (gloss, url)
db_map: dict[str, str] = {}
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gloss = row["GLOSS"].strip()
        url = row["VIDEO_URL"].strip().replace(
            "http://sldict.korean.go.kr",
            "https://sldict.korean.go.kr"
        )
        if gloss and url:
            db_entries.append((gloss, url))
            db_map[gloss] = url

print(f"DB 글로스: {len(db_map)}개\n")

# ── gloss_list.json 로드 ─────────────────────────────────────
with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
    gloss_list = json.load(f)

existing_glosses = {g["gloss"] for g in gloss_list}


def find_url_for_existing(gloss: str) -> tuple[str, str] | None:
    """기존 text-fallback 글로스 → (matched_db_gloss, url)"""
    if gloss in db_map:
        return gloss, db_map[gloss]
    if (gloss + "다") in db_map:
        return gloss + "다", db_map[gloss + "다"]
    if (gloss + "하다") in db_map:
        return gloss + "하다", db_map[gloss + "하다"]
    for db_gloss, url in db_map.items():
        clean = db_gloss.split(")")[-1].strip() if ")" in db_gloss else db_gloss
        if clean == gloss or clean == gloss + "다":
            return db_gloss, url
    return None


def download_video(url: str, save_path: Path) -> bool:
    try:
        res = requests.get(url, stream=True, timeout=30)
        with open(save_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
        if os.path.getsize(save_path) < 1000:
            os.remove(save_path)
            return False
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


# ── 1단계: 기존 text-fallback 글로스 처리 ───────────────────
print("=" * 40)
print("1단계: 기존 text-fallback 글로스 수집")
print("=" * 40)

targets = [g for g in gloss_list if g.get("fallback_type") == "text" and not g.get("video_path")]
gloss_by_key = {g["gloss"]: g for g in gloss_list}
success1, fail1, not_found1 = 0, 0, []

for g in targets:
    word = g["gloss"]
    result = find_url_for_existing(word)
    if not result:
        not_found1.append(word)
        continue

    matched, url = result
    save_path = VIDEO_DIR / f"{word}.mp4"
    if save_path.exists():
        g["video_path"] = str(save_path.resolve())
        g["registered"] = True
        g["fallback_type"] = None
        success1 += 1
        continue

    print(f"  {word} (← {matched})", end=" ... ")
    if download_video(url, save_path):
        print("✅")
        g["video_path"] = str(save_path.resolve())
        g["registered"] = True
        g["fallback_type"] = None
        success1 += 1
    else:
        print("❌")
        fail1 += 1
    time.sleep(0.15)

print(f"\n1단계 완료: 성공 {success1}, 실패 {fail1}, 미발견 {len(not_found1)}\n")

# ── 2단계: DB 신규 글로스 전체 추가 ─────────────────────────
print("=" * 40)
print("2단계: DB 신규 글로스 추가")
print("=" * 40)

# 괄호 있는 글로스(설명형)는 skip — 단순 단어만 추가
def is_clean_gloss(gloss: str) -> bool:
    return "(" not in gloss and ")" not in gloss and len(gloss) <= 10

new_count = 0
skip_count = 0
fail2 = 0

for db_gloss, url in db_entries:
    # 이미 있는 글로스는 video_path만 업데이트
    if db_gloss in existing_glosses:
        entry = gloss_by_key.get(db_gloss)
        if entry and not entry.get("video_path"):
            save_path = VIDEO_DIR / f"{db_gloss}.mp4"
            if not save_path.exists():
                print(f"  [업데이트] {db_gloss}", end=" ... ")
                if download_video(url, save_path):
                    print("✅")
                    entry["video_path"] = str(save_path.resolve())
                    entry["registered"] = True
                    entry["fallback_type"] = None
                else:
                    print("❌")
                    fail2 += 1
                time.sleep(0.15)
        continue

    if not is_clean_gloss(db_gloss):
        skip_count += 1
        continue

    # 신규 글로스 추가
    save_path = VIDEO_DIR / f"{db_gloss}.mp4"
    print(f"  [신규] {db_gloss}", end=" ... ")

    if save_path.exists() or download_video(url, save_path):
        if not save_path.exists():
            print("❌")
            fail2 += 1
            continue
        print("✅")
        gloss_list.append({
            "gloss": db_gloss,
            "pos": "Unknown",
            "freq": 0,
            "registered": True,
            "video_path": str(save_path.resolve()),
            "motion_path": None,
            "emotion_label": None,
            "keypoint_path": None,
            "fallback_type": None,
            "fallback_glosses": [],
        })
        existing_glosses.add(db_gloss)
        gloss_by_key[db_gloss] = gloss_list[-1]
        new_count += 1
    else:
        print("❌")
        fail2 += 1
    time.sleep(0.15)

# ── 저장 ─────────────────────────────────────────────────────
with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
    json.dump(gloss_list, f, ensure_ascii=False, indent=2)

print(f"\n{'='*40}")
print(f"1단계 text-fallback 수집: {success1}개 성공")
print(f"2단계 신규 글로스 추가:   {new_count}개 (건너뜀: {skip_count}개, 실패: {fail2}개)")
print(f"전체 글로스: {len(gloss_list)}개")
print(f"\n다음 단계: python step3_extract_keypoints.py")
