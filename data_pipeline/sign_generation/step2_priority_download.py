"""
step2_priority_download.py
동화 핵심 글로스 346개를 KCISA API에서 우선 다운로드.

1단계: 우선순위 346개 (동화 직접 사용 글로스)
2단계: 나머지 6000+ 개 (전체 파이프라인)
"""
import json
import time
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

BASE_DIR       = Path(__file__).parent
GLOSS_LIST_PATH = BASE_DIR / "data/gloss_list.json"
VIDEO_DIR      = BASE_DIR / "data/videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = "f12f5fa4-0583-4276-8226-62b3748ea583"
API_URL = "https://api.kcisa.kr/openapi/service/rest/meta13/getCTE01701"

PRIORITY_PATH = Path("/tmp/fairy_tale_glosses.json")

# ── 우선순위 글로스 로드 ──────────────────────────────────────
with open(PRIORITY_PATH, encoding="utf-8") as f:
    priority_glosses: list[str] = json.load(f)

with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
    gloss_list: list[dict] = json.load(f)

gloss_by_key = {g["gloss"]: g for g in gloss_list}
existing_glosses = set(gloss_by_key)

# 이미 video_path 있는 것 제외
needed = {
    g for g in priority_glosses
    if not gloss_by_key.get(g, {}).get("video_path")
}
print(f"우선순위 글로스: {len(priority_glosses)}개")
print(f"영상 필요:      {len(needed)}개\n")

if not needed:
    print("모두 이미 다운로드됨.")
    raise SystemExit(0)


# ── KCISA API 전체 스캔 → URL 맵 ─────────────────────────────
def fetch_url_map(needed_set: set[str]) -> dict[str, str]:
    """필요한 글로스의 video_url을 KCISA API에서 수집."""
    url_map: dict[str, str] = {}
    page = 1
    total_pages = None

    while True:
        params = {"serviceKey": API_KEY, "numOfRows": 100, "pageNo": page}
        try:
            res = requests.get(API_URL, params=params, timeout=15)
            root = ET.fromstring(res.text)
        except Exception as e:
            print(f"  [ERROR] page {page}: {e}")
            break

        if total_pages is None:
            try:
                total_count = int(root.findtext(".//totalCount") or 0)
                total_pages = (total_count + 99) // 100
                print(f"전체 {total_count}건 ({total_pages}페이지)")
            except Exception:
                pass

        items = root.findall(".//item")
        if not items:
            break

        for item in items:
            title = item.findtext("title", "").strip()
            video_url = item.findtext("subDescription", "").strip()
            if not title or not video_url:
                continue
            video_url = video_url.replace(
                "http://sldict.korean.go.kr",
                "https://sldict.korean.go.kr"
            )
            for word in title.split(","):
                word = word.strip()
                if word in needed_set and word not in url_map:
                    url_map[word] = video_url

        if page % 10 == 0:
            print(f"  page {page}/{total_pages or '?'} | 매핑: {len(url_map)}/{len(needed_set)}")

        # 필요한 글로스 다 찾으면 조기 종료
        if url_map.keys() >= needed_set:
            print("  → 필요 글로스 모두 찾음, 조기 종료")
            break

        if total_pages and page >= total_pages:
            break
        page += 1
        time.sleep(0.1)

    return url_map


# ── 비디오 다운로드 ───────────────────────────────────────────
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
        print(f"  [ERROR] {url}: {e}")
        return False


# ── 실행 ──────────────────────────────────────────────────────
print("=" * 50)
print("KCISA API 스캔 중...")
print("=" * 50)
url_map = fetch_url_map(needed)
print(f"\n URL 매핑 완료: {len(url_map)}개 / {len(needed)}개\n")

not_found = needed - url_map.keys()
if not_found:
    print(f"API에서 못 찾은 글로스 ({len(not_found)}개): {sorted(not_found)}\n")

# 다운로드
print("=" * 50)
print("영상 다운로드 중...")
print("=" * 50)
success, fail = 0, 0

for word, url in url_map.items():
    save_path = VIDEO_DIR / f"{word}.mp4"

    if save_path.exists() and save_path.stat().st_size > 1000:
        # 기존 파일 활용
        entry = gloss_by_key.get(word)
        if entry:
            entry["video_path"] = str(save_path.resolve())
        else:
            gloss_list.append({
                "gloss": word, "pos": "Unknown", "freq": 0, "registered": False,
                "video_path": str(save_path.resolve()), "motion_path": None,
                "emotion_label": None, "keypoint_path": None,
                "fallback_type": None, "fallback_glosses": [],
            })
            gloss_by_key[word] = gloss_list[-1]
        success += 1
        continue

    print(f"  {word}", end=" ... ", flush=True)
    if download_video(url, save_path):
        print(f"✅ ({save_path.stat().st_size // 1024}KB)")
        entry = gloss_by_key.get(word)
        if entry:
            entry["video_path"] = str(save_path.resolve())
            entry["fallback_type"] = None
        else:
            gloss_list.append({
                "gloss": word, "pos": "Unknown", "freq": 0, "registered": False,
                "video_path": str(save_path.resolve()), "motion_path": None,
                "emotion_label": None, "keypoint_path": None,
                "fallback_type": None, "fallback_glosses": [],
            })
            gloss_by_key[word] = gloss_list[-1]
        success += 1
    else:
        print("❌")
        fail += 1
    time.sleep(0.12)

# gloss_list 저장
with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
    json.dump(gloss_list, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f"완료: 성공 {success}개 / 실패 {fail}개 / 미발견 {len(not_found)}개")
print(f"전체 글로스: {len(gloss_list)}개")
print(f"\n다음 단계: python step3_extract_keypoints.py")
