"""
step2c_retry_fallbacks.py
text-fallback 글로스를 KCISA API로 재수집하여 video_path 업데이트

실행: python step2c_retry_fallbacks.py
"""
import requests
import json
import time
import os
import xml.etree.ElementTree as ET
from pathlib import Path

API_KEY = "f12f5fa4-0583-4276-8226-62b3748ea583"
API_URL = "https://api.kcisa.kr/openapi/service/rest/meta13/getCTE01701"
BASE_DIR = Path(__file__).parent
GLOSS_LIST_PATH = BASE_DIR / "data/gloss_list.json"
VIDEO_DIR = BASE_DIR / "data/videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
    gloss_list = json.load(f)

# text fallback이면서 아직 video_path 없는 글로스만 대상
targets = {g["gloss"] for g in gloss_list if g.get("fallback_type") == "text" and not g.get("video_path")}
print(f"재수집 대상: {len(targets)}개\n")

# ── KCISA 전체 스캔 ───────────────────────────────────────────────
def fetch_sign_url_map(needed: set[str]) -> dict[str, str]:
    gloss_url_map: dict[str, str] = {}
    page = 1

    while True:
        params = {
            "serviceKey": API_KEY,
            "numOfRows": 100,
            "pageNo": page,
        }
        try:
            res = requests.get(API_URL, params=params, timeout=15)
            root = ET.fromstring(res.text)
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
                    if word in needed and word not in gloss_url_map:
                        gloss_url_map[word] = video_url
                        print(f"  ✅ 찾음: {word}")

            total = int(root.findtext(".//totalCount", "0"))
            print(f"  page {page} ({len(items)}건) | 누적: {len(gloss_url_map)}개")

            if gloss_url_map.keys() >= needed:
                print("  → 필요한 글로스 모두 찾음")
                break
            if page * 100 >= total:
                break

            page += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"  [ERROR] page {page}: {e}")
            break

    return gloss_url_map


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
        print(f"  [ERROR] 다운로드 실패: {e}")
        return False


# ── 실행 ─────────────────────────────────────────────────────────
print("API 스캔 중...\n")
url_map = fetch_sign_url_map(targets)

found     = set(url_map.keys())
not_found = targets - found
print(f"\n찾음: {sorted(found)}")
print(f"미발견: {sorted(not_found)}\n")

success_count = 0
fail_count = 0

gloss_by_key = {g["gloss"]: g for g in gloss_list}

for word, url in url_map.items():
    save_path = VIDEO_DIR / f"{word}.mp4"
    print(f"다운로드: {word}", end=" ... ")
    ok = download_video(url, save_path)
    if ok:
        print("✅")
        g = gloss_by_key[word]
        g["video_path"] = str(save_path.resolve())
        g["registered"] = True
        g["fallback_type"] = None   # text fallback 해제
        success_count += 1
    else:
        print("❌")
        fail_count += 1

# 결과 저장
with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
    json.dump(gloss_list, f, ensure_ascii=False, indent=2)

print(f"\n{'='*40}")
print(f"✅ 수집 성공: {success_count}개")
print(f"❌ 다운로드 실패: {fail_count}개")
print(f"⚠️  API 미발견 (유사도 매핑 대상): {len(not_found)}개")
if not_found:
    print(f"   → {sorted(not_found)}")
print(f"\n다음 단계: python step2c_keypoints.py (keypoint 추출)")
