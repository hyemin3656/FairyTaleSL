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

needed = {g["gloss"] for g in gloss_list if not g.get("video_path")}
print(f"수집 필요 글로스: {len(needed)}개\n")

def fetch_all_signs():
    gloss_url_map = {}
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

                # http → https 변환 (서버 마이그레이션 대응)
                video_url = video_url.replace("http://sldict.korean.go.kr", "https://sldict.korean.go.kr")

                # 쉼표로 구분된 여러 단어 처리
                for word in title.split(","):
                    word = word.strip()
                    if word in needed and word not in gloss_url_map:
                        gloss_url_map[word] = video_url
                        print(f"  ✅ 찾음: {word}")

            print(f"  page {page} 완료 ({len(items)}건) | 누적 매핑: {len(gloss_url_map)}개")

            # 필요한 글로스 다 찾으면 조기 종료
            if gloss_url_map.keys() >= needed:
                print("  → 필요한 글로스 모두 찾음, 조기 종료")
                break

            # 전체 건수 확인
            total = int(root.findtext(".//totalCount", "0"))
            if page * 100 >= total:
                break

            page += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"[ERROR] page {page}: {e}")
            break

    return gloss_url_map

def download_video(url, save_path):
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

# 전체 API 스캔
print("API 스캔 중...\n")
gloss_url_map = fetch_all_signs()

print(f"\n찾은 글로스: {list(gloss_url_map.keys())}")
print(f"없는 글로스: {needed - gloss_url_map.keys()}\n")

# 영상 다운로드
for g in gloss_list:
    if g.get("video_path"):
        continue
    word = g["gloss"]
    if word not in gloss_url_map:
        print(f"⚠️  {word}: API에 없음 (fallback 필요)")
        continue

    print(f"다운로드 중: {word}", end=" ... ")
    save_path = VIDEO_DIR / f"{word}.mp4"
    success = download_video(gloss_url_map[word], str(save_path))
    if success:
        print("✅ 완료")
        g["registered"] = True
        g["video_path"] = str(save_path.resolve())
    else:
        print("❌ 실패")

# 결과 저장
with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
    json.dump(gloss_list, f, ensure_ascii=False, indent=2)

registered = [g for g in gloss_list if g.get("video_path")]
missing = [g for g in gloss_list if not g.get("video_path")]
print(f"\n========== 결과 요약 ==========")
print(f"✅ 수집 완료: {len(registered)}개")
print(f"⚠️  누락: {len(missing)}개")
if missing:
    print(f"   누락 글로스: {[g['gloss'] for g in missing]}")