"""
step2f_fetch_missing.py
gloss_list.json의 미등록 글로스를 국립국어원 수어사전에서 직접 수집.

흐름:
  1. 검색 페이지 → fnSearchContentsView(origin_no, category) 추출
  2. 상세 페이지 → preview 이미지 URL (_105X105.jpg)
  3. _105X105.jpg → _320X240.mp4 치환으로 영상 URL 도출
  4. 영상 다운로드 → gloss_list.json 업데이트
"""
import json, re, time, os
import urllib.request, urllib.parse
from pathlib import Path

BASE_DIR = Path(__file__).parent
GLOSS_LIST_PATH = BASE_DIR / "data/gloss_list.json"
VIDEO_DIR = BASE_DIR / "data/videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://sldict.korean.go.kr",
    "Accept": "text/html,application/xhtml+xml,*/*",
}


def fetch(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5)
            else:
                raise e


def search_origin_nos(word: str) -> list[tuple[str, str]]:
    """검색 페이지에서 (origin_no, category) 목록 반환."""
    params = urllib.parse.urlencode(
        {"searchKeyword": word, "gubun": "W", "pageIndex": "1", "pageSize": "10"}
    )
    text = fetch(f"https://sldict.korean.go.kr/front/search/searchAllList.do?{params}")
    return re.findall(r"fnSearchContentsView\s*\(\s*'(\d+)'\s*,\s*'(\w+)'", text)


def get_video_url(origin_no: str, category: str, word: str) -> str | None:
    """상세 페이지 preview 이미지 URL → 영상 URL."""
    detail_url = (
        f"https://sldict.korean.go.kr/front/sign/signContentsView.do"
        f"?origin_no={origin_no}&top_gubun={category}"
        f"&searchKeyword={urllib.parse.quote(word)}"
    )
    text = fetch(detail_url)
    previews = re.findall(
        r'id="preview"\s+value="([^"]+_105X105\.jpg)"', text
    )
    if not previews:
        previews = re.findall(r'(http[^"\']+_105X105\.jpg)', text)
    if not previews:
        return None
    return previews[0].replace("_105X105.jpg", "_320X240.mp4").replace(
        "http://sldict.korean.go.kr", "https://sldict.korean.go.kr"
    )


def download(url: str, save_path: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) < 1000:
            return False
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"    [DL ERROR] {e}")
        return False


def save(gloss_list: list) -> None:
    with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(gloss_list, f, ensure_ascii=False, indent=2)


def main():
    with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
        gloss_list = json.load(f)

    gloss_map = {g["gloss"]: g for g in gloss_list}

    # 이미 다운로드된 파일이 있으면 gloss_list에 반영 (크래시 복구)
    recovered = 0
    for vid_file in VIDEO_DIR.glob("*.mp4"):
        word = vid_file.stem
        if word in gloss_map and not gloss_map[word].get("video_path"):
            gloss_map[word]["registered"] = True
            gloss_map[word]["video_path"] = str(vid_file.resolve())
            gloss_map[word]["fallback_type"] = None
            recovered += 1
    if recovered:
        print(f"[복구] 기존 다운로드 파일 {recovered}개 반영")
        save(gloss_list)

    # 미등록 글로스만 처리 (video_path가 없는 것)
    targets = [
        g for g in gloss_list
        if not g.get("video_path") and not g.get("registered")
    ]
    print(f"수집 대상: {len(targets)}개\n")

    success, fail, not_found = [], [], []

    for g in targets:
        word = g["gloss"]
        # 사전형(어간+다) 포함 검색 시도
        search_words = [word, word + "다"] if not word.endswith("다") else [word]

        found = False
        for sw in search_words:
            candidates = search_origin_nos(sw)
            if not candidates:
                continue

            # CTE(일상생활) 우선, 없으면 SPE(전문용어)
            ordered = sorted(candidates, key=lambda x: (0 if x[1] == "CTE" else 1))
            # 중복 제거
            seen, unique = set(), []
            for c in ordered:
                if c[0] not in seen:
                    seen.add(c[0])
                    unique.append(c)

            for origin_no, category in unique[:2]:
                vid_url = get_video_url(origin_no, category, sw)
                if not vid_url:
                    continue

                save_path = VIDEO_DIR / f"{word}.mp4"
                if save_path.exists():
                    print(f"  [SKIP] {word} (이미 존재)")
                    g["registered"] = True
                    g["video_path"] = str(save_path.resolve())
                    g["fallback_type"] = None
                    found = True
                    success.append(word)
                    break

                print(f"  {word} (origin={origin_no})", end=" ... ")
                if download(vid_url, save_path):
                    print("✅")
                    g["registered"] = True
                    g["video_path"] = str(save_path.resolve())
                    g["fallback_type"] = None
                    found = True
                    success.append(word)
                    save(gloss_list)  # 즉시 저장
                    break
                else:
                    print("❌")

            if found:
                break
            time.sleep(0.3)

        if not found:
            not_found.append(word)
            print(f"  {word}: 사전에서 찾지 못함")

        time.sleep(0.4)

    with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(gloss_list, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*40}")
    print(f"성공: {len(success)}개 → {success}")
    print(f"실패: {len(fail)}개")
    print(f"미발견: {len(not_found)}개 → {not_found}")


if __name__ == "__main__":
    main()
