"""
RunYourAI API 기반 감정 레이블링 (전체 20편 동화, step4 대체).

문장 단위 감정 분류 → 글로스별 최빈 감정 결정
→ gloss_list.json + motion_db.sqlite 업데이트

감정 레이블: 기쁨 | 슬픔 | 분노 | 놀람 | 중립
(seed.py의 _EMOTION_MAP이 영문으로 변환 후 PostgreSQL에 적재)
"""
import json
import sqlite3
import time
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

BASE_DIR         = Path(__file__).parent
GLOSS_LIST_PATH  = BASE_DIR / "data/gloss_list.json"
FAIRY_TALES_PATH = BASE_DIR / "data/fairy_tales_structured.json"
DB_PATH          = BASE_DIR / "data/motion_db.sqlite"
PROGRESS_PATH    = BASE_DIR / "data/step4b_progress.json"

RUNYOURAI_API_KEY = "runyour-v1-23856ca2-645c-45fe-bc7f-e345e2be20b0-PJbcr10QfA_AgWqKDrwQoAmZByj0BJ1UlEyrK_0kFhOi1d__zlTx6zvbW_IUOc74"
RUNYOURAI_BASE_URL = "https://api.runyour.ai/v1"
MODEL = "anthropic/claude-haiku-4-5"

BATCH_SIZE  = 12   # 한 번의 API 호출당 처리 문장 수
BATCH_DELAY = 0.4  # 배치 간 대기 시간(초) — 레이트 리밋 방지
MAX_RETRY   = 3    # API 실패 시 재시도 횟수

VALID_LABELS  = {"기쁨", "슬픔", "분노", "놀람", "중립"}
DEFAULT_LABEL = "중립"

_SYSTEM = (
    "당신은 동화 문장의 감정을 분류하는 전문가입니다. "
    "주어진 JSON 스키마를 정확히 따라 응답하세요."
)

_USER_TEMPLATE = """동화 "{title}"의 문장들을 아래 5가지 감정 중 하나로 분류하세요.

감정 기준:
- 기쁨: 행복, 즐거움, 설렘, 감사, 승리, 긍정적 결말
- 슬픔: 슬픔, 이별, 그리움, 눈물, 상실, 절망
- 분노: 화남, 위협, 갈등, 억울함, 공격, 배신
- 놀람: 예상 밖 사건, 신기함, 충격, 두려움, 긴장
- 중립: 일반 설명, 배경 서술, 평범한 행동 묘사

문장 목록:
{sentences}

JSON만 반환 (다른 텍스트 없이):
{{"emotions": ["레이블1", "레이블2", ...]}}"""


def _call_api(sentences: list[str], tale_title: str, retry: int = 0) -> list[str]:
    """RunYourAI API 호출 → 감정 레이블 리스트 반환. 실패 시 재시도."""
    numbered = "\n".join(f'{i + 1}. "{s}"' for i, s in enumerate(sentences))
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER_TEMPLATE.format(
                title=tale_title, sentences=numbered
            )},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
        "response_format": "json_object",
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{RUNYOURAI_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {RUNYOURAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        content = data["choices"][0]["message"]["content"]
        # 마크다운 코드블록 제거 후 JSON 파싱
        if isinstance(content, str):
            text = content.strip()
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            parsed = json.loads(text)
        else:
            parsed = content
        emotions = parsed.get("emotions", [])

        # 유효성 검사 및 길이 보정
        result = [
            e if e in VALID_LABELS else DEFAULT_LABEL
            for e in emotions
        ]
        while len(result) < len(sentences):
            result.append(DEFAULT_LABEL)
        return result[:len(sentences)]

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"    HTTP {e.code}: {body[:120]}")
        if retry < MAX_RETRY and e.code in (429, 500, 502, 503):
            time.sleep(2 ** retry)
            return _call_api(sentences, tale_title, retry + 1)
        return [DEFAULT_LABEL] * len(sentences)

    except Exception as e:
        print(f"    API 오류: {e}")
        if retry < MAX_RETRY:
            time.sleep(2 ** retry)
            return _call_api(sentences, tale_title, retry + 1)
        return [DEFAULT_LABEL] * len(sentences)


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_tales": [], "sentence_emotions": {}}


def _save_progress(progress: dict) -> None:
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def classify_all_tales(fairy_tales: list) -> dict[str, str]:
    """모든 동화 문장 감정 분류. 중단 후 재시작 시 완료된 동화는 건너뜀.

    반환: {"{tale_id}_{seg_id}_{sent_id}": emotion}
    """
    progress = _load_progress()
    sentence_emotions: dict[str, str] = progress.get("sentence_emotions", {})
    completed_tales: set[str] = set(progress.get("completed_tales", []))

    for tale in fairy_tales:
        tale_id = tale["tale_id"]

        if tale_id in completed_tales:
            print(f"  [{tale['title']}] 이미 완료 (건너뜀)")
            continue

        # 문장 수집 (sentence_id 없는 동화는 enumerate 인덱스 사용)
        sentences, sent_keys = [], []
        global_idx = 0
        for seg in tale["segments"]:
            for s in seg["sentences"]:
                sid = s.get("sentence_id", global_idx)
                key = f"{tale_id}_{seg['segment_id']}_{sid}"
                sentences.append(s["text"])
                sent_keys.append(key)
                global_idx += 1

        print(f"  [{tale['title']}] {len(sentences)}문장 처리 중...")

        for i in range(0, len(sentences), BATCH_SIZE):
            batch_sents = sentences[i:i + BATCH_SIZE]
            batch_keys  = sent_keys[i:i + BATCH_SIZE]
            emotions    = _call_api(batch_sents, tale["title"])

            for key, emo in zip(batch_keys, emotions):
                sentence_emotions[key] = emo

            end = min(i + BATCH_SIZE, len(sentences))
            print(f"    {end}/{len(sentences)}: {emotions}")

            if end < len(sentences):
                time.sleep(BATCH_DELAY)

        completed_tales.add(tale_id)
        progress["completed_tales"] = list(completed_tales)
        progress["sentence_emotions"] = sentence_emotions
        _save_progress(progress)
        print(f"  [{tale['title']}] 완료 ✓\n")

    return sentence_emotions


def build_gloss_emotion_map(fairy_tales: list, sentence_emotions: dict) -> dict[str, str]:
    """문장 감정 → 글로스별 최빈 감정 매핑 반환."""
    gloss_emotions: dict[str, list[str]] = {}
    for tale in fairy_tales:
        global_idx = 0
        for seg in tale["segments"]:
            for s in seg["sentences"]:
                sid    = s.get("sentence_id", global_idx)
                key    = f"{tale['tale_id']}_{seg['segment_id']}_{sid}"
                emotion = sentence_emotions.get(key, DEFAULT_LABEL)
                for gloss in s.get("ksl_glosses", []):
                    gloss_emotions.setdefault(gloss, []).append(emotion)
                global_idx += 1

    return {
        gloss: Counter(emos).most_common(1)[0][0]
        for gloss, emos in gloss_emotions.items()
    }


def update_gloss_list(
    gloss_list: list, emotion_map: dict[str, str]
) -> tuple[int, int]:
    """gloss_list 항목의 emotion_label 갱신. (업데이트 수, 기본값 유지 수) 반환."""
    assigned = default = 0
    for g in gloss_list:
        emotion = emotion_map.get(g["gloss"])
        if emotion:
            g["emotion_label"] = emotion
            assigned += 1
        else:
            if not g.get("emotion_label"):
                g["emotion_label"] = DEFAULT_LABEL
            default += 1
    return assigned, default


def update_motion_db(emotion_map: dict[str, str]) -> int:
    """motion_db.sqlite emotion_label 일괄 업데이트. 업데이트된 행 수 반환."""
    if not DB_PATH.exists():
        print("  motion_db.sqlite 없음 — 건너뜀")
        return 0
    conn = sqlite3.connect(str(DB_PATH))
    updated = 0
    for gloss, emotion in emotion_map.items():
        cur = conn.execute(
            "UPDATE motion_db SET emotion_label = ? WHERE gloss = ?",
            (emotion, gloss),
        )
        updated += cur.rowcount
    conn.commit()
    conn.close()
    return updated


def main() -> None:
    with open(FAIRY_TALES_PATH, encoding="utf-8") as f:
        fairy_tales = json.load(f)
    with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
        gloss_list = json.load(f)

    print(f"=== RunYourAI 감정 레이블링 시작 ===")
    print(f"동화: {len(fairy_tales)}편 / 글로스: {len(gloss_list)}개\n")

    # 1. 문장 감정 분류
    sentence_emotions = classify_all_tales(fairy_tales)

    # 2. 글로스별 최빈 감정 계산
    print("글로스별 최빈 감정 계산 중...")
    emotion_map = build_gloss_emotion_map(fairy_tales, sentence_emotions)
    print(f"  → {len(emotion_map)}개 글로스 감정 결정됨\n")

    # 3. gloss_list.json 업데이트
    print("gloss_list.json 업데이트 중...")
    assigned, default = update_gloss_list(gloss_list, emotion_map)
    with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(gloss_list, f, ensure_ascii=False, indent=2)
    print(f"  → 저장 완료 (RunYourAI 매핑: {assigned}개, 기존값 유지: {default}개)\n")

    # 4. motion_db.sqlite 업데이트
    print("motion_db.sqlite 업데이트 중...")
    db_updated = update_motion_db(emotion_map)
    print(f"  → {db_updated}개 레코드 업데이트\n")

    # 5. 최종 통계
    emotion_counts = Counter(
        g["emotion_label"] for g in gloss_list if g.get("emotion_label")
    )
    print("========== 결과 ==========")
    print(f"RunYourAI 매핑: {assigned}개 / 기존값 유지: {default}개")
    print("\n감정 분포 (gloss_list.json):")
    for emotion, cnt in emotion_counts.most_common():
        pct = cnt / len(gloss_list) * 100
        print(f"  {emotion:<4}: {cnt:>5}개 ({pct:.1f}%)")

    db_counts: list[tuple] = []
    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        db_counts = conn.execute(
            "SELECT emotion_label, COUNT(*) FROM motion_db GROUP BY emotion_label ORDER BY 2 DESC"
        ).fetchall()
        conn.close()
    if db_counts:
        print("\n감정 분포 (motion_db.sqlite):")
        for emotion, cnt in db_counts:
            print(f"  {str(emotion):<4}: {cnt:>5}개")

    print(f"\n다음 단계: backend/scripts/seed.py 재실행 → PostgreSQL emotion_label 영문 반영")
    print(f"진행 파일: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()
