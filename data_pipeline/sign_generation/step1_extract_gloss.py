"""
Step 1: 동화 텍스트 전처리 + 글로스 추출
- 형태소 분석: kiwipiepy (Java 없이 동작하는 순수 Python)
- KSL 어순 변환: 한국어 SOV → KSL 어순 (시간>장소>주어/목적어>부사>부정>서술어)
- 출력: data/gloss_list.json, data/fairy_tales_structured.json
"""

from kiwipiepy import Kiwi
from collections import Counter
import json
import os

from fairy_tales import FAIRY_TALES_STRUCTURED, get_all_sentences

kiwi = Kiwi()

# kiwipiepy POS 태그 prefix → 범주
VALID_POS_PREFIX = {"NNG", "NNP", "VV", "VA", "MAG"}
POS_LABEL = {
    "NNG": "Noun", "NNP": "Noun",
    "VV": "Verb", "VA": "Adjective", "MAG": "Adverb",
}

STOPWORDS = {
    "하다", "이다", "있다", "되다", "않다", "없다", "같다",
    "것", "수", "때", "중", "등", "위", "곳", "데",
    "나", "너", "우리", "그", "이", "저",
}

# KSL 어순 변환용 시간/장소 키워드
TIME_KEYWORDS = {
    "옛날", "어느날", "봄", "여름", "가을", "겨울",
    "아침", "저녁", "밤", "낮", "그때", "마지막",
    "처음", "나중", "결국", "어느", "아주",
}
PLACE_KEYWORDS = {
    "산", "연못", "하늘", "마을", "집", "굴",
    "세상", "결승선", "바다", "장", "나무", "수수밭",
}
NEG_KEYWORDS = {"않다", "못하다", "아니다", "없다", "안"}


# ── KSL 어순 변환 ──────────────────────────────────────────────

def _merge_compounds(tokens) -> list[tuple[str, str]]:
    """NNG/NNP + XSN(명사파생접미사) 연속 토큰을 복합명사로 합치기.
    예: 나무(NNG) + 꾼(XSN) → 나무꾼(NNG)
    """
    merged: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        tag = tokens[i].tag[:3]
        if (tag in ("NNG", "NNP")
                and i + 1 < len(tokens)
                and tokens[i + 1].tag[:3] == "XSN"):
            merged.append((tokens[i].form + tokens[i + 1].form, "NNG"))
            i += 2
        else:
            merged.append((tokens[i].form, tag))
            i += 1
    return merged


def convert_to_ksl_order(text: str) -> list[str]:
    """
    한국어 문장 → KSL 어순 글로스 시퀀스
    KSL 어순: 시간 > 장소 > 명사(주어/목적어) > 부사 > 부정 > 서술어(동사/형용사)
    - 조사(JK*), 어미(E*), 보조사 등 기능어 제거
    - 내용어(명사/동사/형용사/부사)만 남기고 재배열
    """
    result = kiwi.analyze(text)
    raw_tokens = result[0][0]
    tokens = _merge_compounds(raw_tokens)  # 복합명사 합치기

    time_words, place_words, nouns, verbs, adjs, advs, negs = [], [], [], [], [], [], []

    for word, pos3 in tokens:
        if len(word) <= 1 or word in STOPWORDS:
            continue

        if pos3 in ("NNG", "NNP"):
            if word in TIME_KEYWORDS:
                time_words.append(word)
            elif word in PLACE_KEYWORDS:
                place_words.append(word)
            else:
                nouns.append(word)
        elif pos3 == "VV":
            if word in NEG_KEYWORDS:
                negs.append(word)
            else:
                verbs.append(word)
        elif pos3 == "VA":
            adjs.append(word)
        elif pos3 == "MAG":
            if word in NEG_KEYWORDS:
                negs.append(word)
            else:
                advs.append(word)

    # KSL 어순: 시간 > 장소 > 명사 > 부사 > 부정 > 동사 > 형용사
    return time_words + place_words + nouns + advs + negs + verbs + adjs


# ── 글로스 추출 ────────────────────────────────────────────────

def extract_glosses(texts: list[str]) -> list[dict]:
    gloss_counter = Counter()
    gloss_pos_map = {}

    for text in texts:
        result = kiwi.analyze(text)
        tokens = _merge_compounds(result[0][0])
        for word, pos3 in tokens:
            if pos3 in VALID_POS_PREFIX and len(word) > 1 and word not in STOPWORDS:
                gloss_counter[word] += 1
                gloss_pos_map[word] = POS_LABEL.get(pos3, pos3)

    return [
        {
            "gloss": word,
            "pos": gloss_pos_map[word],
            "freq": count,
            "registered": False,
            "video_path": None,
            "motion_path": None,
            "emotion_label": None,
            "keypoint_path": None,
        }
        for word, count in gloss_counter.most_common()
    ]


def merge_with_existing(new_glosses: list[dict], existing_path: str) -> list[dict]:
    """기존 gloss_list.json 등록 정보 보존하며 병합"""
    if not os.path.exists(existing_path):
        return new_glosses

    with open(existing_path, encoding="utf-8") as f:
        existing = json.load(f)

    existing_map = {g["gloss"]: g for g in existing}
    new_words = set()

    for g in new_glosses:
        word = g["gloss"]
        new_words.add(word)
        if word in existing_map:
            old = existing_map[word]
            g["registered"]    = old.get("registered", False)
            g["video_path"]    = old.get("video_path")
            g["motion_path"]   = old.get("motion_path")
            g["emotion_label"] = old.get("emotion_label")
            g["keypoint_path"] = old.get("keypoint_path")

    # 새 텍스트에 없지만 등록된 글로스는 freq=0으로 유지
    for word, old in existing_map.items():
        if word not in new_words and old.get("registered"):
            merged_entry = dict(old)
            merged_entry["freq"] = 0
            new_glosses.append(merged_entry)

    return new_glosses


# ── 구조화 JSON 생성 ───────────────────────────────────────────

def build_structured_json() -> list[dict]:
    """
    동화 구조화 JSON 생성 (FR-016, FR-008용)
    각 문장에 KSL 글로스 시퀀스 추가
    """
    output = []
    for tale in FAIRY_TALES_STRUCTURED:
        tale_out = {
            "tale_id":    tale["tale_id"],
            "title":      tale["title"],
            "category":   tale["category"],
            "target_age": tale["target_age"],
            "segments":   []
        }
        sent_global_id = 1
        for seg in tale["segments"]:
            seg_out = {
                "segment_id": seg["segment_id"],
                "sentences":  []
            }
            for text in seg["sentences"]:
                ksl = convert_to_ksl_order(text)
                seg_out["sentences"].append({
                    "sentence_id": sent_global_id,
                    "text":        text,
                    "ksl_glosses": ksl,
                })
                sent_global_id += 1
            tale_out["segments"].append(seg_out)
        output.append(tale_out)
    return output


# ── 실행 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    sentences = get_all_sentences()
    print(f"전체 문장 수: {len(sentences)}")
    print(f"동화 수: {len(FAIRY_TALES_STRUCTURED)}편\n")

    # 1. 글로스 추출 및 병합
    gloss_path = "data/gloss_list.json"
    new_glosses = extract_glosses(sentences)
    merged = merge_with_existing(new_glosses, gloss_path)

    with open(gloss_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    registered = [g for g in merged if g["registered"]]
    print(f"글로스: {len(merged)}개 (등록 보존: {len(registered)}개)\n")

    # 2. KSL 어순 변환 + 구조화 JSON
    structured_path = "data/fairy_tales_structured.json"
    structured = build_structured_json()

    with open(structured_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f"구조화 JSON 저장: {structured_path}\n")

    # 샘플 출력
    print("=== KSL 어순 변환 샘플 ===")
    samples = [
        "옛날 옛날에 토끼와 거북이가 살았습니다.",
        "거북이는 느리지만 포기하지 않고 열심히 걸었습니다.",
        "호랑이는 참지 못하고 굴을 나왔지만 곰은 끝까지 참았습니다.",
        "심청이는 뱃사람들에게 공양미 삼백 석을 받고 인당수에 몸을 던졌습니다.",
        "산신령은 나무꾼의 정직함을 칭찬하며 세 도끼를 모두 주었습니다.",
    ]
    for s in samples:
        ksl = convert_to_ksl_order(s)
        print(f"  원문: {s}")
        print(f"  KSL:  {' > '.join(ksl)}\n")

    # 글로스 목록 상위 출력
    print(f"{'글로스':15s} {'품사':12s} {'빈도':>5s}  {'등록':>4s}")
    print("-" * 45)
    for g in merged[:30]:
        reg = "O" if g["registered"] else "-"
        print(f"  {g['gloss']:13s} ({g['pos']:10s}) freq={g['freq']:3d}  [{reg}]")
    if len(merged) > 30:
        print(f"  ... 외 {len(merged) - 30}개")
