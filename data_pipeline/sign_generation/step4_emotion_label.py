"""
KoBERT 기반 감정 레이블링 (FR-007, 목표 정확도 ≥80%).

모델: hun3359/klue-bert-base-sentiment (44클래스 한국어 감정 분류)
→ 5클래스로 매핑: 기쁨 | 슬픔 | 분노 | 놀람 | 중립

fairy_tales_structured.json의 8편 전체 문장에서 각 문장의 감정을 분류하고
글로스별 최빈 감정을 gloss_list.json에 저장.
"""
import json
from collections import Counter
from pathlib import Path

BASE_DIR          = Path(__file__).parent
GLOSS_LIST_PATH   = BASE_DIR / "data/gloss_list.json"
FAIRY_TALES_PATH  = BASE_DIR / "data/fairy_tales_structured.json"

# 44클래스 → 5클래스 매핑
_LABEL_MAP: dict[str, str] = {
    # 기쁨
    "신이 난":    "기쁨", "기쁨":      "기쁨", "행복한":    "기쁨",
    "설레는":     "기쁨", "즐거운":    "기쁨", "홀가분한":  "기쁨",
    "만족스러운": "기쁨", "기대하는":  "기쁨", "감사하는":  "기쁨",
    "자랑스러운": "기쁨", "안도":      "기쁨", "평온한":    "중립",
    # 슬픔
    "슬픔":       "슬픔", "외로운":    "슬픔", "그리운":    "슬픔",
    "상처받은":   "슬픔", "희생된":    "슬픔", "후회되는":  "슬픔",
    "무기력한":   "슬픔", "공허한":    "슬픔", "비참한":    "슬픔",
    "우울한":     "슬픔", "절망적인":  "슬픔", "안타까운":  "슬픔",
    # 분노
    "분노":       "분노", "화난":      "분노", "악의적인":  "분노",
    "짜증난":     "분노", "억울한":    "분노", "불안한":    "분노",
    "두려운":     "분노", "혐오스러운":"분노", "지루한":    "중립",
    # 놀람
    "당황스러운": "놀람", "놀란":      "놀람", "흥미로운":  "놀람",
    "신기한":     "놀람",
}
_DEFAULT_LABEL = "중립"


def _map_label(raw_label: str) -> str:
    return _LABEL_MAP.get(raw_label, _DEFAULT_LABEL)


def _load_classifier():
    from transformers import pipeline
    print("KoBERT 모델 로드 중...")
    clf = pipeline(
        "text-classification",
        model="hun3359/klue-bert-base-sentiment",
        device=-1,
        truncation=True,
        max_length=128,
    )
    print("  → 완료\n")
    return clf


def classify_sentences(sentences: list[str], clf) -> list[str]:
    """문장 리스트 → 감정 레이블 리스트 (배치 처리)."""
    results = clf(sentences, batch_size=16)
    return [_map_label(r["label"]) for r in results]


def build_gloss_emotion_map(fairy_tales: list, clf) -> dict[str, str]:
    """글로스별 최빈 감정 반환."""
    gloss_emotions: dict[str, list[str]] = {}

    for tale in fairy_tales:
        sentences, glosses_per_sent = [], []
        for seg in tale["segments"]:
            for s in seg["sentences"]:
                sentences.append(s["text"])
                glosses_per_sent.append(s.get("ksl_glosses", []))

        emotions = classify_sentences(sentences, clf)

        for glosses, emotion in zip(glosses_per_sent, emotions):
            for gloss in glosses:
                gloss_emotions.setdefault(gloss, []).append(emotion)

        print(f"  [{tale['title']}] {len(sentences)}문장 분류 완료")

    return {
        gloss: Counter(emos).most_common(1)[0][0]
        for gloss, emos in gloss_emotions.items()
    }


def main():
    with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
        gloss_list = json.load(f)
    with open(FAIRY_TALES_PATH, encoding="utf-8") as f:
        fairy_tales = json.load(f)

    clf = _load_classifier()
    gloss_emotion_map = build_gloss_emotion_map(fairy_tales, clf)

    assigned = default = 0
    for g in gloss_list:
        emotion = gloss_emotion_map.get(g["gloss"], "중립")
        g["emotion_label"] = emotion
        if g["gloss"] in gloss_emotion_map:
            assigned += 1
        else:
            default += 1

    with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(gloss_list, f, ensure_ascii=False, indent=2)

    emotion_counts = Counter(g["emotion_label"] for g in gloss_list)
    print(f"\n========== 결과 ==========")
    print(f"KoBERT 매핑: {assigned}개 / 기본값: {default}개")
    print("\n감정 분포:")
    for emotion, cnt in emotion_counts.most_common():
        print(f"  {emotion}: {cnt}개")


if __name__ == "__main__":
    main()
