"""
step2g_decompose_fallback.py
사전에 없는 글로스에 decompose 폴백 매핑 적용.
- text_display: 원본 단어 그대로 화면 표시
- fallback_type: "decompose"
- fallback_glosses: 의미를 전달하는 등록 글로스 목록
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
GLOSS_LIST_PATH = BASE_DIR / "data/gloss_list.json"

# 미등록 단어 → 의미 기반 decompose 매핑
# 규칙: fallback_glosses는 반드시 registered=True인 어휘만 사용
DECOMPOSE_MAP = {
    # ── 고유명사 (캐릭터명) ──────────────────────────────────
    "나무꾼":    ["나무", "남자"],
    "콩쥐":      ["여자", "착하다"],
    "팥쥐":      ["여자", "나쁘다"],
    "심청이":    ["여자", "부모", "사랑"],
    "심학규":    ["아버지", "장님"],
    "놀부":      ["형", "욕심"],
    "환웅":      ["왕", "하늘"],
    "단군왕검":  ["왕", "나라"],
    "고조선":    ["나라", "한국"],
    "태백산":    ["산"],
    "인당수":    ["바다"],
    "우리나라":  ["한국"],
    "산신령":    ["산", "신"],

    # ── 사물/장소 ────────────────────────────────────────────
    "도끼":      ["나무", "자르다"],
    "금도끼":    ["금", "자르다"],
    "쇠도끼":    ["자르다"],
    "동아줄":    ["줄"],
    "날개옷":    ["날개", "옷"],
    "두레박":    ["물", "위"],
    "해님":      ["해"],
    "달님":      ["달"],
    "금은보화":  ["금", "돈"],
    "오물":      ["더럽다"],
    "연못가":    ["연못"],
    "수탉":      ["닭"],
    "밀가루":    ["가루"],
    "교훈":      ["배우다"],

    # ── 복수형/집합 ──────────────────────────────────────────
    "아이들":    ["아이"],
    "동물들":    ["동물"],
    "뱃사람들":  ["배", "사람"],
    "선녀들":    ["선녀"],
    "사람들":    ["사람"],

    # ── 동사/형용사 ──────────────────────────────────────────
    "눈멀":      ["장님"],
    "쏟아지":    ["나오다"],
    "빠뜨리":    ["떨어지다"],
    "돌아가시":  ["죽다"],
    "돌려보내":  ["보내다"],
    "떠다니":    ["물", "위"],
    "거느리":    ["이끌다"],
    "찾아오":    ["오다"],
    "찾아가":    ["가다"],
    "살아가":    ["살다"],
    "잡아먹":    ["먹다"],
    "쫓기":      ["달리다", "도망치다"],

    # ── 부사/기타 ────────────────────────────────────────────
    "정성껏":    ["열심히"],
    "오래도록":  ["오래"],
    "그때":      ["때"],
    "욕심쟁이":  ["욕심"],
    "꼭대기":    ["위", "산"],
    "원님":      ["왕"],
    "임금님":    ["왕"],
    "용왕":      ["왕", "물"],
    "효심":      ["부모", "사랑"],
    "효도":      ["부모", "사랑"],
}


def main():
    with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
        gloss_list = json.load(f)

    registered = {g["gloss"] for g in gloss_list if g.get("registered")}
    gloss_map = {g["gloss"]: g for g in gloss_list}

    applied, skipped, missing_targets = [], [], []

    for word, decomp in DECOMPOSE_MAP.items():
        # decompose 대상 글로스가 등록됐는지 검증
        bad = [d for d in decomp if d not in registered and d + "다" not in registered]
        if bad:
            print(f"⚠️  {word}: fallback 글로스 미등록 → {bad}")
            missing_targets.extend(bad)

        if word in gloss_map:
            g = gloss_map[word]
            if not g.get("registered"):
                g["fallback_type"] = "decompose"
                g["fallback_glosses"] = decomp
                applied.append(word)
        else:
            # gloss_list에 없는 경우 신규 추가
            gloss_list.append({
                "gloss": word,
                "pos": "Unknown",
                "freq": 0,
                "registered": False,
                "video_path": None,
                "motion_path": None,
                "emotion_label": None,
                "keypoint_path": None,
                "fallback_type": "decompose",
                "fallback_glosses": decomp,
            })
            applied.append(f"{word}(신규)")

    with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(gloss_list, f, ensure_ascii=False, indent=2)

    print(f"decompose 적용: {len(applied)}개")
    for w in applied:
        decomp = DECOMPOSE_MAP.get(w.replace("(신규)", ""), [])
        print(f"  {w} → {decomp}")
    if missing_targets:
        print(f"\n미등록 대상 글로스: {list(set(missing_targets))}")


if __name__ == "__main__":
    main()
