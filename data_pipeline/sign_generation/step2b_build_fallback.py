"""
미등록 글로스에 fallback 규칙을 gloss_list.json에 추가.

fallback_type:
  "decompose" - 등록된 글로스 조합으로 대체
  "text"      - 텍스트 자막 표시
"""
import json
from pathlib import Path

GLOSS_LIST_PATH = Path(__file__).parent / "data/gloss_list.json"

# 등록 글로스로 대체 가능한 경우만 decompose, 나머지는 text
DECOMPOSE_MAP: dict[str, list[str]] = {
    "살아가":   ["살다"],
    "이기":     ["이기다"],
    "느리":     ["느리다"],
    "우리나라": ["나라"],
    "전국":     ["나라"],
    "연못가":   ["연못"],
    "금은보화": ["재산"],
    "임금":     ["왕비"],       # 의미 유사 대체
    "용왕":     ["왕비"],
    "자만":     ["욕심"],
    "교훈":     ["이야기"],
    "그때":     ["동안"],
    "지상":     ["세상"],
    "최초":     ["시작"],
    "태어나":   ["아이"],
    "놀라":     ["감동"],
    "효도":     ["사랑"],
    "효심":     ["사랑"],
    "슬프":     ["후회"],
    "슬퍼하":   ["후회"],
    "그리워하": ["사랑"],
    "착하":     ["정직"],
    "나쁘":     ["의심"],
    "거짓말":   ["의심"],
    "사냥":     ["사슴"],
    "수탉":     ["동물"],
    "날개":     ["제비"],
    "날개옷":   ["제비"],
}


def main():
    with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
        gloss_list = json.load(f)

    registered = {g["gloss"] for g in gloss_list if g.get("video_path") or g.get("keypoint_path")}

    decompose_cnt = 0
    text_cnt = 0

    for g in gloss_list:
        if g.get("video_path") or g.get("keypoint_path"):
            g["fallback_type"] = None
            g["fallback_glosses"] = []
            continue

        gloss = g["gloss"]
        if gloss in DECOMPOSE_MAP:
            targets = DECOMPOSE_MAP[gloss]
            # 대체 글로스가 실제로 등록되어 있는지 검증
            valid = [t for t in targets if t in registered]
            if valid:
                g["fallback_type"] = "decompose"
                g["fallback_glosses"] = valid
                decompose_cnt += 1
                continue

        g["fallback_type"] = "text"
        g["fallback_glosses"] = []
        text_cnt += 1

    with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(gloss_list, f, ensure_ascii=False, indent=2)

    print(f"decompose fallback: {decompose_cnt}개")
    print(f"text fallback:      {text_cnt}개")
    print(f"등록 완료:          {len(registered)}개")
    print(f"합계:               {decompose_cnt + text_cnt + len(registered)}/220개")


if __name__ == "__main__":
    main()
