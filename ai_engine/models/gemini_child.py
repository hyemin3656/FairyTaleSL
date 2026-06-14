"""
gemini_child — 아이가 동화에 대해 한 질문에 LLM으로 답변.
RunYourAI API (OpenAI 호환) 사용 — LLM_API_KEY 환경변수 필요.
"""
from __future__ import annotations

import json
import os
import sqlite3
import urllib.request
from functools import lru_cache
from pathlib import Path

RUNYOURAI_BASE_URL = "https://api.runyour.ai/v1"
_MOTION_DB_PATH = Path("/data_pipeline/sign_generation/data/motion_db.sqlite")


def _llm_call(messages: list[dict], temperature: float = 0.7, max_tokens: int = 256) -> str:
    # 기존 GEMINI_API_KEY 호환 — LLM_API_KEY 없으면 GEMINI_API_KEY로 폴백
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    model   = os.environ.get("LLM_MODEL", "anthropic/claude-haiku-4-5")
    if not api_key:
        raise RuntimeError("LLM_API_KEY (or GEMINI_API_KEY) not set in environment")

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{RUNYOURAI_BASE_URL}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return (content or "").strip()


CHILD_QA_SYSTEM = """당신은 4~7세 아동에게 동화를 설명해주는 친절한 도우미입니다.
아이의 질문에 다음 안내를 참고해 너그럽고 유연하게 답해주세요.

[입력 처리]
아이의 질문은 두 가지 형태로 들어올 수 있어요.
  (A) 키보드로 친 자연스러운 한국어 문장 — 예: "직녀가 누구야?"
  (B) 수어로 표현한 **글로스 시퀀스** — 단어들의 기본형이 공백으로 이어진 형태.
      조사·어미가 생략돼 있고, 1~2개 단어만 올 수도 있습니다.

글로스 시퀀스가 들어오면 머릿속으로 자연스러운 한국어 질문으로 재구성한 뒤
그 의미에 따라 답하세요. 재구성한 문장 자체를 답변에 적지는 마세요.

[유연한 의미 추론 — 매우 중요]
짧은 단어 1~2개만 와도 동화 본문과 연결 가능한 모든 해석을 시도하세요.
한 단어가 본문에 직접 등장하지 않아도, 비유나 함의로 연결되면 그 의미로 답하세요.

예시 — 같은 글로스도 본문에 따라 여러 의미로 해석:
  "누구 땀"  → 본문에 누가 열심히 일했는지 묻는 것으로 해석 가능
               (땀 = 노력, 부지런함의 함의)
               → "개미가 부지런히 일하면서 땀을 흘렸어요."
  "누구 먹다" → "누가 먹었어?" / "누가 먹이를 모았어?"
  "노력 좋다" → "노력하는 것이 좋아?" 같은 가치 판단으로 해석
  "직녀 누구" → "직녀가 누구야?"

원칙: **redirect는 최후의 수단**. 단어 하나하나의 의미와 본문 사이에
조금이라도 연결 고리가 있으면 답을 시도하세요.

[답변 규칙]
1. 4~7세 아동이 이해할 수 있는 쉬운 단어만 사용
2. 한 문장 이상, 두 문장 이하로 답하기 (단어 하나만 던지지 말 것)
3. 가능하면 동화 본문에 등장한 내용을 근거로 답하되, 본문의 함의·교훈을
   살리는 자연스러운 추론은 허용 (예: 부지런히 일했다 → 땀, 노력)
4. 인물·사물 소개 질문에는 역할·관계·행동 중 하나 이상을 포함해 설명
5. 부드럽고 따뜻한 말투
6. **정말로 어떤 해석도 불가능하거나, 본문과 전혀 관련 없는 외부 지식을
   요구하는 질문일 때만** 다음처럼 부드럽게 다시 물어봐 주세요:
   "지금 읽고 있는 동화와 관련된 질문을 해줄래? 예를 들면 등장인물이나
    이야기 속에서 일어난 일에 대해 물어봐도 좋아."

   단, 직접적 본문 등장이 없어도 본문 내용에서 추론할 수 있는 질문이면
   redirect하지 말고 답을 시도하세요."""


@lru_cache(maxsize=1)
def _load_ksl_glosses() -> str:
    try:
        conn = sqlite3.connect(str(_MOTION_DB_PATH))
        rows = conn.execute("SELECT gloss FROM motion_db ORDER BY gloss").fetchall()
        conn.close()
        glosses = [r[0] for r in rows if r[0] and not r[0].startswith("-")]
        return ", ".join(glosses)
    except Exception:
        return ""


KSL_REWRITE_SYSTEM = """당신은 한국수어(KSL) 전문가입니다.
아래 [원문]의 내용어(명사·동사·형용사·부사)를 [사용 가능한 수어 단어] 목록에 있는 단어로만 공백 구분 나열해주세요.

규칙:
1. [사용 가능한 수어 단어] 목록에 없는 단어는 목록 안의 의미상 가장 가까운 단어로 대체
2. 부정 표현 풀기: "행복하지 않다" → "슬프다", "못하다" → "어렵다"
3. 복합 동사·명사 분리: "잡아먹다" → "잡다 먹다", "날아가다" → "날다", "날개옷" → "날개 옷"
4. 동사·형용사는 기본형(~다)으로 출력
5. 조사·어미·기능어는 출력하지 않음
6. 원문의 모든 의미 단위를 빠짐없이 포함 (단어 수 제한 없음)
7. 공백으로 구분된 단어 목록만 출력, 문장 부호·설명 없이"""


def rewrite_for_sign(answer: str) -> str:
    gloss_list = _load_ksl_glosses()
    if not gloss_list:
        return answer
    try:
        return _llm_call(
            messages=[
                {"role": "system", "content": KSL_REWRITE_SYSTEM},
                {"role": "user", "content": (
                    f"[사용 가능한 수어 단어]\n{gloss_list}\n\n"
                    f"[원문]\n{answer.strip()}\n\n[재구성]"
                )},
            ],
            temperature=0.2,
            max_tokens=512,
        )
    except Exception:
        return answer


def answer_child_question(question: str, story_context: str) -> str:
    return _llm_call(
        messages=[
            {"role": "system", "content": CHILD_QA_SYSTEM},
            {"role": "user", "content": (
                f"[동화 본문]\n{story_context.strip()}\n\n"
                f"[아이의 질문]\n{question.strip()}"
            )},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
