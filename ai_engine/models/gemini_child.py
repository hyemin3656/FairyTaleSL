"""
gemini_child — 아이가 동화에 대해 한 질문에 Gemini 2.5 Flash로 답변.

원칙(시스템 프롬프트):
  - 4~7세 어휘
  - 1~2 문장
  - 동화 본문에 근거 (없는 내용 만들지 않기)
  - 따뜻한 톤
"""
from __future__ import annotations

import os
from functools import lru_cache

CHILD_QA_SYSTEM = """당신은 4~7세 아동에게 동화를 설명해주는 친절한 도우미입니다.
아이의 질문에 다음 규칙으로 답해주세요:

[입력 처리 — 매우 중요]
0. 아이의 질문은 두 가지 형태로 들어올 수 있습니다.
   (A) 키보드로 친 자연스러운 한국어 문장 — 예: "직녀가 누구야?"
   (B) 수어로 표현한 **글로스 시퀀스** — 단어들의 기본형이 공백으로 이어진 형태.
       조사(이/가/은/는/을/를/에/에서 등)와 어미가 생략돼 있습니다.
       예: "직녀 누구"           → "직녀가 누구야?"
            "은하수 어디"         → "은하수가 어디에 있어?"
            "베 짜다 직녀"        → "직녀가 베를 짰어?"
            "사슴 나무꾼 도와주다" → "사슴이 나무꾼을 도와줬어?"

   글로스 시퀀스 같으면 먼저 머릿속으로 **자연스러운 한국어 질문**으로 재구성한 뒤,
   그 의미에 따라 답하세요. 재구성 과정 자체는 답변에 적지 마세요.
   짧은 단어 1~2개만 와도 가능한 한 질문 의도를 추론하세요.

[답변 규칙]
1. 4~7세 아동이 이해할 수 있는 쉬운 단어만 사용
2. 한 문장 이상, 두 문장 이하로 답하기 (단어 하나만 던지지 말 것)
3. 동화 본문에 나온 내용만 답하기 (없는 내용은 만들지 않기)
4. "X가 누구야?" / "X가 뭐야?" 같은 인물·사물 소개 질문에는 반드시 본문에 등장한
   역할·관계·행동 중 하나 이상을 포함해 한 문장으로 설명 (예: "직녀는 하늘에서 베를
   짜는 선녀였어요.")
5. 부드럽고 따뜻한 말투
6. 글로스 시퀀스가 너무 단편적이라 질문 의도를 도저히 추론할 수 없거나, 재구성한
   질문이 [동화 본문]과 관련이 없거나, 동화 안에서 답을 찾을 수 없으면, 답을
   지어내지 말고 정확히 다음처럼 부드럽게 다시 물어봐 주세요:
   "지금 읽고 있는 동화와 관련된 질문을 해줄래? 예를 들면 등장인물이나
    이야기 속에서 일어난 일에 대해 물어봐도 좋아."
"""


@lru_cache(maxsize=1)
def _get_client():
    from google import genai  # lazy: 패키지 미설치 시 import 시점에 터지지 않게
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key)


KSL_REWRITE_PROMPT = """당신은 한국수어(KSL) 전문가입니다.
아래 한국어 문장을 수어로 표현하기 좋은 형태로 다시 써주세요.

규칙:
1. 초등학교 1학년 수준의 쉬운 단어만 사용 (국립국어원 수어사전 등재 단어 위주)
2. 부정 표현 풀기: "행복하지 않다" → "슬프다", "못하다" → "어렵다"
3. 복합 동사 분리: "잡아먹다" → "잡다 먹다", "날아가다" → "날다"
4. 추상 단어를 구체적으로: "기억" → "알다", "관계" → "친구"
5. 1~2 문장, 최대 15 단어 이내
6. 조사·어미는 유지하되 본문 의미를 최대한 살리기
7. 원문과 의미가 달라지면 안 됨

원문: {original}

수어 친화 버전 (한국어 문장으로, 설명 없이 바로):"""


def rewrite_for_sign(answer: str) -> str:
    """Gemini 답변을 KSL 어휘로 재작성."""
    from google.genai import types
    client = _get_client()
    model = "gemini-2.5-flash"
    prompt = KSL_REWRITE_PROMPT.format(original=answer.strip())
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=128,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or answer).strip()


def answer_child_question(question: str, story_context: str) -> str:
    from google.genai import types
    client = _get_client()
    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
    # TODO: 토큰 절약 위해 story_context를 섹션 단위로만 전달 — 향후 전체 책으로 확장 시 캐시 검토
    prompt = (
        f"{CHILD_QA_SYSTEM}\n\n"
        f"[동화 본문]\n{story_context.strip()}\n\n"
        f"[아이의 질문]\n{question.strip()}\n\n"
        f"[답변]"
    )
    # Gemini 2.5 Flash는 기본적으로 "thinking"이 켜져 있어 max_output_tokens 중
    # 상당 부분을 사고 토큰이 잠식한다. 동화 응답에는 추론이 거의 필요 없으므로
    # thinking_budget=0으로 끄고, max_output_tokens도 안전 여유(~256)로 확장한다.
    # 끄지 않으면 "음, 소는" 처럼 답변이 도중에 끊긴다.
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=256,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip()
