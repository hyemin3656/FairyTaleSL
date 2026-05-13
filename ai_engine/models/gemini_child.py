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

1. 4~7세 아동이 이해할 수 있는 쉬운 단어만 사용
2. 1~2 문장으로 짧게 답하기
3. 동화 본문에 나온 내용만 답하기 (없는 내용은 만들지 않기)
4. 부드럽고 따뜻한 말투
"""


@lru_cache(maxsize=1)
def _get_client():
    from google import genai  # lazy: 패키지 미설치 시 import 시점에 터지지 않게
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key)


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
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=120,
        ),
    )
    return (response.text or "").strip()
