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
import sqlite3
from functools import lru_cache
from pathlib import Path

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

# motion_db 경로: ai_engine 컨테이너에서 /data_pipeline으로 마운트됨
_MOTION_DB_PATH = Path("/data_pipeline/sign_generation/data/motion_db.sqlite")


@lru_cache(maxsize=1)
def _load_ksl_glosses() -> str:
    """motion_db.sqlite에서 글로스 목록을 로드해 쉼표 구분 문자열로 반환."""
    try:
        conn = sqlite3.connect(str(_MOTION_DB_PATH))
        rows = conn.execute("SELECT gloss FROM motion_db ORDER BY gloss").fetchall()
        conn.close()
        glosses = [r[0] for r in rows if r[0] and not r[0].startswith("-")]
        return ", ".join(glosses)
    except Exception:
        return ""


KSL_REWRITE_SYSTEM = """당신은 한국수어(KSL) 전문가입니다.
아래 [원문]을 [사용 가능한 수어 단어] 목록에 있는 단어들만 사용해서 재구성해주세요.

규칙:
1. [사용 가능한 수어 단어] 목록에 없는 단어는 반드시 목록 안의 유사한 단어로 대체
2. 부정 표현 풀기: "행복하지 않다" → "슬프다", "못하다" → "어렵다"
3. 복합 동사 분리: "잡아먹다" → "잡다 먹다", "날아가다" → "날다"
4. 의미는 최대한 유지하되 자연스러운 한국어 문장으로 작성
5. 1~2 문장, 최대 15 단어 이내
6. 설명 없이 재구성된 문장만 출력"""


def rewrite_for_sign(answer: str) -> str:
    """Gemini 답변을 국립국어원 KSL 등재 어휘만 사용해 재작성."""
    from google.genai import types
    client = _get_client()

    gloss_list = _load_ksl_glosses()
    if not gloss_list:
        return answer

    prompt = (
        f"{KSL_REWRITE_SYSTEM}\n\n"
        f"[사용 가능한 수어 단어]\n{gloss_list}\n\n"
        f"[원문]\n{answer.strip()}\n\n"
        f"[재구성]"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=128,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or answer).strip()


@lru_cache(maxsize=1)
def _get_client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key)


def answer_child_question(question: str, story_context: str) -> str:
    from google.genai import types
    client = _get_client()
    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
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
            max_output_tokens=256,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip()
