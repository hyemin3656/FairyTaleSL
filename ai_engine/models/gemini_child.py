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
    api_key = os.environ.get("LLM_API_KEY")
    model   = os.environ.get("LLM_MODEL", "anthropic/claude-haiku-4-5")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set in environment")

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

   글로스 시퀀스 같으면 먼저 머릿속으로 자연스러운 한국어 질문으로 재구성한 뒤,
   그 의미에 따라 답하세요. 재구성 과정 자체는 답변에 적지 마세요.
   짧은 단어 1~2개만 와도 가능한 한 질문 의도를 추론하세요.

[답변 규칙]
1. 4~7세 아동이 이해할 수 있는 쉬운 단어만 사용
2. 한 문장 이상, 두 문장 이하로 답하기
3. 동화 본문에 나온 내용만 답하기 (없는 내용은 만들지 않기)
4. 인물·사물 소개 질문에는 역할·관계·행동을 포함해 설명
5. 부드럽고 따뜻한 말투
6. 질문 의도를 추론할 수 없거나 동화와 관련 없으면:
   "지금 읽고 있는 동화와 관련된 질문을 해줄래?"라고 답하기"""


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
