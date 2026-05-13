"""
quiz_service — 사전 합성된 동의어와 사용자 답안의 exact-match (정규화 후).

정규화 규칙:
  1. trim + 모든 공백 제거
  2. 문장부호 제거 (.,!?~)
  3. 마지막 위치의 한국어 조사/종결어 1개 제거
  4. 소문자화

조사 목록은 보수적 — 너무 많이 자르면 다른 단어와 충돌 가능. 단답형 정답은
대개 명사(주인공명/감정어/짧은 행동어)라 이 정도면 충분.
"""
from __future__ import annotations

import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.quiz import Quiz

# 길이 내림차순 — endswith 매칭 시 긴 조사·종결어 먼저 잡기
# 주의: 1자 '이'는 명사 어미('호랑이','토끼이름' 같은 어휘)와 충돌하므로 제외한다.
_PARTICLES = sorted(
    [
        # 조사
        "을", "를", "가", "은", "는", "에", "에서", "께", "께서",
        "와", "과", "도", "만", "의", "로", "으로", "랑", "이랑",
        # 종결어 (4~7세 흔한 형태 — LLM 동의어 풀이 다양한 어말 변형을 커버한다고 가정)
        "이었어요", "였어요", "이에요", "예요",
        "어요", "아요", "요",
        "이야", "야",
    ],
    key=len, reverse=True,
)

_PUNCT_RE = re.compile(r"[!?.,~·…\"'`()\[\]{}<>]")
_WS_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = s.strip()
    s = _PUNCT_RE.sub("", s)
    s = _WS_RE.sub("", s)
    for p in _PARTICLES:
        if len(s) > len(p) and s.endswith(p):
            s = s[: -len(p)]
            break
    return s.lower()


def is_correct(user_answer: str, expected_answer: str, synonyms: list[str]) -> bool:
    target = normalize(user_answer)
    if not target:
        return False
    candidates = {normalize(expected_answer), *(normalize(s) for s in synonyms)}
    candidates.discard("")
    return target in candidates


async def get_quiz_by_book_section(
    db: AsyncSession, book_id: uuid.UUID, section_order: int
) -> Quiz | None:
    result = await db.execute(
        select(Quiz).where(
            Quiz.book_id == book_id,
            Quiz.section_order == section_order,
        )
    )
    return result.scalar_one_or_none()


async def get_quizzes_by_book(db: AsyncSession, book_id: uuid.UUID) -> list[Quiz]:
    result = await db.execute(
        select(Quiz).where(Quiz.book_id == book_id).order_by(Quiz.section_order)
    )
    return list(result.scalars().all())


async def get_quiz_by_id(db: AsyncSession, quiz_id: uuid.UUID) -> Quiz | None:
    result = await db.execute(select(Quiz).where(Quiz.id == quiz_id))
    return result.scalar_one_or_none()
