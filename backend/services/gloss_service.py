"""
글로스 변환 파이프라인
텍스트 → 토큰(글로스 후보) → Motion DB 조회 → MotionClip 리스트 반환
"""
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.motion import GlossMotion
from schemas.motion import MotionClip

FALLBACK_GLOSS = "FALLBACK"
FALLBACK_URL = "/static/motions/fallback.glb"
FALLBACK_DURATION = 1.0

# 제거할 문장부호
_PUNCT_RE = re.compile(r"[.,!?。、「」『』\(\)\[\]《》〈〉""''…·\-–—]+")

# 한국어 조사/어미 (긴 것부터 순서 중요)
_JOSA = [
    "에서는", "에서도", "에게서", "으로서", "로부터", "에서의", "에서가",
    "에서를", "에게는", "에게도", "이라고", "라고는", "라고도", "이라는",
    "라는것", "에서", "에게", "에는", "에도", "에를", "으로", "로는",
    "로도", "로를", "라는", "이라", "라고", "이고", "이며", "이나",
    "이랑", "이든", "이서", "이야", "이요", "이여", "이지", "이다",
    "은가", "는가", "을까", "를까", "하며", "하고", "하는", "하여",
    "했다", "했습", "한다", "합니", "하기", "하자", "해서", "하다",
    "에서", "부터", "까지", "만을", "만은", "만이",
    "를", "을", "은", "는", "이", "가", "의", "에", "와", "과",
    "도", "만", "로", "으", "야", "아", "여", "랑", "든", "서",
]

# 어미 패턴 (동사/형용사 어간 추출) — 긴 것부터 순서 중요
_EOMI = [
    # 6자+
    "하였습니다", "이었습니다", "였습니다",
    # 5자
    "었습니다", "았습니다", "겠습니다", "했습니다",
    "있습니다", "없습니다", "합니다", "됩니다",
    "하여서", "되어서",
    "하기로",
    # 4자
    "한다고", "하였다", "되었다",
    "하기", "하고", "하자", "하며", "하여",
    "하는", "되는", "있는", "없는",
    "해서",
    # 3자
    "습니다", "ㅂ니다",
    "었습", "았습", "겠습",
    "었다", "았다", "겠다",
    "한다", "했다",
    "으며", "이며",
    "에서", "에게", "으로", "로써",
]


def _strip_josa(word: str) -> str:
    """단어에서 조사/어미를 제거하여 어간 추출"""
    for eomi in _EOMI:
        if word.endswith(eomi) and len(word) > len(eomi) + 1:
            return word[: -len(eomi)]
    for josa in _JOSA:
        if word.endswith(josa) and len(word) > len(josa):
            return word[: -len(josa)]
    return word


def tokenize_text(text: str) -> list[str]:
    """
    한국어 텍스트를 글로스 토큰 리스트로 변환.
    문장부호 제거 + 공백 분리 + 조사/어미 제거.
    """
    cleaned = _PUNCT_RE.sub(" ", text)
    raw_tokens = [t.strip() for t in cleaned.split() if t.strip()]
    # 조사 제거
    tokens = [_strip_josa(t) for t in raw_tokens]
    # 빈 토큰 필터링
    tokens = [t for t in tokens if t]
    return tokens


async def _fetch_motion_map(
    db: AsyncSession, glosses: list[str]
) -> dict[str, GlossMotion]:
    """glosses + FALLBACK 을 한 번의 쿼리로 조회하여 {gloss: GlossMotion} 반환"""
    lookup = set(glosses) | {FALLBACK_GLOSS}
    stmt = select(GlossMotion).where(GlossMotion.gloss.in_(lookup))
    result = await db.execute(stmt)
    return {row.gloss: row for row in result.scalars().all()}


async def resolve_motions(
    db: AsyncSession, tokens: list[str]
) -> list[MotionClip]:
    """
    토큰 리스트를 받아 각 토큰에 대응하는 MotionClip 리스트를 반환.
    DB에 해당 글로스가 없으면 FALLBACK 클립 사용.
    """
    motion_map = await _fetch_motion_map(db, tokens)
    fallback = motion_map.get(FALLBACK_GLOSS)

    clips: list[MotionClip] = []
    for token in tokens:
        motion = motion_map.get(token)
        if motion:
            clips.append(
                MotionClip(
                    gloss=motion.gloss,
                    gltf_clip_url=motion.gltf_clip_url,
                    emotion_label=motion.emotion_label,
                    blendshape_params=motion.blendshape_params or {},
                    duration_sec=motion.duration_sec,
                    is_fallback=False,
                )
            )
        else:
            clips.append(
                MotionClip(
                    gloss=token,
                    gltf_clip_url=fallback.gltf_clip_url if fallback else FALLBACK_URL,
                    emotion_label=fallback.emotion_label if fallback else "neutral",
                    blendshape_params=fallback.blendshape_params or {} if fallback else {},
                    duration_sec=fallback.duration_sec if fallback else FALLBACK_DURATION,
                    is_fallback=True,
                )
            )

    return clips
