"""
글로스 변환 파이프라인
텍스트 → 토큰(글로스 후보) → Motion DB 조회 → MotionClip 리스트 반환

형태소 분석: kiwipiepy (Java 불필요, 순수 Python)
KSL 어순: 시간 > 장소 > 명사 > 부사 > 부정 > 서술어
"""
from kiwipiepy import Kiwi

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.motion import GlossMotion
from schemas.motion import MotionClip

FALLBACK_GLOSS = "FALLBACK"
FALLBACK_URL = "/static/motions/fallback.glb"
FALLBACK_DURATION = 1.0

_kiwi = Kiwi()

# kiwipiepy POS prefix → 유효 품사
_VALID_POS = {"NNG", "NNP", "VV", "VA", "MAG"}

_STOPWORDS = {
    "하다", "이다", "있다", "되다", "않다", "없다", "같다",
    "것", "수", "때", "중", "등", "위", "곳", "데",
    "나", "너", "우리", "그", "이", "저",
}

# KSL 어순 분류용 키워드
_TIME_KEYWORDS = {
    "옛날", "어느날", "봄", "여름", "가을", "겨울",
    "아침", "저녁", "밤", "낮", "그때", "마지막",
    "처음", "나중", "결국", "어느", "아주",
}
_PLACE_KEYWORDS = {
    "산", "연못", "하늘", "마을", "집", "굴",
    "세상", "결승선", "바다", "장", "수수밭",
}
_NEG_KEYWORDS = {"않다", "못하다", "아니다", "없다", "안"}


def tokenize_text(text: str) -> list[str]:
    """
    한국어 텍스트 → KSL 어순 글로스 토큰 리스트.

    kiwipiepy로 형태소 분석 후 KSL 어순으로 재배열:
    시간 > 장소 > 명사(주어/목적어) > 부사 > 부정 > 서술어(동사/형용사)

    기능어(조사, 어미 등)는 제거하고 내용어만 유지.
    """
    result = _kiwi.analyze(text)
    tokens = result[0][0]  # 최적 분석 결과

    time_w, place_w, nouns, verbs, adjs, advs, negs = [], [], [], [], [], [], []

    for token in tokens:
        word = token.form
        pos3 = token.tag[:3]  # e.g. 'NNG', 'VV-I' → 'VV-'[:3] = 'VV-' → use [:3]

        if pos3 not in _VALID_POS or len(word) <= 1 or word in _STOPWORDS:
            continue

        if pos3 in ("NNG", "NNP"):
            if word in _TIME_KEYWORDS:
                time_w.append(word)
            elif word in _PLACE_KEYWORDS:
                place_w.append(word)
            else:
                nouns.append(word)
        elif pos3 == "VV":
            (negs if word in _NEG_KEYWORDS else verbs).append(word)
        elif pos3 == "VA":
            adjs.append(word)
        elif pos3 == "MAG":
            (negs if word in _NEG_KEYWORDS else advs).append(word)

    # KSL 어순: 시간 > 장소 > 명사 > 부사 > 부정 > 동사 > 형용사
    return time_w + place_w + nouns + advs + negs + verbs + adjs


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
