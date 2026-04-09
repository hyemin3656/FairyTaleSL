from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from schemas.motion import GlossConvertRequest, GlossConvertResponse
from services.gloss_service import tokenize_text, resolve_motions

router = APIRouter(prefix="/gloss", tags=["gloss"])


@router.post("/convert", response_model=GlossConvertResponse)
async def convert_text_to_gloss(
    body: GlossConvertRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    텍스트 → 글로스 변환 파이프라인.
    1. 텍스트를 토큰(글로스 후보)으로 분리
    2. Motion DB에서 각 글로스 조회 (없으면 FALLBACK)
    3. 순서대로 MotionClip 리스트 반환
    """
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text가 비어 있습니다.")

    tokens = tokenize_text(text)
    if not tokens:
        raise HTTPException(status_code=422, detail="토크나이징 결과가 없습니다.")

    clips = await resolve_motions(db, tokens)
    total_duration = sum(c.duration_sec for c in clips)

    return GlossConvertResponse(
        original_text=text,
        tokens=tokens,
        clips=clips,
        total_duration_sec=round(total_duration, 2),
    )
