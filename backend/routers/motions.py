from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models.motion import GlossMotion
from schemas.motion import GlossMotionOut

router = APIRouter(prefix="/motions", tags=["motions"])


@router.get("", response_model=list[GlossMotionOut])
async def list_motions(db: AsyncSession = Depends(get_db)):
    """Motion DB 전체 조회"""
    result = await db.execute(select(GlossMotion).order_by(GlossMotion.gloss))
    return result.scalars().all()


@router.get("/{gloss}", response_model=GlossMotionOut)
async def get_motion(gloss: str, db: AsyncSession = Depends(get_db)):
    """글로스 이름으로 단일 모션 조회"""
    result = await db.execute(
        select(GlossMotion).where(GlossMotion.gloss == gloss)
    )
    motion = result.scalar_one_or_none()
    if not motion:
        raise HTTPException(status_code=404, detail=f"글로스 '{gloss}'를 찾을 수 없습니다.")
    return motion
