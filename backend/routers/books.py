from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models.book import BookCategory
from schemas.book import BookDetail, BookListResponse, BookListItem
from services.book_service import get_book_detail, search_books

router = APIRouter(prefix="/books", tags=["books"])


@router.get("", response_model=BookListResponse)
async def list_books(
    keyword: str | None = Query(None, description="제목·내용 키워드 검색"),
    categories: list[str] | None = Query(None, description="카테고리 필터 (복수 OR)"),
    db: AsyncSession = Depends(get_db),
):
    """
    동화책 목록 조회.
    - keyword 없으면 전체 반환
    - categories 없으면 전체 카테고리 반환
    - 복수 카테고리 선택 시 OR 조건 적용
    """
    books = await search_books(db, keyword=keyword, categories=categories)
    return BookListResponse(
        total=len(books),
        items=[BookListItem.model_validate(b) for b in books],
    )


@router.get("/categories", response_model=list[str])
async def get_categories():
    """사용 가능한 카테고리 목록 반환"""
    return BookCategory.ALL


@router.get("/{book_id}", response_model=BookDetail)
async def get_book(book_id: str, db: AsyncSession = Depends(get_db)):
    """동화책 상세 조회 (구간 포함)"""
    book = await get_book_detail(db, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="존재하지 않는 동화책입니다.")
    return BookDetail.model_validate(book)
