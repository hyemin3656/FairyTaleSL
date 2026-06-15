import uuid

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.book import Book


async def search_books(
    db: AsyncSession,
    keyword: str | None = None,
    categories: list[str] | None = None,
) -> list[Book]:
    """
    키워드(제목·설명) DB 필터 + 카테고리 Python-side OR 필터.
    keyword 없으면 전체 반환, categories 없으면 카테고리 필터 미적용.
    """
    stmt = select(Book)

    if keyword:
        like = f"%{keyword}%"
        stmt = stmt.where(
            or_(Book.title.ilike(like), Book.description.ilike(like))
        )

    result = await db.execute(stmt)
    books = list(result.scalars().all())

    # JSON 컬럼은 DB-side overlap이 불가하므로 Python-side 필터 적용
    if categories:
        cat_set = set(categories)
        books = [b for b in books if cat_set.intersection(b.categories or [])]

    # PostgreSQL collation이 한국어 가나다순을 보장하지 않으므로 Python-side 정렬
    import locale
    try:
        locale.setlocale(locale.LC_COLLATE, "ko_KR.UTF-8")
        books.sort(key=lambda b: locale.strxfrm(b.title))
    except locale.Error:
        books.sort(key=lambda b: b.title)

    return books


async def get_book_detail(db: AsyncSession, book_id: str) -> Book | None:
    try:
        book_uuid = uuid.UUID(book_id)
    except ValueError:
        return None
    stmt = (
        select(Book)
        .where(Book.id == book_uuid)
        .options(selectinload(Book.sections))
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()
