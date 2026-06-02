"""
book_sections 테이블의 image_url을 /images/sections/ 경로로 일괄 업데이트합니다.
실행: docker compose exec backend python scripts/update_image_urls.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.database import AsyncSessionLocal
from sqlalchemy import text

# tale_id → (title, section_count) 매핑
TALES = [
    ("토끼와_거북이",        2),
    ("흥부와_놀부",          3),
    ("콩쥐팥쥐",            3),
    ("금도끼_은도끼",        3),
    ("심청전",              3),
    ("단군신화",            3),
    ("선녀와_나무꾼",        3),
    ("해님달님",            4),
    ("혹부리_영감",          3),
    ("별을_찾아서",          3),
    ("팥죽할머니와_호랑이",   3),
    ("견우와_직녀",          3),
    ("장화홍련",            3),
    ("개미와_베짱이",        3),
    ("빨간_모자",           3),
    ("신데렐라",            3),
    ("아기_돼지_삼형제",     3),
    ("인어공주",            3),
    ("벌거벗은_임금님",      3),
    ("사계절_친구들",        3),
]

# tale_id → 동화 제목 매핑 (books.title과 일치해야 함)
TITLE_MAP = {
    "토끼와_거북이":        "토끼와 거북이",
    "흥부와_놀부":          "흥부와 놀부",
    "콩쥐팥쥐":            "콩쥐팥쥐",
    "금도끼_은도끼":        "금도끼 은도끼",
    "심청전":              "심청전",
    "단군신화":            "단군신화",
    "선녀와_나무꾼":        "선녀와 나무꾼",
    "해님달님":            "해님달님",
    "혹부리_영감":          "혹부리 영감",
    "별을_찾아서":          "별이를 찾아서",
    "팥죽할머니와_호랑이":   "팥죽할머니와 호랑이",
    "견우와_직녀":          "견우와 직녀",
    "장화홍련":            "장화홍련",
    "개미와_베짱이":        "개미와 베짱이",
    "빨간_모자":           "빨간 모자",
    "신데렐라":            "신데렐라",
    "아기_돼지_삼형제":     "아기 돼지 삼형제",
    "인어공주":            "인어공주",
    "벌거벗은_임금님":      "벌거벗은 임금님",
    "사계절_친구들":        "사계절 친구들",
}


async def update():
    async with AsyncSessionLocal() as db:
        updated_total = 0

        for tale_id, section_count in TALES:
            title = TITLE_MAP[tale_id]

            # book_id 조회
            row = await db.execute(
                text("SELECT id FROM books WHERE title = :title"),
                {"title": title},
            )
            book = row.fetchone()
            if not book:
                print(f"  ⚠ 동화 없음: {title}")
                continue

            book_id = book[0]

            for order in range(1, section_count + 1):
                image_url = f"/images/sections/{tale_id}_{order}.png"
                result = await db.execute(
                    text(
                        "UPDATE book_sections SET image_url = :url "
                        "WHERE book_id = :book_id AND \"order\" = :order"
                    ),
                    {"url": image_url, "book_id": book_id, "order": order},
                )
                updated_total += result.rowcount

            print(f"  ✅ {title}: {section_count}개 섹션 업데이트")

        await db.commit()
        print(f"\n총 {updated_total}개 섹션 image_url 업데이트 완료")


if __name__ == "__main__":
    asyncio.run(update())
