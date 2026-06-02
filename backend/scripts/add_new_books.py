"""
신규 7권 동화를 DB에 추가합니다 (기존 데이터 유지).
실행: docker compose exec backend python scripts/add_new_books.py
"""
import asyncio
import sys
import os
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.database import AsyncSessionLocal
from models.book import Book, BookSection
from sqlalchemy import text

NEW_BOOKS = [
    {
        "title": "개미와 베짱이",
        "description": "여름 내내 놀기만 한 베짱이와 열심히 일한 개미의 이야기입니다.",
        "cover_image_url": "/static/covers/ant_grasshopper.svg",
        "target_age_min": 5,
        "target_age_max": 8,
        "categories": ["우화"],
        "author": "이솝",
        "sections": [
            {"order": 1, "title": "여름날의 베짱이", "text": "여름 내내 개미는 열심히 먹이를 모았습니다. 베짱이는 노래하고 춤추며 놀기만 했습니다. 베짱이는 개미를 보며 왜 그렇게 힘들게 일하냐고 비웃었습니다.", "image_url": "/images/sections/개미와_베짱이_1.png"},
            {"order": 2, "title": "추운 겨울", "text": "겨울이 되자 눈이 내리고 먹을 것이 사라졌습니다. 베짱이는 배가 고파 개미의 집을 찾아갔습니다. 베짱이는 여름에 놀기만 했던 것을 후회했습니다.", "image_url": "/images/sections/개미와_베짱이_2.png"},
            {"order": 3, "title": "교훈", "text": "개미는 베짱이에게 여름에 놀기만 했으니 스스로 책임져야 한다고 했습니다. 베짱이는 게으름을 반성하며 다음에는 열심히 일하겠다고 다짐했습니다. 부지런함이 얼마나 중요한지 알게 되었습니다.", "image_url": "/images/sections/개미와_베짱이_3.png"},
        ],
    },
    {
        "title": "빨간 모자",
        "description": "빨간 모자 소녀가 할머니 댁에 가다가 나쁜 늑대를 만나는 이야기입니다.",
        "cover_image_url": "/static/covers/red_hood.svg",
        "target_age_min": 5,
        "target_age_max": 8,
        "categories": ["세계명작"],
        "author": "그림형제",
        "sections": [
            {"order": 1, "title": "숲 속의 만남", "text": "빨간 모자를 쓴 소녀가 할머니께 음식을 가져다드리러 숲 속으로 갔습니다. 엄마는 절대 길에서 멈추지 말라고 했습니다. 그러나 숲 속에서 나쁜 늑대를 만났습니다.", "image_url": "/images/sections/빨간_모자_1.png"},
            {"order": 2, "title": "늑대의 속임수", "text": "늑대는 먼저 할머니 집에 도착해 할머니를 잡아먹었습니다. 늑대는 할머니인 척하며 침대에 누워 빨간 모자를 기다렸습니다. 빨간 모자가 도착해서 할머니의 이상한 모습을 발견했습니다.", "image_url": "/images/sections/빨간_모자_2.png"},
            {"order": 3, "title": "구조", "text": "빨간 모자는 늑대라는 것을 눈치채고 소리를 질렀습니다. 사냥꾼이 달려와 늑대를 물리쳤습니다. 할머니와 빨간 모자는 모두 무사히 구조되었습니다.", "image_url": "/images/sections/빨간_모자_3.png"},
        ],
    },
    {
        "title": "신데렐라",
        "description": "못된 새어머니 밑에서 고생하던 신데렐라가 왕자를 만나 행복해지는 이야기입니다.",
        "cover_image_url": "/static/covers/cinderella.svg",
        "target_age_min": 5,
        "target_age_max": 9,
        "categories": ["세계명작"],
        "author": "샤를 페로",
        "sections": [
            {"order": 1, "title": "신데렐라의 슬픔", "text": "신데렐라는 못된 새어머니와 언니들에게 구박을 받으며 살았습니다. 왕자의 무도회 소식이 들렸지만 신데렐라는 갈 수 없었습니다. 신데렐라는 혼자 남아 슬피 울었습니다.", "image_url": "/images/sections/신데렐라_1.png"},
            {"order": 2, "title": "요정의 선물", "text": "요정 할머니가 나타나 신데렐라를 아름다운 모습으로 변신시켜 주었습니다. 신데렐라는 무도회에서 왕자와 춤을 추었습니다. 자정이 되자 황급히 도망치다가 유리구두 한 짝을 잃었습니다.", "image_url": "/images/sections/신데렐라_2.png"},
            {"order": 3, "title": "행복한 결말", "text": "왕자는 유리구두를 들고 온 나라를 돌아다니며 신데렐라를 찾았습니다. 신데렐라에게만 딱 맞는 유리구두를 신겨보았습니다. 왕자와 신데렐라는 행복하게 결혼했습니다.", "image_url": "/images/sections/신데렐라_3.png"},
        ],
    },
    {
        "title": "아기 돼지 삼형제",
        "description": "세 마리 아기 돼지가 늑대로부터 집을 지키는 이야기입니다.",
        "cover_image_url": "/static/covers/three_pigs.svg",
        "target_age_min": 5,
        "target_age_max": 8,
        "categories": ["세계명작"],
        "author": "전래동화",
        "sections": [
            {"order": 1, "title": "집 짓기", "text": "아기 돼지 삼형제가 각자 집을 짓기로 했습니다. 첫째는 짚으로, 둘째는 나무로, 셋째는 튼튼한 벽돌로 집을 지었습니다. 셋째는 오래 걸렸지만 가장 튼튼하게 지었습니다.", "image_url": "/images/sections/아기_돼지_삼형제_1.png"},
            {"order": 2, "title": "늑대의 습격", "text": "나쁜 늑대가 나타나 짚 집과 나무 집을 훅 불어 무너뜨렸습니다. 첫째와 둘째는 셋째의 벽돌 집으로 도망쳤습니다. 늑대는 벽돌 집도 무너뜨리려 했습니다.", "image_url": "/images/sections/아기_돼지_삼형제_2.png"},
            {"order": 3, "title": "승리", "text": "늑대는 아무리 불어도 벽돌 집을 무너뜨릴 수 없었습니다. 굴뚝으로 들어오려다 냄비에 빠져 도망갔습니다. 삼형제는 안전하게 살았고 튼튼하게 만드는 것이 중요하다는 것을 배웠습니다.", "image_url": "/images/sections/아기_돼지_삼형제_3.png"},
        ],
    },
    {
        "title": "인어공주",
        "description": "바다 속 인어공주가 왕자를 사랑하게 되는 아름답고 슬픈 이야기입니다.",
        "cover_image_url": "/static/covers/little_mermaid.svg",
        "target_age_min": 6,
        "target_age_max": 10,
        "categories": ["세계명작"],
        "author": "안데르센",
        "sections": [
            {"order": 1, "title": "바다 속 공주", "text": "바다 속에 아름다운 인어공주가 살았습니다. 인어공주는 바다 위 세상과 왕자에게 관심이 많았습니다. 어느 날 폭풍우에 빠진 왕자를 구해 해변에 데려다 주었습니다.", "image_url": "/images/sections/인어공주_1.png"},
            {"order": 2, "title": "마녀와의 거래", "text": "인어공주는 마녀에게 아름다운 목소리를 주고 사람의 다리를 얻었습니다. 왕자를 만날 수 있었지만 말을 할 수 없었습니다. 왕자는 인어공주가 자신을 구해준 것을 알지 못했습니다.", "image_url": "/images/sections/인어공주_2.png"},
            {"order": 3, "title": "영원한 사랑", "text": "왕자는 다른 공주와 결혼했습니다. 인어공주는 바다 거품이 되었습니다. 그러나 진실한 사랑과 희생의 마음은 영원히 기억됩니다.", "image_url": "/images/sections/인어공주_3.png"},
        ],
    },
    {
        "title": "벌거벗은 임금님",
        "description": "새 옷을 좋아하는 임금님이 사기꾼들에게 속아 벌거벗고 행진하는 이야기입니다.",
        "cover_image_url": "/static/covers/emperors_new_clothes.svg",
        "target_age_min": 6,
        "target_age_max": 10,
        "categories": ["세계명작"],
        "author": "안데르센",
        "sections": [
            {"order": 1, "title": "사기꾼의 등장", "text": "옷을 무척 좋아하는 임금님이 있었습니다. 두 사기꾼이 나타나 멍청한 사람에게는 보이지 않는 신기한 옷을 만들어 주겠다고 했습니다. 임금님은 그 옷을 갖고 싶어했습니다.", "image_url": "/images/sections/벌거벗은_임금님_1.png"},
            {"order": 2, "title": "아무것도 없는 옷", "text": "신하들은 아무것도 없는 것을 보면서도 멋지다고 칭찬했습니다. 임금님도 아무것도 보이지 않았지만 멍청한 티를 내기 싫어 칭찬했습니다. 임금님은 아무것도 입지 않은 채 행진을 했습니다.", "image_url": "/images/sections/벌거벗은_임금님_2.png"},
            {"order": 3, "title": "진실을 말한 아이", "text": "한 어린 아이가 임금님이 아무것도 안 입었다고 외쳤습니다. 사람들은 그제야 진실을 깨달으며 웃음을 터뜨렸습니다. 진실을 말하는 용기가 얼마나 중요한지 알게 되었습니다.", "image_url": "/images/sections/벌거벗은_임금님_3.png"},
        ],
    },
    {
        "title": "사계절 친구들",
        "description": "계절마다 함께하는 동물 친구들의 우정 이야기입니다.",
        "cover_image_url": "/static/covers/four_seasons.svg",
        "target_age_min": 5,
        "target_age_max": 8,
        "categories": ["창작동화"],
        "author": "김은하",
        "sections": [
            {"order": 1, "title": "봄과 여름", "text": "봄이 되자 토끼와 사슴이 꽃밭에서 뛰어놀았습니다. 여름에는 개구리와 물고기가 시냇가에서 신나게 물놀이를 했습니다. 친구들은 매일 함께 즐거운 시간을 보냈습니다.", "image_url": "/images/sections/사계절_친구들_1.png"},
            {"order": 2, "title": "가을의 준비", "text": "가을이 되자 다람쥐는 도토리를 모으고 새들은 남쪽으로 떠났습니다. 친구들은 서로 따뜻하게 작별 인사를 나누었습니다. 각자 겨울을 준비하며 다시 만날 날을 기약했습니다.", "image_url": "/images/sections/사계절_친구들_2.png"},
            {"order": 3, "title": "겨울과 재회", "text": "겨울에는 눈이 내려 온 세상이 하얗게 변했습니다. 친구들은 따뜻한 곳에서 봄을 기다렸습니다. 봄이 다시 오자 친구들은 모두 만나 기쁘게 인사하며 새로운 계절을 맞이했습니다.", "image_url": "/images/sections/사계절_친구들_3.png"},
        ],
    },
]


async def add_books():
    async with AsyncSessionLocal() as db:
        now = datetime.now(timezone.utc)
        added = 0

        for book_data in NEW_BOOKS:
            # 이미 존재하면 건너뜀
            row = await db.execute(
                text("SELECT id FROM books WHERE title = :title"),
                {"title": book_data["title"]},
            )
            if row.fetchone():
                print(f"  ⏭ 이미 존재: {book_data['title']}")
                continue

            sections_data = book_data.pop("sections")
            book = Book(
                id=uuid.uuid4(),
                created_at=now,
                **book_data,
            )
            db.add(book)
            await db.flush()

            for sec in sections_data:
                db.add(BookSection(
                    id=uuid.uuid4(),
                    book_id=book.id,
                    **sec,
                ))

            print(f"  ✅ 추가: {book.title}")
            added += 1

        await db.commit()
        print(f"\n총 {added}권 추가 완료")


if __name__ == "__main__":
    asyncio.run(add_books())
