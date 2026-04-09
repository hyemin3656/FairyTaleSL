"""
샘플 동화 3권 및 기본 데이터를 DB에 삽입합니다.
실행: docker compose exec backend python scripts/seed.py
"""
import asyncio
import uuid
from datetime import datetime, timezone

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.database import AsyncSessionLocal
from models.book import Book, BookSection
from models.motion import GlossMotion


BOOKS = [
    {
        "id": uuid.uuid4(),
        "title": "토끼와 거북이",
        "description": "빠른 토끼와 느리지만 꾸준한 거북이의 경주 이야기입니다.",
        "cover_image_url": "/static/covers/rabbit_turtle.svg",
        "target_age_min": 5,
        "target_age_max": 8,
        "categories": ["동물", "우정"],
        "author": "이솝",
        "sections": [
            {
                "order": 1,
                "title": "출발",
                "text": "어느 날 토끼와 거북이가 경주를 하기로 했습니다. 토끼는 자신만만하게 출발했습니다.",
            },
            {
                "order": 2,
                "title": "낮잠",
                "text": "토끼는 너무 빠르게 달린 나머지 중간에 낮잠을 자기로 했습니다. 거북이는 쉬지 않고 천천히 걸었습니다.",
            },
            {
                "order": 3,
                "title": "결승",
                "text": "거북이가 결승선에 먼저 도착했습니다. 잠에서 깬 토끼는 깜짝 놀랐습니다. 꾸준함이 중요하다는 것을 배웠습니다.",
            },
        ],
    },
    {
        "id": uuid.uuid4(),
        "title": "혹부리 영감",
        "description": "혹이 달린 할아버지가 도깨비들과 만나는 전래 민담입니다.",
        "cover_image_url": "/static/covers/hump_grandpa.svg",
        "target_age_min": 6,
        "target_age_max": 10,
        "categories": ["전래민담", "판타지"],
        "author": "전래동화",
        "sections": [
            {
                "order": 1,
                "title": "할아버지의 혹",
                "text": "얼굴에 큰 혹이 달린 할아버지가 있었습니다. 어느 날 숲 속에서 길을 잃었습니다.",
            },
            {
                "order": 2,
                "title": "도깨비 잔치",
                "text": "할아버지는 도깨비들의 잔치를 발견했습니다. 할아버지가 노래를 부르자 도깨비들이 좋아했습니다.",
            },
            {
                "order": 3,
                "title": "혹을 떼다",
                "text": "도깨비들은 노래의 비밀이 혹 속에 있다고 생각해 혹을 떼어 주었습니다. 할아버지는 기뻐하며 집으로 돌아갔습니다.",
            },
        ],
    },
    {
        "id": uuid.uuid4(),
        "title": "별이를 찾아서",
        "description": "우주를 여행하며 잃어버린 별을 찾는 소녀 이야기입니다.",
        "cover_image_url": "/static/covers/find_star.svg",
        "target_age_min": 7,
        "target_age_max": 10,
        "categories": ["판타지", "과학"],
        "author": "김은하",
        "sections": [
            {
                "order": 1,
                "title": "별똥별",
                "text": "소녀 하나는 밤하늘에서 반짝이는 별 하나가 사라진 것을 발견했습니다. 별을 찾아 우주 여행을 시작했습니다.",
            },
            {
                "order": 2,
                "title": "달나라",
                "text": "달에 도착한 하나는 달토끼를 만났습니다. 달토끼는 별이 블랙홀 근처에 있다고 알려 주었습니다.",
            },
            {
                "order": 3,
                "title": "별의 귀환",
                "text": "하나는 용기를 내어 블랙홀 근처로 날아갔습니다. 별을 찾아 하늘에 돌려보내자 온 우주가 환하게 빛났습니다.",
            },
        ],
    },
]

GLOSS_MOTIONS = [
    # ── 토끼와 거북이 ──────────────────────────────────────────
    {"gloss": "토끼",   "gltf_clip_url": "/static/motions/rabbit.glb",   "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.3}, "duration_sec": 1.2},
    {"gloss": "거북이", "gltf_clip_url": "/static/motions/turtle.glb",   "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.5},
    {"gloss": "경주",   "gltf_clip_url": "/static/motions/race.glb",     "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.5, "eyeWide": 0.2}, "duration_sec": 1.8},
    {"gloss": "출발",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.4}, "duration_sec": 1.0},
    {"gloss": "낮잠",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.3},
    {"gloss": "결승",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.6}, "duration_sec": 1.2},
    {"gloss": "달리",   "gltf_clip_url": "/static/motions/race.glb",     "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.3}, "duration_sec": 1.0},
    {"gloss": "빠르",   "gltf_clip_url": "/static/motions/race.glb",     "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.2}, "duration_sec": 0.9},
    {"gloss": "느리",   "gltf_clip_url": "/static/motions/turtle.glb",   "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.4},
    {"gloss": "자신",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.3}, "duration_sec": 1.0},
    {"gloss": "쉬",     "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.0},
    {"gloss": "먼저",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.4}, "duration_sec": 1.0},
    {"gloss": "꾸준",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.5}, "duration_sec": 1.2},
    {"gloss": "중요",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.1},
    {"gloss": "잠",     "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.1},
    {"gloss": "도착",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.6}, "duration_sec": 1.2},
    {"gloss": "놀라",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "surprised","blendshape_params": {"eyeWide": 0.7}, "duration_sec": 1.0},
    {"gloss": "배우",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.1},

    # ── 혹부리 영감 ────────────────────────────────────────────
    {"gloss": "할아버지","gltf_clip_url": "/static/motions/fallback.glb","emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.3},
    {"gloss": "혹",     "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.0},
    {"gloss": "숲",     "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.0},
    {"gloss": "도깨비", "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "surprised","blendshape_params": {"eyeWide": 0.5}, "duration_sec": 1.2},
    {"gloss": "노래",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.6}, "duration_sec": 1.3},
    {"gloss": "잔치",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.5}, "duration_sec": 1.2},
    {"gloss": "길",     "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 0.9},
    {"gloss": "기뻐",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.7}, "duration_sec": 1.1},
    {"gloss": "집",     "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.0},
    {"gloss": "비밀",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.1},

    # ── 별이를 찾아서 ──────────────────────────────────────────
    {"gloss": "소녀",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.3}, "duration_sec": 1.1},
    {"gloss": "별",     "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.5, "eyeWide": 0.3}, "duration_sec": 1.2},
    {"gloss": "우주",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "surprised","blendshape_params": {"eyeWide": 0.4}, "duration_sec": 1.3},
    {"gloss": "달",     "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.4}, "duration_sec": 1.1},
    {"gloss": "달나라", "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.4, "eyeWide": 0.2}, "duration_sec": 1.3},
    {"gloss": "블랙홀", "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "surprised","blendshape_params": {"eyeWide": 0.7}, "duration_sec": 1.4},
    {"gloss": "용기",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.6}, "duration_sec": 1.2},
    {"gloss": "여행",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.4}, "duration_sec": 1.2},
    {"gloss": "찾",     "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.0},
    {"gloss": "빛",     "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.5, "eyeWide": 0.3}, "duration_sec": 1.1},
    {"gloss": "밤하늘", "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "neutral",  "blendshape_params": {}, "duration_sec": 1.2},
    {"gloss": "사라",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "sad",      "blendshape_params": {}, "duration_sec": 1.1},
    {"gloss": "반짝",   "gltf_clip_url": "/static/motions/fallback.glb", "emotion_label": "happy",    "blendshape_params": {"mouthSmile": 0.5, "eyeWide": 0.4}, "duration_sec": 1.1},

    # ── FALLBACK ───────────────────────────────────────────────
    {
        "gloss": "FALLBACK",
        "gltf_clip_url": "/static/motions/fallback.glb",
        "emotion_label": "neutral",
        "blendshape_params": {},
        "duration_sec": 1.0,
        "description": "미등록 글로스 fallback 모션",
    },
]


async def seed():
    # SQLite 로컬 테스트 시 테이블 자동 생성 (PostgreSQL은 Alembic 사용)
    from core.database import engine, Base
    import models as _  # noqa: F401 — 모든 모델 메타데이터 등록
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        # 중복 실행 방지
        from sqlalchemy import select
        existing = await db.execute(select(Book).limit(1))
        if existing.scalar_one_or_none():
            print("이미 시드 데이터가 존재합니다. 건너뜁니다.")
            return

        now = datetime.now(timezone.utc)

        for book_data in BOOKS:
            sections_data = book_data.pop("sections")
            book = Book(
                **book_data,
                created_at=now,
            )
            db.add(book)
            await db.flush()  # book.id 확보

            for sec in sections_data:
                db.add(BookSection(id=uuid.uuid4(), book_id=book.id, **sec))

        for motion_data in GLOSS_MOTIONS:
            db.add(GlossMotion(id=uuid.uuid4(), created_at=now, **motion_data))

        await db.commit()
        print(f"시드 완료: 동화 {len(BOOKS)}권, 모션 {len(GLOSS_MOTIONS)}개 삽입")


if __name__ == "__main__":
    asyncio.run(seed())
