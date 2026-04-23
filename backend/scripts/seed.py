"""
전래동화 8권 및 기본 모션 데이터를 DB에 삽입합니다.
실행: docker compose exec backend python scripts/seed.py
"""
import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.database import AsyncSessionLocal
from models.book import Book, BookSection
from models.motion import GlossMotion

sys.path.insert(0, os.path.dirname(__file__))
from fairy_tales import FAIRY_TALES_STRUCTURED

# 감정 레이블 한→영 매핑 (gloss_list.json → GlossMotion.emotion_label)
_EMOTION_MAP = {
    "기쁨": "happy",
    "슬픔": "sad",
    "분노": "angry",
    "놀람": "surprised",
    "중립": "neutral",
}

# emotion_label → blendshape 기본값
_BLENDSHAPE_MAP = {
    "happy":     {"mouthSmile": 0.6},
    "sad":       {"mouthFrown": 0.5},
    "angry":     {"browDown": 0.6},
    "surprised": {"eyeWide": 0.6},
    "neutral":   {},
}

# gloss_list.json 경로: 컨테이너(/data_pipeline) 또는 로컬(../data_pipeline) 자동 선택
_GLOSS_LIST_PATH = (
    Path("/data_pipeline/sign_generation/data/gloss_list.json")
    if Path("/data_pipeline").exists()
    else Path(__file__).parent.parent.parent.parent / "data_pipeline/sign_generation/data/gloss_list.json"
)


def _build_gloss_motions() -> list[dict]:
    """gloss_list.json → GlossMotion seed 리스트 변환."""
    if not _GLOSS_LIST_PATH.exists():
        print(f"⚠️  gloss_list.json 없음: {_GLOSS_LIST_PATH}")
        return []

    with open(_GLOSS_LIST_PATH, encoding="utf-8") as f:
        gloss_list = json.load(f)

    motions = []
    for g in gloss_list:
        emotion_kr   = g.get("emotion_label", "중립")
        emotion_en   = _EMOTION_MAP.get(emotion_kr, "neutral")
        video_path   = g.get("video_path")
        # gltf_clip_url: 영상 경로를 임시 사용 (GLTF 미생성 단계)
        gltf_url     = video_path or "/static/motions/fallback.glb"

        motions.append({
            "gloss":            g["gloss"],
            "gltf_clip_url":    gltf_url,
            "emotion_label":    emotion_en,
            "blendshape_params": _BLENDSHAPE_MAP[emotion_en],
            "duration_sec":     1.2,
            "keypoint_path":    g.get("keypoint_path"),
            "video_path":       video_path,
            "fallback_type":    g.get("fallback_type"),
            "fallback_glosses": g.get("fallback_glosses", []) or None,
        })

    # FALLBACK 항목 추가
    motions.append({
        "gloss":            "FALLBACK",
        "gltf_clip_url":    "/static/motions/fallback.glb",
        "emotion_label":    "neutral",
        "blendshape_params": {},
        "duration_sec":     1.0,
        "description":      "미등록 글로스 fallback 모션",
        "keypoint_path":    None,
        "video_path":       None,
        "fallback_type":    None,
        "fallback_glosses": None,
    })
    return motions


def _build_books() -> list[dict]:
    """FAIRY_TALES_STRUCTURED → seed용 BOOKS 리스트 변환."""
    category_meta = {
        "토끼와_거북이": {"cover": "/static/covers/rabbit_turtle.svg",  "author": "이솝",    "age": (5, 8)},
        "흥부와_놀부":   {"cover": "/static/covers/heungbu_nolbu.svg",  "author": "전래동화", "age": (6, 9)},
        "콩쥐팥쥐":     {"cover": "/static/covers/kongjwi_patjwi.svg", "author": "전래동화", "age": (6, 9)},
        "금도끼_은도끼": {"cover": "/static/covers/golden_axe.svg",     "author": "전래동화", "age": (5, 8)},
        "심청전":       {"cover": "/static/covers/simcheong.svg",      "author": "전래동화", "age": (7, 10)},
        "단군신화":     {"cover": "/static/covers/dangun.svg",         "author": "전래동화", "age": (8, 11)},
        "선녀와_나무꾼": {"cover": "/static/covers/sunnyeo.svg",        "author": "전래동화", "age": (6, 9)},
        "해님달님":     {"cover": "/static/covers/haenim_dalnim.svg",  "author": "전래동화", "age": (5, 8)},
    }

    books = []
    for tale in FAIRY_TALES_STRUCTURED:
        tid  = tale["tale_id"]
        meta = category_meta.get(tid, {"cover": "/static/covers/fallback.svg", "author": "전래동화", "age": (6, 9)})

        # seg["sentences"]는 문자열 리스트
        def _sent_text(s):
            return s["text"] if isinstance(s, dict) else s

        full_text = " ".join(
            _sent_text(s)
            for seg in tale["segments"]
            for s in seg["sentences"]
        )
        sections = [
            {
                "order": seg["segment_id"],
                "title": f"{tale['title']} {seg['segment_id']}부",
                "text":  " ".join(_sent_text(s) for s in seg["sentences"]),
            }
            for seg in tale["segments"]
        ]
        books.append({
            "id":              uuid.uuid4(),
            "title":           tale["title"],
            "description":     full_text[:80] + "…",
            "cover_image_url": meta["cover"],
            "target_age_min":  meta["age"][0],
            "target_age_max":  meta["age"][1],
            "categories":      [tale["category"]],
            "author":          meta["author"],
            "sections":        sections,
        })
    return books


BOOKS = _build_books() + [
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
    {
        "id": uuid.uuid4(),
        "title": "팥죽할머니와 호랑이",
        "description": "팥죽을 끓이던 할머니를 도와준 친구들이 호랑이를 물리치는 이야기입니다.",
        "cover_image_url": "/static/covers/patjuk_tiger.svg",
        "target_age_min": 5,
        "target_age_max": 8,
        "categories": ["전래동화"],
        "author": "전래동화",
        "sections": [
            {
                "order": 1,
                "title": "할머니의 팥죽",
                "text": "옛날에 혼자 사는 할머니가 있었습니다. 할머니는 팥죽을 끓이러 산에 팥을 따러 갔습니다. 그때 호랑이가 나타나 할머니를 잡아먹으려 했습니다.",
            },
            {
                "order": 2,
                "title": "친구들의 도움",
                "text": "할머니는 팥죽을 나눠 주겠다고 약속하며 위기를 넘겼습니다. 집에 돌아온 할머니는 팥죽을 끓였습니다. 알밤, 송곳, 맷돌, 지게, 소 등 친구들이 모여 호랑이를 기다렸습니다.",
            },
            {
                "order": 3,
                "title": "호랑이를 물리치다",
                "text": "밤에 호랑이가 나타났습니다. 알밤이 호랑이 눈을 때리고, 송곳이 발을 찌르고, 맷돌이 등을 눌렀습니다. 호랑이는 혼비백산 도망쳤습니다. 할머니는 친구들과 함께 팥죽을 맛있게 먹었습니다.",
            },
        ],
    },
    {
        "id": uuid.uuid4(),
        "title": "견우와 직녀",
        "description": "하늘나라에서 사랑에 빠진 견우와 직녀가 은하수를 사이에 두고 칠월 칠석에 만나는 이야기입니다.",
        "cover_image_url": "/static/covers/gyeonu_jiknyeo.svg",
        "target_age_min": 6,
        "target_age_max": 10,
        "categories": ["전래동화", "신화"],
        "author": "전래동화",
        "sections": [
            {
                "order": 1,
                "title": "하늘나라의 만남",
                "text": "하늘나라에 소를 모는 견우와 베를 짜는 직녀가 살았습니다. 둘은 서로 만나 사랑에 빠졌습니다. 그러나 사랑에 빠진 후 일을 게을리하여 하느님의 노여움을 샀습니다.",
            },
            {
                "order": 2,
                "title": "은하수의 이별",
                "text": "하느님은 두 사람을 은하수 양쪽으로 갈라놓았습니다. 견우와 직녀는 서로 바라보며 눈물을 흘렸습니다. 직녀의 눈물이 비가 되어 세상에 내렸습니다.",
            },
            {
                "order": 3,
                "title": "칠월 칠석의 만남",
                "text": "하느님은 일 년에 한 번, 칠월 칠석에만 만날 수 있도록 허락했습니다. 까치와 까마귀들이 날아와 다리를 놓아 주었습니다. 두 사람은 오작교 위에서 기쁘게 만났습니다.",
            },
        ],
    },
    {
        "id": uuid.uuid4(),
        "title": "장화홍련",
        "description": "계모의 구박을 받는 장화와 홍련 자매가 억울한 누명을 벗고 원한을 푸는 이야기입니다.",
        "cover_image_url": "/static/covers/janghwa_hongryeon.svg",
        "target_age_min": 8,
        "target_age_max": 12,
        "categories": ["전래동화"],
        "author": "전래동화",
        "sections": [
            {
                "order": 1,
                "title": "계모의 구박",
                "text": "옛날에 장화와 홍련이라는 두 자매가 살았습니다. 어머니가 돌아가신 뒤 계모가 들어왔습니다. 계모는 자매를 미워하며 심하게 구박했습니다.",
            },
            {
                "order": 2,
                "title": "억울한 죽음",
                "text": "계모는 장화에게 누명을 씌워 연못에 빠뜨렸습니다. 홍련도 언니를 따라 연못에 몸을 던졌습니다. 두 자매의 원혼은 매일 밤 부사 앞에 나타나 억울함을 호소했습니다.",
            },
            {
                "order": 3,
                "title": "원한을 풀다",
                "text": "용감한 부사가 진실을 밝혀 계모를 벌했습니다. 장화와 홍련의 원한이 풀렸습니다. 두 자매는 다시 태어나 행복하게 살았습니다.",
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
        from sqlalchemy import select, func

        now = datetime.now(timezone.utc)

        # Book 삽입 (없는 경우만)
        book_count = (await db.execute(select(func.count()).select_from(Book))).scalar()
        if book_count:
            print(f"동화 데이터 이미 존재 ({book_count}권). 건너뜁니다.")
        else:
            books_data = BOOKS
            for book_data in books_data:
                sections_data = book_data.pop("sections")
                book = Book(**book_data, created_at=now)
                db.add(book)
                await db.flush()
                for sec in sections_data:
                    db.add(BookSection(id=uuid.uuid4(), book_id=book.id, **sec))
            await db.commit()
            print(f"동화 {len(books_data)}권 삽입 완료")

        # GlossMotion 삽입 (없는 경우만)
        motion_count = (await db.execute(select(func.count()).select_from(GlossMotion))).scalar()
        if motion_count:
            print(f"모션 데이터 이미 존재 ({motion_count}개). 건너뜁니다.")
        else:
            motions_data = _build_gloss_motions()
            for motion_data in motions_data:
                db.add(GlossMotion(id=uuid.uuid4(), created_at=now, **motion_data))
            await db.commit()
            print(f"모션 {len(motions_data)}개 삽입 완료")


if __name__ == "__main__":
    asyncio.run(seed())
