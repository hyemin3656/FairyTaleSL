"""
글로스 변환 파이프라인
텍스트 → 토큰(글로스 후보) → Motion DB 조회 → MotionClip 리스트 반환

형태소 분석: kiwipiepy (Java 불필요, 순수 Python)
KSL 어순: 시간 > 장소 > 명사 > 부사 > 부정 > 서술어
"""
import json
import sqlite3
import numpy as np
from pathlib import Path
from kiwipiepy import Kiwi

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.motion import GlossMotion
from schemas.motion import MotionClip

FALLBACK_GLOSS = "FALLBACK"
FALLBACK_URL = "/static/motions/fallback.glb"
FALLBACK_DURATION = 1.0

_GLOSS_LIST_PATH = Path(__file__).parent.parent.parent / \
    "data_pipeline/sign_generation/data/gloss_list.json"
_MOTION_DB_PATH = Path(__file__).parent.parent.parent / \
    "data_pipeline/sign_generation/data/motion_db.sqlite"


def _load_fallback_map() -> dict[str, dict]:
    try:
        with open(_GLOSS_LIST_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {
            g["gloss"]: {
                "type": g.get("fallback_type"),
                "glosses": g.get("fallback_glosses", []),
            }
            for g in data
            if g.get("fallback_type")
        }
    except FileNotFoundError:
        return {}

_FALLBACK_MAP: dict[str, dict] = _load_fallback_map()


KEYPOINT_FPS = 15.0  # step3에서 추출한 fps


def _load_from_sqlite(gloss: str) -> tuple[list[list[float]] | None, dict | None]:
    """motion_db.sqlite에서 keypoint_data(BLOB)와 anim_data(JSON)를 함께 조회."""
    if not _MOTION_DB_PATH.exists():
        return None, None
    try:
        conn = sqlite3.connect(str(_MOTION_DB_PATH))
        row = conn.execute(
            "SELECT keypoint_data, anim_data FROM motion_db WHERE gloss = ?", (gloss,)
        ).fetchone()
        conn.close()
        if not row:
            return None, None
        keypoints = None
        if row[0]:
            arr = np.frombuffer(row[0], dtype=np.float32)
            n_frames = max(1, len(arr) // 225)
            seq = arr[: n_frames * 225].reshape(n_frames, 225)
            keypoints = seq.tolist()
        anim_data = None
        if row[1]:
            try:
                anim_data = json.loads(row[1])
            except Exception:
                pass
        return keypoints, anim_data
    except Exception:
        pass
    return None, None


def _load_anim(gloss: str, keypoint_path: str | None) -> tuple[list[list[float]] | None, dict | None]:
    """keypoint + anim_data 로드: SQLite 우선, 없으면 .npy 파일 시도."""
    kp, anim = _load_from_sqlite(gloss)
    if kp:
        return kp, anim
    if not keypoint_path:
        return None, None
    p = Path(keypoint_path)
    if not p.exists() or p.suffix != ".npy":
        return None, None
    arr = np.load(str(p), allow_pickle=False).astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.tolist(), None

_kiwi = Kiwi()

# 동화에 등장하는 복합명사 — kiwipiepy가 분리하므로 사전에 등록
_COMPOUND_NOUNS = [
    "박씨", "알밤", "동아줄", "두레박", "오작교", "날개옷",
    "날개", "나무꾼", "공양미", "인당수", "단군왕검", "태백산",
    "금도끼", "은도끼", "쇠도끼", "연못가", "팥죽", "수수밭",
]
for _noun in _COMPOUND_NOUNS:
    try:
        _kiwi.add_user_word(_noun, "NNG", score=10)
    except Exception:
        pass

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
    한국어 텍스트 → 글로스 토큰 리스트 (텍스트 출현 순서 유지).

    kiwipiepy로 형태소 분석 후 내용어(명사, 동사, 형용사, 부사)만 추출.
    기능어(조사, 어미 등)는 제거. 중복 단어는 첫 출현만 유지.
    """
    result = _kiwi.analyze(text)
    tokens = result[0][0]  # 최적 분석 결과

    ordered: list[str] = []

    for token in tokens:
        word = token.form
        pos3 = token.tag[:3]

        if pos3 not in _VALID_POS:
            continue

        # 동사/형용사: 어간 → 기본형 (살→살다, 가→가다, 크→크다)
        if pos3 in ("VV", "VA") and not word.endswith("다"):
            word = word + "다"

        # 명사는 단음절(봄, 박, 소 등)도 허용; 동사/형용사는 "다" 붙으면 최소 2자
        if word in _STOPWORDS:
            continue
        if pos3 not in ("NNG", "NNP") and len(word) < 2:
            continue
        ordered.append(word)

    return ordered


async def _fetch_motion_map(
    db: AsyncSession, glosses: list[str]
) -> dict[str, GlossMotion]:
    """glosses + FALLBACK 을 한 번의 쿼리로 조회하여 {gloss: GlossMotion} 반환"""
    lookup = set(glosses) | {FALLBACK_GLOSS}
    stmt = select(GlossMotion).where(GlossMotion.gloss.in_(lookup))
    result = await db.execute(stmt)
    return {row.gloss: row for row in result.scalars().all()}


def _expand_tokens(tokens: list[str]) -> list[str]:
    """decompose fallback 적용: 토큰 → 대체 글로스로 확장."""
    expanded = []
    for token in tokens:
        fb = _FALLBACK_MAP.get(token)
        if fb and fb["type"] == "decompose" and fb["glosses"]:
            expanded.extend(fb["glosses"])
        else:
            expanded.append(token)
    return expanded


async def resolve_motions(
    db: AsyncSession, tokens: list[str]
) -> list[MotionClip]:
    """
    토큰 리스트를 받아 각 토큰에 대응하는 MotionClip 리스트를 반환.

    우선순위:
    1. DB에 해당 글로스가 있으면 해당 클립 반환
    2. decompose fallback: 등록된 대체 글로스 클립으로 교체
    3. text fallback: gltf_clip_url 없이 text_display만 설정
    4. 기본 FALLBACK 클립 사용
    """
    expanded = _expand_tokens(tokens)
    motion_map = await _fetch_motion_map(db, expanded)
    fallback = motion_map.get(FALLBACK_GLOSS)

    clips: list[MotionClip] = []
    for token in tokens:
        fb = _FALLBACK_MAP.get(token)

        # 1. DB에 직접 등록된 글로스
        motion = motion_map.get(token)
        if motion:
            kp_seq, anim_data = _load_anim(motion.gloss, motion.keypoint_path)
            duration = len(kp_seq) / KEYPOINT_FPS if kp_seq else motion.duration_sec
            clips.append(MotionClip(
                gloss=motion.gloss,
                gltf_clip_url=motion.gltf_clip_url,
                emotion_label=motion.emotion_label,
                blendshape_params=motion.blendshape_params or {},
                duration_sec=duration,
                is_fallback=False,
                keypoints=kp_seq,
                fps=KEYPOINT_FPS,
                animation_data=anim_data,
            ))

        # 2. decompose fallback
        elif fb and fb["type"] == "decompose":
            for alt_gloss in fb["glosses"]:
                alt_motion = motion_map.get(alt_gloss)
                if alt_motion:
                    alt_kp, alt_anim = _load_anim(alt_motion.gloss, alt_motion.keypoint_path)
                    alt_dur = len(alt_kp) / KEYPOINT_FPS if alt_kp else alt_motion.duration_sec
                    clips.append(MotionClip(
                        gloss=token,
                        gltf_clip_url=alt_motion.gltf_clip_url,
                        emotion_label=alt_motion.emotion_label,
                        blendshape_params=alt_motion.blendshape_params or {},
                        duration_sec=alt_dur,
                        is_fallback=True,
                        fallback_type="decompose",
                        keypoints=alt_kp,
                        fps=KEYPOINT_FPS,
                        animation_data=alt_anim,
                    ))

        # 3. text fallback
        elif fb and fb["type"] == "text":
            clips.append(MotionClip(
                gloss=token,
                gltf_clip_url="",
                emotion_label="neutral",
                blendshape_params={},
                duration_sec=FALLBACK_DURATION,
                is_fallback=True,
                fallback_type="text",
                text_display=token,
            ))

        # 4. 기본 FALLBACK
        else:
            clips.append(MotionClip(
                gloss=token,
                gltf_clip_url=fallback.gltf_clip_url if fallback else FALLBACK_URL,
                emotion_label=fallback.emotion_label if fallback else "neutral",
                blendshape_params=fallback.blendshape_params or {} if fallback else {},
                duration_sec=fallback.duration_sec if fallback else FALLBACK_DURATION,
                is_fallback=True,
                fallback_type=None,
            ))

    return clips
