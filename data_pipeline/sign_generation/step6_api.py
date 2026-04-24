"""
수어 생성 Motion DB API (FastAPI)
localhost:8001에서 실행

엔드포인트:
  GET /motion          - 전체 글로스 목록
  GET /motion/{gloss}  - 글로스별 keypoint + 감정 반환
  GET /gloss?text=...  - 텍스트 → 글로스 시퀀스 변환
  GET /motion/missing  - keypoint 미등록 글로스 목록
"""
import json
import sqlite3
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from kiwipiepy import Kiwi

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "data/motion_db.sqlite"

app = FastAPI(title="수어 생성 Motion DB API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_kiwi = Kiwi()
_VALID_POS = {"NNG", "NNP", "VV", "VA", "MAG"}
_STOPWORDS = {"하다", "이다", "있다", "되다", "않다", "없다", "같다", "것", "수", "때"}

# KSL 어순 분류
_TIME_KW  = {"옛날", "어느날", "봄", "여름", "가을", "겨울", "아침", "저녁", "밤", "낮", "그때", "마지막", "처음", "결국"}
_PLACE_KW = {"산", "연못", "하늘", "마을", "집", "굴", "세상", "바다"}
_NEG_KW   = {"않다", "못하다", "아니다", "없다", "안"}


def _get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _tokenize(text: str) -> list[str]:
    """텍스트 → KSL 어순 글로스 토큰 리스트 (kiwipiepy)."""
    tokens = _kiwi.analyze(text)[0][0]
    time_w, place_w, nouns, verbs, adjs, advs, negs = [], [], [], [], [], [], []

    for token in tokens:
        word, pos3 = token.form, token.tag[:3]
        if pos3 not in _VALID_POS or len(word) <= 1 or word in _STOPWORDS:
            continue
        if pos3 in ("NNG", "NNP"):
            if word in _TIME_KW:    time_w.append(word)
            elif word in _PLACE_KW: place_w.append(word)
            else:                   nouns.append(word)
        elif pos3 == "VV":
            (negs if word in _NEG_KW else verbs).append(word)
        elif pos3 == "VA":
            adjs.append(word)
        elif pos3 == "MAG":
            (negs if word in _NEG_KW else advs).append(word)

    return time_w + place_w + nouns + advs + negs + verbs + adjs


@app.get("/motion")
def list_motions():
    """등록된 전체 글로스 목록."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT gloss, emotion_label, fallback_type, is_registered FROM motion_db ORDER BY id"
    ).fetchall()
    conn.close()
    return {
        "total": len(rows),
        "registered": sum(1 for r in rows if r["is_registered"]),
        "motions": [dict(r) for r in rows],
    }


@app.get("/motion/missing")
def get_missing():
    """keypoint 미등록 글로스 목록."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT gloss, fallback_type FROM motion_db WHERE is_registered = 0 ORDER BY gloss"
    ).fetchall()
    conn.close()
    return {"count": len(rows), "glosses": [dict(r) for r in rows]}


@app.get("/motion/{gloss}")
def get_motion(gloss: str):
    """글로스 → keypoint 벡터(225차원) + 감정 레이블 반환."""
    conn = _get_db()
    row = conn.execute("SELECT * FROM motion_db WHERE gloss = ?", (gloss,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"'{gloss}' 글로스가 DB에 없습니다.")

    keypoints = None
    if row["keypoint_path"] and Path(row["keypoint_path"]).exists():
        keypoints = np.load(row["keypoint_path"]).tolist()  # 225차원 list

    return {
        "gloss":         row["gloss"],
        "emotion_label": row["emotion_label"],
        "fallback_type": row["fallback_type"],
        "is_registered": bool(row["is_registered"]),
        "keypoints":     keypoints,   # 225차원 벡터 or null
        "video_path":    row["video_path"],
    }


@app.get("/gloss")
def text_to_gloss(text: str = Query(..., description="변환할 한국어 텍스트")):
    """텍스트 → KSL 글로스 시퀀스 + DB 등록 여부 반환."""
    tokens = _tokenize(text)

    conn = _get_db()
    result = []
    for token in tokens:
        row = conn.execute(
            "SELECT gloss, emotion_label, fallback_type, is_registered FROM motion_db WHERE gloss = ?",
            (token,)
        ).fetchone()
        result.append({
            "gloss":         token,
            "in_db":         row is not None,
            "is_registered": bool(row["is_registered"]) if row else False,
            "emotion_label": row["emotion_label"] if row else None,
            "fallback_type": row["fallback_type"] if row else None,
        })
    conn.close()

    return {
        "text":           text,
        "gloss_sequence": tokens,
        "details":        result,
    }
