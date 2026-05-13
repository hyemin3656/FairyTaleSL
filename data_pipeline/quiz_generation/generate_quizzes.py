"""
generate_quizzes.py — 동화 섹션별 단답형 퀴즈 사전 생성 (Gemini 2.5 Flash)

흐름:
  DB(BookSection) → Gemini 2.5 Flash → JSON 파일
  생성된 JSON은 backend/scripts/load_quizzes.py 로 DB에 적재한다.

사전 요구:
  cd backend && source .venv/bin/activate
  pip install google-genai python-dotenv

실행 (프로젝트 루트에서):
  python -m data_pipeline.quiz_generation.generate_quizzes \\
      --book-id <UUID> [--output <path.json>] [--max-retries 2]

또는 모든 책 일괄:
  python -m data_pipeline.quiz_generation.generate_quizzes --all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

# ── path 보정 ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from google import genai                                       # noqa: E402
from google.genai import types as genai_types                  # noqa: E402
from sqlalchemy import select                                  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession                # noqa: E402

from core.database import AsyncSessionLocal                  # noqa: E402
from models.book import Book, BookSection                      # noqa: E402

# ── 상수 ─────────────────────────────────────────────────────────────────
PROMPT_PATH = Path(__file__).parent / "prompts" / "quiz_prompt.txt"
DEFAULT_OUT_DIR = Path(__file__).parent / "output"
MODEL_NAME = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
API_KEY = os.environ.get("GEMINI_API_KEY")

# ── Gemini 호출 ──────────────────────────────────────────────────────────
def build_prompt(section_text: str) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{section_text}", section_text)


def call_gemini(client: genai.Client, prompt: str) -> dict:
    """Gemini 호출 후 JSON dict 반환. 응답 형식 검증 포함."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini empty response")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # ```json ... ``` 펜스 형태 fallback
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise RuntimeError(f"non-JSON response: {text[:200]}")
        data = json.loads(m.group(0))

    for k in ("question", "expected_answer", "synonyms"):
        if k not in data:
            raise RuntimeError(f"missing field '{k}' in response: {data}")
    if not isinstance(data["synonyms"], list):
        raise RuntimeError(f"'synonyms' must be list: {data}")
    return data


def generate_for_section(
    client: genai.Client, section_text: str, max_retries: int = 2
) -> dict:
    """1개 섹션에 대해 퀴즈 생성. JSON 파싱 실패 시 재시도."""
    last_err = None
    for attempt in range(1, max_retries + 2):
        try:
            return call_gemini(client, build_prompt(section_text))
        except Exception as e:
            last_err = e
            print(f"  retry {attempt}: {e}", file=sys.stderr)
    raise RuntimeError(f"Gemini failed after {max_retries+1} attempts: {last_err}")


# ── DB 조회 ──────────────────────────────────────────────────────────────
async def fetch_book(db: AsyncSession, book_id: uuid.UUID) -> tuple[Book, list[BookSection]]:
    book = (await db.execute(select(Book).where(Book.id == book_id))).scalar_one_or_none()
    if book is None:
        raise SystemExit(f"book not found: {book_id}")
    sections = (
        await db.execute(
            select(BookSection)
            .where(BookSection.book_id == book_id)
            .order_by(BookSection.order)
        )
    ).scalars().all()
    return book, list(sections)


async def fetch_all_books(db: AsyncSession) -> list[Book]:
    return list((await db.execute(select(Book).order_by(Book.created_at))).scalars().all())


# ── 메인 처리 ────────────────────────────────────────────────────────────
async def run_for_book(book_id: uuid.UUID, output_path: Path, max_retries: int):
    if not API_KEY:
        raise SystemExit("GEMINI_API_KEY not set in .env")
    client = genai.Client(api_key=API_KEY)

    async with AsyncSessionLocal() as db:
        book, sections = await fetch_book(db, book_id)

    print(f"[generate] book={book.title} ({len(sections)} sections), model={MODEL_NAME}")

    quizzes = []
    for sec in sections:
        print(f"  section {sec.order}: {(sec.title or sec.text)[:30]}...")
        data = generate_for_section(client, sec.text, max_retries=max_retries)
        quizzes.append({
            "section_order": sec.order,
            "question": data["question"].strip(),
            "expected_answer": data["expected_answer"].strip(),
            "synonyms": [s.strip() for s in data["synonyms"] if isinstance(s, str)],
        })

    output = {
        "book_id": str(book.id),
        "book_title": book.title,
        "model": MODEL_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "quizzes": quizzes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[generate] wrote {output_path} ({len(quizzes)} quizzes)")


def safe_filename(title: str) -> str:
    return re.sub(r"[^\w\-가-힣]+", "_", title).strip("_") or "book"


async def main_async(args):
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUT_DIR

    if args.all:
        async with AsyncSessionLocal() as db:
            books = await fetch_all_books(db)
        for b in books:
            out = out_dir / f"{safe_filename(b.title)}.json"
            await run_for_book(b.id, out, args.max_retries)
        return

    if not args.book_id:
        raise SystemExit("either --book-id or --all is required")
    book_uuid = uuid.UUID(args.book_id)
    out = Path(args.output) if args.output else out_dir / f"{book_uuid}.json"
    await run_for_book(book_uuid, out, args.max_retries)


def main():
    ap = argparse.ArgumentParser(description="Generate single-answer quizzes per book section via Gemini.")
    ap.add_argument("--book-id", help="대상 책 UUID")
    ap.add_argument("--all", action="store_true", help="DB의 모든 책에 대해 일괄 생성")
    ap.add_argument("--output", help="단일 책 출력 JSON 경로")
    ap.add_argument("--output-dir", help="일괄 모드 출력 디렉토리")
    ap.add_argument("--max-retries", type=int, default=2, help="섹션당 LLM 호출 재시도 횟수")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
