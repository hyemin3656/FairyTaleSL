"""
load_quizzes.py — generate_quizzes.py 출력 JSON 을 quizzes 테이블에 적재.

실행 (backend 루트에서):
  cd backend && source .venv/bin/activate
  python scripts/load_quizzes.py --input ../data_pipeline/quiz_generation/output/<book>.json

옵션:
  --replace : 기존 (book_id, section_order) 행을 덮어쓴다 (기본은 skip)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

# backend root 보정
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select                  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession # noqa: E402

from core.database import AsyncSessionLocal  # noqa: E402
from models.book import Book                  # noqa: E402
from models.quiz import Quiz                   # noqa: E402


async def load(input_path: Path, replace: bool):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    book_title = payload.get("book_title")
    quizzes = payload["quizzes"]

    async with AsyncSessionLocal() as db:  # type: AsyncSession
        book = (await db.execute(
            select(Book).where(Book.title == book_title)
        )).scalar_one_or_none()
        if book is None:
            print(f"[load] skip: book '{book_title}' not found in DB")
            return
        book_id = book.id
        print(f"[load] book='{book_title}' id={book_id} count={len(quizzes)} replace={replace}")

        for q in quizzes:
            existing = (await db.execute(
                select(Quiz).where(
                    Quiz.book_id == book_id,
                    Quiz.section_order == q["section_order"],
                )
            )).scalar_one_or_none()

            if existing and not replace:
                print(f"  skip section_order={q['section_order']} (exists)")
                continue
            if existing and replace:
                existing.question = q["question"]
                existing.expected_answer = q["expected_answer"]
                existing.synonyms = q["synonyms"]
            else:
                db.add(Quiz(
                    book_id=book_id,
                    section_order=q["section_order"],
                    question=q["question"],
                    expected_answer=q["expected_answer"],
                    synonyms=q["synonyms"],
                ))
        await db.commit()
    print("[load] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="generate_quizzes.py 출력 JSON 경로")
    ap.add_argument("--replace", action="store_true", help="기존 행 덮어쓰기")
    args = ap.parse_args()
    asyncio.run(load(Path(args.input), args.replace))


if __name__ == "__main__":
    main()
