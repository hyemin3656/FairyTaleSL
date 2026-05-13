"""학습 시나리오 통합 자동 검증 (Phase 7)

목적:
  - Phase 6에서 추가한 section_results 저장 흐름이 끝-끝 작동하는지 확인
  - 인증 → 세션 시작 → 섹션별 결과 upsert → 퀴즈 정답 검사(정규화) → 세션 마감 → /me 조회

실행:
  cd backend
  ./.venv/bin/python -m scripts.test_scenario_flow

전제:
  - 환경변수 DATABASE_URL을 sqlite로 오버라이드하여 임시 DB에서 검증
  - main 앱을 ASGITransport로 in-process 실행 (서버 띄우지 않음)
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path

# ── 1. 환경 오버라이드 (반드시 백엔드 모듈 import 전에) ───────────────────────
_tmp = tempfile.NamedTemporaryFile(prefix="psycho_test_", suffix=".db", delete=False)
_tmp.close()
TEST_DB_PATH = _tmp.name
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH}"
os.environ["APP_ENV"] = "test"
os.environ["SECRET_KEY"] = "test-secret-key-32-bytes-min-length"

# 백엔드 루트를 sys.path 에 추가 (backend/ 디렉토리에서 실행한다고 가정)
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# ── 2. 모듈 import (env 적용된 상태) ──────────────────────────────────────────
import httpx
from httpx import ASGITransport

from core.database import Base, engine, AsyncSessionLocal
import models  # noqa: F401 — Base.metadata에 등록되도록 import
from models.book import Book, BookSection
from models.quiz import Quiz
from models.section_result import SectionResult
from main import app


# ── 3. 헬퍼 ──────────────────────────────────────────────────────────────────
_RED = "\033[31m"
_GRN = "\033[32m"
_YLW = "\033[33m"
_END = "\033[0m"


class TestFailure(Exception):
    pass


def check(cond: bool, label: str, detail: str = ""):
    if cond:
        print(f"  {_GRN}✓{_END} {label}")
    else:
        print(f"  {_RED}✗{_END} {label}  {detail}")
        raise TestFailure(label)


# ── 4. 시드 데이터 ────────────────────────────────────────────────────────────
async def seed() -> tuple[uuid.UUID, list[uuid.UUID]]:
    """책 1권 + 섹션 2개 + 퀴즈 2개. (book_id, [quiz_id1, quiz_id2]) 반환"""
    async with AsyncSessionLocal() as db:
        book = Book(
            title="테스트 동화",
            description="자동 검증용",
            categories=["동물"],
            author="tester",
        )
        db.add(book)
        await db.flush()

        s1 = BookSection(book_id=book.id, order=1, title="첫 장", text="옛날에 호랑이가 살았어요.")
        s2 = BookSection(book_id=book.id, order=2, title="둘째 장", text="호랑이는 외로웠어요.")
        db.add_all([s1, s2])

        q1 = Quiz(
            book_id=book.id, section_order=1,
            question="누가 살았나요?",
            expected_answer="호랑이",
            synonyms=["호랑이요", "호랑이에요"],
        )
        q2 = Quiz(
            book_id=book.id, section_order=2,
            question="호랑이의 기분은 어땠나요?",
            expected_answer="외로워요",
            synonyms=["외로움", "쓸쓸해요"],
        )
        db.add_all([q1, q2])

        await db.commit()
        return book.id, [q1.id, q2.id]


# ── 5. 메인 시나리오 ──────────────────────────────────────────────────────────
async def run() -> None:
    print(f"{_YLW}[설정]{_END} DB={TEST_DB_PATH}")

    # 5-1. 테이블 생성
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print(f"  {_GRN}✓{_END} 테이블 생성 ({len(Base.metadata.tables)}개)")

    book_id, quiz_ids = await seed()
    print(f"  {_GRN}✓{_END} 시드 데이터 (book_id={book_id})")

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        # ── 1) 회원가입 + 로그인 ────────────────────────────────────────────
        print(f"\n{_YLW}[1] 인증{_END}")
        r = await client.post("/api/v1/auth/register", json={
            "email": "tester@example.com",
            "username": "tester",
            "nickname": "테스터",
            "password": "pw123456",
        })
        check(r.status_code == 201, "POST /auth/register → 201", r.text)
        token_r = r.json()["access_token"]
        check(bool(token_r), "register access_token 반환")

        r = await client.post("/api/v1/auth/login", json={
            "username": "tester", "password": "pw123456",
        })
        check(r.status_code == 200, "POST /auth/login → 200", r.text)
        token = r.json()["access_token"]
        auth = {"Authorization": f"Bearer {token}"}

        # ── 2) 책 + 퀴즈 prefetch ──────────────────────────────────────────
        print(f"\n{_YLW}[2] 책/퀴즈 조회{_END}")
        r = await client.get(f"/api/v1/books/{book_id}/quizzes")
        check(r.status_code == 200, "GET /books/{id}/quizzes → 200", r.text)
        body = r.json()
        check(len(body) == 2, f"퀴즈 2건 반환 (실제 {len(body)})")
        check(
            "expected_answer" not in body[0] and "synonyms" not in body[0],
            "QuizOut에 expected_answer/synonyms 미노출 (보안)",
            str(body[0].keys()),
        )

        # ── 3) 세션 시작 ───────────────────────────────────────────────────
        print(f"\n{_YLW}[3] 세션 시작{_END}")
        r = await client.post("/api/v1/sessions", json={"book_id": str(book_id)}, headers=auth)
        check(r.status_code == 201, "POST /sessions → 201", r.text)
        session_id = r.json()["session_id"]

        # ── 4) 퀴즈 정답 검사 (오답 → 정답 + 정규화) ───────────────────────
        print(f"\n{_YLW}[4] /quizzes/{{id}}/check 정규화{_END}")
        q1_id = str(quiz_ids[0])
        r = await client.post(f"/api/v1/quizzes/{q1_id}/check", json={"user_answer": "사자"})
        check(r.status_code == 200, "오답 → 200")
        check(r.json()["correct"] is False, "오답 → correct=false")
        check(r.json()["expected_answer"] is None, "오답 (reveal=false) → expected_answer 미노출")

        # 정규화 검증: '호랑이가' (조사 '가' 제거) → 'expected_answer'='호랑이'와 일치
        r = await client.post(f"/api/v1/quizzes/{q1_id}/check", json={"user_answer": "호랑이가"})
        check(r.json()["correct"] is True, "'호랑이가' → 조사 제거 후 정답 매칭")
        check(r.json()["expected_answer"] == "호랑이", "정답 시 expected_answer 노출")

        # 동의어 매칭
        r = await client.post(f"/api/v1/quizzes/{q1_id}/check", json={"user_answer": "호랑이요"})
        check(r.json()["correct"] is True, "동의어 '호랑이요' → 정답")

        # reveal=true: 오답이어도 expected_answer 노출
        r = await client.post(f"/api/v1/quizzes/{q1_id}/check?reveal=true", json={"user_answer": "사자"})
        check(
            r.json()["correct"] is False and r.json()["expected_answer"] == "호랑이",
            "reveal=true → 오답이어도 정답 노출",
        )

        # ── 5) 섹션별 결과 upsert (Phase 6 핵심) ───────────────────────────
        print(f"\n{_YLW}[5] /sessions/{{id}}/section-result upsert{_END}")
        # 5-a 섹션1: 따라하기 통과
        r = await client.post(
            f"/api/v1/sessions/{session_id}/section-result",
            json={"section_order": 1, "follow_along_passed": True},
            headers=auth,
        )
        check(r.status_code == 201, "섹션1 follow_along_passed=true → 201", r.text)

        # 5-b 섹션1: 퀴즈 정답 (같은 row 덮어쓰기)
        r = await client.post(
            f"/api/v1/sessions/{session_id}/section-result",
            json={"section_order": 1, "quiz_correct": True, "quiz_attempts": 1},
            headers=auth,
        )
        check(r.status_code == 201, "섹션1 quiz upsert → 201")

        # 5-c 섹션2: 따라하기 스킵 + 퀴즈 정답보기 (correct=None)
        r = await client.post(
            f"/api/v1/sessions/{session_id}/section-result",
            json={"section_order": 2, "follow_along_passed": False},
            headers=auth,
        )
        check(r.status_code == 201, "섹션2 follow_along_passed=false")

        r = await client.post(
            f"/api/v1/sessions/{session_id}/section-result",
            json={"section_order": 2, "quiz_correct": None, "quiz_attempts": 0},
            headers=auth,
        )
        check(r.status_code == 201, "섹션2 quiz_correct=null (정답보기)")

        # 5-d 존재하지 않는 세션 ID → 404
        bogus = uuid.uuid4()
        r = await client.post(
            f"/api/v1/sessions/{bogus}/section-result",
            json={"section_order": 1, "follow_along_passed": True},
            headers=auth,
        )
        check(r.status_code == 404, "없는 세션 → 404")

        # ── 6) DB 직조회로 upsert 결과 검증 ────────────────────────────────
        print(f"\n{_YLW}[6] DB 검증{_END}")
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            rows = (await db.execute(
                select(SectionResult).where(SectionResult.session_id == uuid.UUID(session_id))
                .order_by(SectionResult.section_order)
            )).scalars().all()
            check(len(rows) == 2, f"section_results 행수 2 (실제 {len(rows)})")
            r1, r2 = rows
            check(
                r1.section_order == 1 and r1.follow_along_passed is True
                and r1.quiz_correct is True and r1.quiz_attempts == 1,
                "섹션1: follow=true, quiz_correct=true, attempts=1",
                f"got {r1.follow_along_passed} / {r1.quiz_correct} / {r1.quiz_attempts}",
            )
            check(
                r2.section_order == 2 and r2.follow_along_passed is False
                and r2.quiz_correct is None and r2.quiz_attempts == 0,
                "섹션2: follow=false, quiz_correct=null, attempts=0",
                f"got {r2.follow_along_passed} / {r2.quiz_correct} / {r2.quiz_attempts}",
            )

        # ── 7) 세션 마감 ───────────────────────────────────────────────────
        print(f"\n{_YLW}[7] 세션 마감{_END}")
        r = await client.patch(
            f"/api/v1/sessions/{session_id}",
            json={"status": "completed", "avg_recognition_accuracy": 0.5},
            headers=auth,
        )
        check(r.status_code == 200, "PATCH /sessions/{id} → 200")

        # ── 8) /me 마이페이지 ──────────────────────────────────────────────
        print(f"\n{_YLW}[8] GET /sessions/me{_END}")
        r = await client.get("/api/v1/sessions/me", headers=auth)
        check(r.status_code == 200, "GET /sessions/me → 200", r.text)
        me = r.json()
        check(me["total_sessions"] == 1, f"total_sessions=1 (실제 {me['total_sessions']})")
        check(me["completed_sessions"] == 1, "completed_sessions=1")
        sess = me["sessions"][0]
        check(sess["status"] == "completed", "세션 status=completed")
        check(sess["last_section_order"] == 2, f"last_section_order=2 (실제 {sess['last_section_order']})")

        # ── 9) 비로그인 거부 ───────────────────────────────────────────────
        print(f"\n{_YLW}[9] 인증 가드{_END}")
        r = await client.post("/api/v1/sessions", json={"book_id": str(book_id)})
        check(r.status_code == 401, "토큰 없이 POST /sessions → 401")

        r = await client.post(
            f"/api/v1/sessions/{session_id}/section-result",
            json={"section_order": 1, "follow_along_passed": True},
        )
        check(r.status_code == 401, "토큰 없이 section-result → 401")


async def main() -> int:
    try:
        await run()
    except TestFailure as e:
        print(f"\n{_RED}✗ 실패: {e}{_END}")
        return 1
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\n{_RED}✗ 예외: {e}{_END}")
        return 2
    finally:
        await engine.dispose()
        try:
            os.unlink(TEST_DB_PATH)
        except OSError:
            pass
    print(f"\n{_GRN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{_END}")
    print(f"{_GRN}✓ 통합 검증 통과{_END}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
