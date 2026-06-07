/**
 * RecommendationCarousel — 학습 이력 기반 동화책 추천 + 광고형 슬라이드 캐러셀
 *
 * 추천 규칙:
 *   1) 완료(status="completed") 세션이 있으면 → 그 동화책들의 카테고리와
 *      겹치는 다른(아직 완료 안 한) 책을 우선 추천
 *   2) 없으면(=신규 사용자 또는 완료 없음) 랜덤
 *
 * UI:
 *   - 1장씩 자동 슬라이드 (4초)
 *   - 호버 시 자동 슬라이드 일시정지
 *   - 인디케이터 dots + 좌우 화살표
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchBooks, type BookListItem } from "../../api/books";
import { fetchMyPage } from "../../api/auth";
import BookCard from "../book/BookCard";

const SLIDE_INTERVAL_MS = 4000;
const MAX_RECOMMENDATIONS = 6;

export default function RecommendationCarousel() {
  const [books, setBooks] = useState<BookListItem[]>([]);
  const [reason, setReason] = useState<"category" | "random" | null>(null);
  const [loading, setLoading] = useState(true);
  const [idx, setIdx] = useState(0);
  const [paused, setPaused] = useState(false);

  // 추천 목록 계산
  useEffect(() => {
    (async () => {
      try {
        const [allBooksRes, my] = await Promise.all([
          fetchBooks({}),
          fetchMyPage().catch(() => null),
        ]);
        const all = allBooksRes.items;
        if (!my) {
          // 비로그인 또는 실패 → 랜덤
          setBooks(shuffle(all).slice(0, MAX_RECOMMENDATIONS));
          setReason("random");
          return;
        }
        // 완료된 책의 book_id 집합 + 카테고리 집합
        const completedIds = new Set<string>();
        const completedCats = new Set<string>();
        for (const s of my.sessions) {
          if (s.status === "completed" && s.book_id) {
            completedIds.add(s.book_id);
          }
        }
        // 완료된 책들의 카테고리 채우기 (books 목록에서 lookup)
        for (const b of all) {
          if (completedIds.has(b.id)) {
            b.categories.forEach((c) => completedCats.add(c));
          }
        }
        // 1) 같은 카테고리의 미완료 책
        const sameCategoryCandidates = all.filter(
          (b) =>
            !completedIds.has(b.id) &&
            b.categories.some((c) => completedCats.has(c)),
        );
        if (sameCategoryCandidates.length > 0) {
          setBooks(shuffle(sameCategoryCandidates).slice(0, MAX_RECOMMENDATIONS));
          setReason("category");
          return;
        }
        // 2) 없으면 랜덤(완료한 것 제외)
        const others = all.filter((b) => !completedIds.has(b.id));
        const pool = others.length > 0 ? others : all;
        setBooks(shuffle(pool).slice(0, MAX_RECOMMENDATIONS));
        setReason("random");
      } catch (e) {
        // 추천 실패는 조용히 — 빈 상태
        console.error("recommendation failed:", e);
        setBooks([]);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // 자동 슬라이드
  useEffect(() => {
    if (books.length < 2 || paused) return;
    const t = setInterval(() => {
      setIdx((cur) => (cur + 1) % books.length);
    }, SLIDE_INTERVAL_MS);
    return () => clearInterval(t);
  }, [books.length, paused]);

  const goPrev = useCallback(() => {
    setIdx((cur) => (cur - 1 + books.length) % books.length);
  }, [books.length]);
  const goNext = useCallback(() => {
    setIdx((cur) => (cur + 1) % books.length);
  }, [books.length]);

  const headerLabel = useMemo(() => {
    if (reason === "category") return "지금까지 읽은 책과 비슷한 이야기";
    if (reason === "random") return "오늘은 이런 이야기 어때요?";
    return "추천 동화";
  }, [reason]);

  if (loading) {
    return (
      <aside style={wrapStyle}>
        <div style={headerStyle}>
          <span style={eyebrowStyle}>FOR YOU</span>
          <h3 style={titleStyle}>추천 동화 준비 중…</h3>
        </div>
        <div style={skeletonStyle} />
      </aside>
    );
  }

  if (books.length === 0) {
    return null;
  }

  const current = books[idx];

  return (
    <aside
      style={wrapStyle}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >
      <div style={headerStyle}>
        <span style={eyebrowStyle}>FOR YOU</span>
        <h3 style={titleStyle}>{headerLabel}</h3>
      </div>

      <div style={cardContainerStyle}>
        {/* 슬라이드 카드 — BookCard가 자체 onClick으로 /books/:id로 이동 */}
        <div key={current.id} style={cardSlideStyle}>
          <BookCard book={current} />
        </div>

        {books.length > 1 && (
          <>
            <button onClick={goPrev} style={{ ...arrowBtnStyle, left: -8 }} aria-label="이전">‹</button>
            <button onClick={goNext} style={{ ...arrowBtnStyle, right: -8 }} aria-label="다음">›</button>
          </>
        )}
      </div>

      {books.length > 1 && (
        <div style={dotsRowStyle}>
          {books.map((_, i) => (
            <button
              key={i}
              onClick={() => setIdx(i)}
              style={{
                ...dotStyle,
                background: i === idx ? "#4f46e5" : "#cbd5e1",
                width: i === idx ? 22 : 8,
              }}
              aria-label={`${i + 1}번째 추천`}
            />
          ))}
        </div>
      )}
    </aside>
  );
}

// ── 유틸 ───────────────────────────────────────────────────────────────────
function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ── 스타일 ─────────────────────────────────────────────────────────────────
const wrapStyle: React.CSSProperties = {
  width: 340,
  padding: 18,
  background: "linear-gradient(180deg, #eef2ff 0%, #fdf4ff 100%)",
  border: "1px solid #e0e7ff",
  borderRadius: 18,
  display: "flex",
  flexDirection: "column",
  gap: 14,
  position: "relative",
  flexShrink: 0,
};
const headerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
};
const eyebrowStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 800,
  color: "#7c3aed",
  letterSpacing: "1.3px",
};
const titleStyle: React.CSSProperties = {
  fontSize: 16,
  fontWeight: 800,
  color: "#1e293b",
  margin: 0,
  letterSpacing: "-0.3px",
};
const cardContainerStyle: React.CSSProperties = {
  position: "relative",
  width: "100%",
};
const cardSlideStyle: React.CSSProperties = {
  width: "100%",
  cursor: "pointer",
  animation: "slideInRight 0.45s ease-out",
};
const arrowBtnStyle: React.CSSProperties = {
  position: "absolute",
  top: "40%",
  transform: "translateY(-50%)",
  width: 30,
  height: 30,
  borderRadius: "50%",
  background: "white",
  color: "#4f46e5",
  border: "1px solid #c7d2fe",
  cursor: "pointer",
  fontSize: 20,
  fontWeight: 700,
  boxShadow: "0 2px 8px rgba(79,70,229,0.18)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};
const dotsRowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "center",
  gap: 6,
  marginTop: 4,
};
const dotStyle: React.CSSProperties = {
  height: 8,
  borderRadius: 999,
  border: "none",
  cursor: "pointer",
  padding: 0,
  transition: "all 0.2s",
};
const skeletonStyle: React.CSSProperties = {
  width: "100%",
  height: 220,
  background: "linear-gradient(90deg, #e2e8f0 0%, #f1f5f9 50%, #e2e8f0 100%)",
  backgroundSize: "200% 100%",
  borderRadius: 14,
  animation: "shimmer 1.5s linear infinite",
};
