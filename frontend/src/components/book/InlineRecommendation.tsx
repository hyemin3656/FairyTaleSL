/**
 * InlineRecommendation — 책 선택 페이지의 가로 추천 슬라이더
 *
 * - 5권 추천 (학습 완료 책과 같은 카테고리 우선, 없으면 랜덤)
 * - 가로 스크롤 + 좌우 ← → 버튼으로 수동 이동
 * - 자동 슬라이드 없음
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchBooks, type BookListItem } from "../../api/books";
import { fetchMyPage } from "../../api/auth";
import BookCard from "./BookCard";

const MAX_RECOMMENDATIONS = 5;

export default function InlineRecommendation() {
  const [books, setBooks] = useState<BookListItem[]>([]);
  const [reason, setReason] = useState<"category" | "random" | null>(null);
  const [loading, setLoading] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    (async () => {
      try {
        const [allBooksRes, my] = await Promise.all([
          fetchBooks({}),
          fetchMyPage().catch(() => null),
        ]);
        const all = allBooksRes.items;

        if (!my) {
          setBooks(shuffle(all).slice(0, MAX_RECOMMENDATIONS));
          setReason("random");
          return;
        }

        const completedIds = new Set<string>();
        for (const s of my.sessions) {
          if (s.status === "completed" && s.book_id) completedIds.add(s.book_id);
        }
        const completedCats = new Set<string>();
        for (const b of all) {
          if (completedIds.has(b.id)) b.categories.forEach((c) => completedCats.add(c));
        }

        // 1순위: 같은 카테고리의 미완료 책
        const sameCategoryCandidates = all.filter(
          (b) => !completedIds.has(b.id) && b.categories.some((c) => completedCats.has(c)),
        );
        if (sameCategoryCandidates.length > 0) {
          setBooks(shuffle(sameCategoryCandidates).slice(0, MAX_RECOMMENDATIONS));
          setReason("category");
          return;
        }

        // 2순위: 랜덤(완료한 것 제외)
        const others = all.filter((b) => !completedIds.has(b.id));
        const pool = others.length > 0 ? others : all;
        setBooks(shuffle(pool).slice(0, MAX_RECOMMENDATIONS));
        setReason("random");
      } catch (e) {
        console.error("recommendation failed:", e);
        setBooks([]);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const headerLabel = useMemo(() => {
    if (reason === "category") return "지금까지 읽은 책과 비슷한 이야기";
    if (reason === "random") return "오늘은 이런 이야기 어때요?";
    return "추천 동화";
  }, [reason]);

  const scrollByDelta = (delta: number) => {
    scrollRef.current?.scrollBy({ left: delta, behavior: "smooth" });
  };

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

  if (books.length === 0) return null;

  return (
    <aside style={wrapStyle}>
      <div style={accentBarStyle} />
      <div style={headerStyle}>
        <div>
          <div style={eyebrowRowStyle}>
            <span style={badgePillStyle}>✨ 추천</span>
            <span style={eyebrowStyle}>FOR YOU</span>
          </div>
          <h3 style={titleStyle}>{headerLabel}</h3>
        </div>
        <div style={navBtnsStyle}>
          <button onClick={() => scrollByDelta(-260)} style={arrowBtnStyle} aria-label="이전">‹</button>
          <button onClick={() => scrollByDelta(260)} style={arrowBtnStyle} aria-label="다음">›</button>
        </div>
      </div>

      <div ref={scrollRef} style={trackStyle}>
        {books.map((b) => (
          <div key={b.id} style={cardWrapStyle}>
            <BookCard book={b} />
          </div>
        ))}
      </div>
    </aside>
  );
}

// ── 유틸 ─────────────────────────────────────────────────────────────────
function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ── 스타일 ───────────────────────────────────────────────────────────────
const wrapStyle: React.CSSProperties = {
  position: "relative",
  width: "100%",
  padding: "22px 22px 20px",
  background:
    "linear-gradient(135deg, #ede9fe 0%, #fae8ff 55%, #fce7f3 100%)",
  border: "1.5px solid #c4b5fd",
  borderRadius: 18,
  display: "flex",
  flexDirection: "column",
  gap: 14,
  boxShadow:
    "0 10px 26px -14px rgba(124, 58, 237, 0.35), 0 2px 4px rgba(15, 23, 42, 0.04)",
  overflow: "hidden",
};
const accentBarStyle: React.CSSProperties = {
  position: "absolute",
  top: 0,
  left: 0,
  bottom: 0,
  width: 5,
  background: "linear-gradient(180deg, #7c3aed 0%, #ec4899 100%)",
};
const headerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "flex-end",
  justifyContent: "space-between",
  gap: 12,
  paddingLeft: 4,
};
const eyebrowRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  marginBottom: 6,
};
const badgePillStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 800,
  color: "white",
  background: "linear-gradient(135deg, #7c3aed 0%, #db2777 100%)",
  padding: "3px 10px",
  borderRadius: 999,
  letterSpacing: "0.3px",
  boxShadow: "0 2px 4px rgba(124, 58, 237, 0.3)",
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
const navBtnsStyle: React.CSSProperties = {
  display: "flex",
  gap: 6,
};
const arrowBtnStyle: React.CSSProperties = {
  width: 32,
  height: 32,
  borderRadius: "50%",
  background: "white",
  color: "#4f46e5",
  border: "1px solid #c7d2fe",
  cursor: "pointer",
  fontSize: 20,
  fontWeight: 700,
  boxShadow: "0 2px 6px rgba(79,70,229,0.12)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};
const trackStyle: React.CSSProperties = {
  display: "flex",
  gap: 14,
  overflowX: "auto",
  scrollSnapType: "x mandatory",
  paddingBottom: 4,
  scrollbarWidth: "thin",
};
const cardWrapStyle: React.CSSProperties = {
  flex: "0 0 230px",
  scrollSnapAlign: "start",
};
const skeletonStyle: React.CSSProperties = {
  width: "100%",
  height: 180,
  background: "linear-gradient(90deg, #e2e8f0 0%, #f1f5f9 50%, #e2e8f0 100%)",
  backgroundSize: "200% 100%",
  borderRadius: 14,
  animation: "shimmer 1.5s linear infinite",
};
