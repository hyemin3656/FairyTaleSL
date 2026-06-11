import { useEffect, useState } from "react";
import { fetchBooks, fetchCategories, type BookListItem } from "../../api/books";
import BookCard from "./BookCard";
import InlineRecommendation from "./InlineRecommendation";
import SafeBoundary from "../common/SafeBoundary";

export default function BookList() {
  const [books, setBooks] = useState<BookListItem[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [keyword, setKeyword] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 카테고리 목록 초기 로드
  useEffect(() => {
    fetchCategories().then(setCategories).catch(console.error);
  }, []);

  // 책 목록 조회 (keyword, selectedCategories 변경 시)
  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchBooks({ keyword: keyword || undefined, categories: selectedCategories })
      .then((res) => {
        setBooks(res.items);
        setTotal(res.total);
      })
      .catch(() => setError("동화책 목록을 불러오지 못했습니다."))
      .finally(() => setLoading(false));
  }, [keyword, selectedCategories]);

  const handleSearch = () => {
    setKeyword(inputValue.trim());
  };

  const toggleCategory = (cat: string) => {
    setSelectedCategories((prev) =>
      prev.includes(cat) ? prev.filter((c) => c !== cat) : [...prev, cat]
    );
  };

  return (
    <div style={containerStyle}>
      {/* 검색바 */}
      <div style={searchBarStyle}>
        <input
          style={inputStyle}
          type="text"
          placeholder="동화책 제목이나 내용으로 검색..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
        />
        <button style={searchBtnStyle} onClick={handleSearch}>
          검색
        </button>
      </div>

      {/* 카테고리 필터 */}
      <div style={categoryFilterStyle}>
        {categories.map((cat) => (
          <button
            key={cat}
            style={
              selectedCategories.includes(cat)
                ? { ...categoryBtnStyle, ...categoryBtnActiveStyle }
                : categoryBtnStyle
            }
            onClick={() => toggleCategory(cat)}
          >
            {cat}
          </button>
        ))}
        {selectedCategories.length > 0 && (
          <button
            style={clearBtnStyle}
            onClick={() => setSelectedCategories([])}
          >
            초기화
          </button>
        )}
      </div>

      {/* 결과 수 */}
      <p style={resultCountStyle}>
        {keyword || selectedCategories.length > 0
          ? `검색 결과 ${total}건`
          : `전체 ${total}권`}
      </p>

      {/* 추천 — 검색/필터 미적용 상태에서만 노출 */}
      {!keyword && selectedCategories.length === 0 && (
        <SafeBoundary label="InlineRecommendation">
          <InlineRecommendation />
        </SafeBoundary>
      )}

      {/* 상태별 UI */}
      {loading && <div style={stateBoxStyle}>불러오는 중...</div>}

      {!loading && error && <div style={{ ...stateBoxStyle, color: "#ef4444" }}>{error}</div>}

      {!loading && !error && books.length === 0 && (
        <div style={emptyStyle}>
          <span style={{ fontSize: 48 }}>🔍</span>
          <p>검색 결과가 없습니다.</p>
        </div>
      )}

      {!loading && !error && books.length > 0 && (
        <div style={gridStyle}>
          {books.map((book) => (
            <BookCard key={book.id} book={book} />
          ))}
        </div>
      )}
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  maxWidth: 1100,
  margin: "0 auto",
  padding: "24px 16px",
  display: "flex",
  flexDirection: "column",
  gap: 16,
};

const searchBarStyle: React.CSSProperties = {
  display: "flex",
  gap: 8,
};

const inputStyle: React.CSSProperties = {
  flex: 1,
  padding: "10px 16px",
  fontSize: 15,
  border: "1.5px solid var(--color-border)",
  borderRadius: 8,
  outline: "none",
  background: "var(--color-surface)",
};

const searchBtnStyle: React.CSSProperties = {
  padding: "10px 20px",
  background: "var(--color-primary)",
  color: "#fff",
  borderRadius: 8,
  fontSize: 15,
  fontWeight: 600,
};

const categoryFilterStyle: React.CSSProperties = {
  display: "flex",
  gap: 8,
  flexWrap: "wrap",
};

const categoryBtnStyle: React.CSSProperties = {
  padding: "6px 16px",
  borderRadius: 20,
  fontSize: 13,
  fontWeight: 500,
  border: "1.5px solid var(--color-border)",
  background: "var(--color-surface)",
  color: "var(--color-text-muted)",
  transition: "all 0.12s",
};

const categoryBtnActiveStyle: React.CSSProperties = {
  background: "var(--color-primary)",
  color: "#fff",
  borderColor: "var(--color-primary)",
};

const clearBtnStyle: React.CSSProperties = {
  ...categoryBtnStyle,
  color: "#ef4444",
  borderColor: "#fecaca",
};

const resultCountStyle: React.CSSProperties = {
  fontSize: 13,
  color: "var(--color-text-muted)",
};

const gridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
  gap: 20,
};

const stateBoxStyle: React.CSSProperties = {
  textAlign: "center",
  padding: "60px 0",
  color: "var(--color-text-muted)",
};

const emptyStyle: React.CSSProperties = {
  textAlign: "center",
  padding: "60px 0",
  color: "var(--color-text-muted)",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: 12,
};
