import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { fetchBookDetail, type BookDetail, type BookSection } from "../api/books";
import { useGlossWS } from "../hooks/useGlossWS";
import AvatarScene from "../components/avatar/AvatarScene";
import QASection from "../components/qa/QASection";

export default function BookReadPage() {
  const { bookId } = useParams<{ bookId: string }>();
  const navigate = useNavigate();

  const [book, setBook] = useState<BookDetail | null>(null);
  const [fetchError, setFetchError] = useState("");
  const [sectionIdx, setSectionIdx] = useState(0);

  // 섹션별 Q&A 평균 점수 (null = 아직 미완료)
  const [sectionScores, setSectionScores] = useState<(number | null)[]>([]);

  // 전체 완료 요약 표시 여부
  const [showSummary, setShowSummary] = useState(false);

  const {
    status, currentClip, currentIndex, total, tokens, errorMsg,
    sendText, reset, reconnect,
  } = useGlossWS();

  useEffect(() => {
    if (!bookId) return;
    fetchBookDetail(bookId)
      .then((b) => {
        setBook(b);
        setSectionScores(new Array(b.sections.length).fill(null));
      })
      .catch(() => setFetchError("책 정보를 불러오지 못했습니다."));
  }, [bookId]);

  const currentSection: BookSection | undefined = book?.sections[sectionIdx];

  const handlePlay = () => {
    if (!currentSection) return;
    if (status === "closed" || status === "error") reconnect();
    sendText(currentSection.text);
  };

  const changeSection = (delta: number) => {
    setSectionIdx((i) => i + delta);
    reset();
  };

  // QASection 완료 콜백 — 해당 섹션에 점수 기록
  const handleQAComplete = (avgScore: number) => {
    setSectionScores((prev) => {
      const next = [...prev];
      next[sectionIdx] = avgScore;
      return next;
    });
  };

  // 모든 섹션 Q&A 완료 여부
  const allDone =
    book != null && sectionScores.length > 0 && sectionScores.every((s) => s !== null);

  const overallAvg =
    allDone
      ? (sectionScores as number[]).reduce((a, b) => a + b, 0) / sectionScores.length
      : 0;

  const scoreColor = (s: number) =>
    s >= 0.75 ? "#10b981" : s >= 0.5 ? "#f59e0b" : "#ef4444";

  // ── 에러 / 로딩 ──────────────────────────────────────────────
  if (fetchError) {
    return (
      <div className="page-error">
        <p>{fetchError}</p>
        <button onClick={() => navigate("/books")}>목록으로</button>
      </div>
    );
  }
  if (!book) return <div className="page-loading">불러오는 중…</div>;

  const isPlaying  = status === "streaming";
  const canPlay    = status === "idle" || status === "done";
  const wsError    = status === "error";

  // ── 전체 완료 요약 ───────────────────────────────────────────
  if (showSummary) {
    return (
      <div className="read-page">
        <div className="summary-card">
          <h2>🎉 {book.title} 완료!</h2>
          <p className="summary-subtitle">모든 섹션의 이해도 결과입니다.</p>

          <ul className="summary-list">
            {book.sections.map((sec, i) => {
              const score = sectionScores[i];
              return (
                <li key={i} className="summary-item">
                  <span className="summary-section-name">
                    {sec.title ?? `섹션 ${i + 1}`}
                  </span>
                  {score !== null ? (
                    <span
                      className="summary-score"
                      style={{ color: scoreColor(score) }}
                    >
                      {Math.round(score * 100)}점
                    </span>
                  ) : (
                    <span className="summary-score-na">미완료</span>
                  )}
                </li>
              );
            })}
          </ul>

          <div className="summary-overall">
            <span>전체 평균</span>
            <span
              className="summary-overall-score"
              style={{ color: scoreColor(overallAvg) }}
            >
              {Math.round(overallAvg * 100)}점
            </span>
          </div>

          <div className="summary-actions">
            <button className="btn-nav" onClick={() => setShowSummary(false)}>
              다시 보기
            </button>
            <button className="btn-play" onClick={() => navigate("/books")}>
              목록으로
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="read-page">
      {/* 헤더 */}
      <header className="read-header">
        <button className="btn-back" onClick={() => navigate("/books")}>
          ← 목록
        </button>
        <h1 className="read-title">{book.title}</h1>
        <span className="section-counter">
          {sectionIdx + 1} / {book.sections.length}
        </span>
      </header>

      {/* 섹션별 점수 진행 바 */}
      <div className="section-progress-row">
        {book.sections.map((_, i) => {
          const score = sectionScores[i];
          return (
            <div
              key={i}
              className={`section-progress-dot ${i === sectionIdx ? "active" : ""}`}
              title={score !== null ? `${Math.round(score * 100)}점` : "미완료"}
              style={
                score !== null
                  ? { background: scoreColor(score), opacity: 1 }
                  : {}
              }
            />
          );
        })}
      </div>

      <div className="read-body">
        {/* 3D 아바타 */}
        <section className="avatar-section">
          <AvatarScene
            clip={currentClip}
            status={status}
            currentIndex={currentIndex}
            total={total}
            tokens={tokens}
          />

          {wsError && (
            <div className="ws-error">
              <span>⚠ {errorMsg || "WS 오류"}</span>
              <button onClick={reconnect}>재연결</button>
            </div>
          )}
        </section>

        {/* 텍스트 + 컨트롤 */}
        <section className="text-section">
          {currentSection?.title && (
            <h2 className="section-title">{currentSection.title}</h2>
          )}
          <p className="section-text">{currentSection?.text}</p>

          <div className="read-controls">
            <button
              className="btn-play"
              onClick={handlePlay}
              disabled={!canPlay && !wsError}
            >
              {isPlaying
                ? "▶ 재생 중…"
                : status === "connecting"
                  ? "연결 중…"
                  : "▶ 수어로 보기"}
            </button>

            {(status === "done" || canPlay) && tokens.length > 0 && (
              <button
                className="btn-practice"
                onClick={() => {
                  const gloss = tokens[0];
                  navigate(
                    `/practice?gloss=${encodeURIComponent(gloss)}&bookId=${bookId}&section=${sectionIdx}`
                  );
                }}
              >
                🤟 따라해보기
              </button>
            )}
          </div>

          {/* Q&A — 섹션 재생 완료 후 표시 */}
          {status === "done" && currentSection && (
            <QASection
              sectionText={currentSection.text}
              sectionTitle={currentSection.title ?? undefined}
              onComplete={handleQAComplete}
            />
          )}

          <div className="section-nav">
            <button
              className="btn-nav"
              onClick={() => changeSection(-1)}
              disabled={sectionIdx === 0}
            >
              ← 이전
            </button>

            {sectionIdx < book.sections.length - 1 ? (
              <button
                className="btn-nav"
                onClick={() => changeSection(1)}
              >
                다음 →
              </button>
            ) : (
              /* 마지막 섹션 → 결과 요약 버튼 */
              <button
                className="btn-play"
                onClick={() => setShowSummary(true)}
              >
                📊 결과 보기
              </button>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
