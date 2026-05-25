import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchMyPage, type MyPageResponse, type LearningSessionOut } from "../api/auth";
import { useAuthContext } from "../context/AuthContext";

// 같은 책에 대한 여러 세션을 하나로 합친다.
//   - 완료(completed) 세션이 하나라도 있으면 그 책은 '완료'
//   - 없으면 '진행 중'(가장 최근 세션 사용)
//   - 점수는 표시되는 세션의 avg_recognition_accuracy 사용
function dedupeByBook(sessions: LearningSessionOut[]): LearningSessionOut[] {
  const map = new Map<string, LearningSessionOut>();
  // 최근순 정렬 (started_at 내림차순) 후 처리하면 첫 등장이 가장 최근 세션
  const sorted = [...sessions].sort(
    (a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
  );
  for (const s of sorted) {
    const key = s.book_id ?? `__title:${s.book_title ?? "unknown"}`;
    const existing = map.get(key);
    if (!existing) {
      map.set(key, s);
      continue;
    }
    // 이미 있는데 새 세션이 completed면 그걸로 교체 (완료 우선)
    if (s.status === "completed" && existing.status !== "completed") {
      map.set(key, s);
    }
  }
  // 표시 순서: 최근 활동(시작일) 내림차순
  return [...map.values()].sort(
    (a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
  );
}

export default function MyPage() {
  const { user, logout } = useAuthContext();
  const navigate = useNavigate();
  const [data, setData] = useState<MyPageResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    fetchMyPage()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  // ⚠️ 훅 호출은 항상 같은 순서로 — early return보다 먼저 선언.
  // 책 단위로 합쳐서 한 동화책당 한 줄만 노출
  const uniqueSessions = useMemo(
    () => dedupeByBook(data?.sessions ?? []),
    [data?.sessions],
  );

  const scoreColor = (s: number) =>
    s >= 0.75 ? "#10b981" : s >= 0.5 ? "#f59e0b" : "#ef4444";

  const statusLabel = (s: string) =>
    s === "completed" ? "완료" : s === "abandoned" ? "중단" : "진행 중";
  const statusClass = (s: string) =>
    s === "completed" ? "badge-completed" : s === "abandoned" ? "badge-abandoned" : "badge-progress";

  if (loading) return <div className="page-loading">불러오는 중…</div>;
  if (!data) return <div className="page-error"><p>학습 기록을 불러오지 못했습니다.</p></div>;

  const { total_sessions, completed_sessions, avg_score } = data;
  const nickname = data.user.nickname;

  return (
    <div className="mypage">
      {/* 헤더 */}
      <header className="mypage-header">
        <button className="btn-back" onClick={() => navigate("/books")}>← 홈</button>
        <h1 className="mypage-title">마이페이지</h1>
        <button className="btn-logout" onClick={handleLogout}>로그아웃</button>
      </header>

      {/* 프로필 카드 */}
      <div className="profile-card">
        <div className="profile-avatar">{nickname[0]}</div>
        <div className="profile-info">
          <p className="profile-name">{nickname}</p>
          <p className="profile-sub">@{data.user.username}</p>
        </div>
      </div>

      {/* 통계 */}
      <div className="stats-row">
        <div className="stat-card">
          <span className="stat-value">{total_sessions}</span>
          <span className="stat-label">총 학습</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{completed_sessions}</span>
          <span className="stat-label">완료</span>
        </div>
        <div className="stat-card">
          <span className="stat-value" style={{ color: avg_score != null ? scoreColor(avg_score) : "#94a3b8" }}>
            {avg_score != null ? `${Math.round(avg_score * 100)}점` : "-"}
          </span>
          <span className="stat-label">평균 이해도</span>
        </div>
      </div>

      {/* 학습 이력 */}
      <section className="history-section">
        <h2 className="history-title">학습 이력</h2>
        {uniqueSessions.length === 0 ? (
          <div className="history-empty">
            <p>아직 학습 기록이 없어요.</p>
            <button className="btn-play" onClick={() => navigate("/books")}>동화책 읽기</button>
          </div>
        ) : (
          <ul className="history-list">
            {uniqueSessions.map((s) => (
              <SessionItem
                key={s.id}
                session={s}
                expanded={expanded === s.id}
                onToggle={() => setExpanded(expanded === s.id ? null : s.id)}
                scoreColor={scoreColor}
                statusLabel={statusLabel}
                statusClass={statusClass}
              />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function SessionItem({
  session, expanded, onToggle, scoreColor, statusLabel, statusClass,
}: {
  session: LearningSessionOut;
  expanded: boolean;
  onToggle: () => void;
  scoreColor: (s: number) => string;
  statusLabel: (s: string) => string;
  statusClass: (s: string) => string;
}) {
  const date = new Date(session.started_at).toLocaleDateString("ko-KR", {
    year: "numeric", month: "short", day: "numeric",
  });

  return (
    <li className="history-item">
      <div className="history-item-header" onClick={onToggle}>
        <div className="history-item-left">
          <span className={`badge ${statusClass(session.status)}`}>
            {statusLabel(session.status)}
          </span>
          <div>
            <p className="history-book-title">{session.book_title ?? "알 수 없는 책"}</p>
            <p className="history-date">{date}</p>
          </div>
        </div>
        <div className="history-item-right">
          {session.avg_recognition_accuracy != null && (
            <span
              className="history-score"
              style={{ color: scoreColor(session.avg_recognition_accuracy) }}
            >
              {Math.round(session.avg_recognition_accuracy * 100)}점
            </span>
          )}
          <span className="history-chevron">{expanded ? "▲" : "▼"}</span>
        </div>
      </div>

      {/* QA 기록 펼치기 */}
      {expanded && session.qa_records.length > 0 && (
        <ul className="qa-history-list">
          {session.qa_records.map((q, i) => (
            <li key={q.id} className="qa-history-item">
              <p className="qa-history-q"><strong>Q{i + 1}.</strong> {q.question_text}</p>
              {q.user_answer_text && (
                <p className="qa-history-a">내 답: {q.user_answer_text}</p>
              )}
              {q.recognition_accuracy != null && (
                <span
                  className="qa-history-score"
                  style={{ color: scoreColor(q.recognition_accuracy) }}
                >
                  {Math.round(q.recognition_accuracy * 100)}점
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
      {expanded && session.qa_records.length === 0 && (
        <p className="qa-history-empty">Q&A 기록이 없습니다.</p>
      )}
    </li>
  );
}
