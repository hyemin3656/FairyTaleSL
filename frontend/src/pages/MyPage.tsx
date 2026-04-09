import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchMyPage, type MyPageResponse, type LearningSessionOut } from "../api/auth";
import { useAuthContext } from "../context/AuthContext";

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

  const scoreColor = (s: number) =>
    s >= 0.75 ? "#10b981" : s >= 0.5 ? "#f59e0b" : "#ef4444";

  const statusLabel = (s: string) =>
    s === "completed" ? "완료" : s === "abandoned" ? "중단" : "진행 중";
  const statusClass = (s: string) =>
    s === "completed" ? "badge-completed" : s === "abandoned" ? "badge-abandoned" : "badge-progress";

  if (loading) return <div className="page-loading">불러오는 중…</div>;
  if (!data) return <div className="page-error"><p>학습 기록을 불러오지 못했습니다.</p></div>;

  const { sessions, total_sessions, completed_sessions, avg_score } = data;
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
        {sessions.length === 0 ? (
          <div className="history-empty">
            <p>아직 학습 기록이 없어요.</p>
            <button className="btn-play" onClick={() => navigate("/books")}>동화책 읽기</button>
          </div>
        ) : (
          <ul className="history-list">
            {sessions.map((s) => (
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
