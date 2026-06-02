/**
 * BookReadPage — FairyTaleSL 학습 시나리오 호스트
 *
 * 흐름 (scenarioStore.mode 기반):
 *   STORY_PLAYING  : 아바타가 섹션 본문을 수어로 재생. 일시정지 / 따라해보기 진입 가능.
 *   PAUSED         : 재개 / 아이가 질문하기 진입
 *   CHILD_QUESTION : 수어로 자유 질문 → Gemini 응답 → 아바타가 답변 재생
 *   SECTION_DONE   : 따라하기(선택) → 퀴즈(필수) → 다음 섹션
 *   COMPLETED      : 결과 요약
 *
 * 토큰 인덱스 보존:
 *   PAUSED → resume()은 ws 서버가 보존한 위치에서 이어감.
 *   CHILD_QUESTION을 거쳐 답변을 재생하면 ws 위치가 변경되므로,
 *   "질문 마치기" 후에는 사용자가 "재생"을 다시 눌러 섹션을 재시작한다.
 *   (TODO: ws 서버에 seek 기능 추가 시 자동 이어감 가능)
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { fetchBookDetail, type BookDetail } from "../api/books";
import { fetchBookQuizzes, type QuizOut } from "../api/quizzes";
import {
  startSession as apiStartSession,
  completeSession as apiCompleteSession,
  saveSectionResult,
} from "../api/auth";
import { useGlossWS } from "../hooks/useGlossWS";
import { useScenarioStore } from "../stores/scenarioStore";
import { useAvatarStore } from "../stores/avatarStore";

import AvatarScene from "../components/avatar/AvatarScene";
import FollowAlongPanel from "../components/session/FollowAlongPanel";
import ChildQuestionPanel from "../components/session/ChildQuestionPanel";
import QuizPanel from "../components/session/QuizPanel";

export default function BookReadPage() {
  const { bookId } = useParams<{ bookId: string }>();
  const navigate = useNavigate();
  const avatarUrl = useAvatarStore((s) => s.selectedAvatar().url);

  const [book, setBook] = useState<BookDetail | null>(null);
  const [fetchError, setFetchError] = useState("");
  const [quizzes, setQuizzes] = useState<Map<number, QuizOut>>(new Map());

  // scenarioStore
  const mode             = useScenarioStore((s) => s.mode);
  const sectionIdx       = useScenarioStore((s) => s.sectionIdx);
  const totalSections    = useScenarioStore((s) => s.totalSections);
  const followAlongTarget= useScenarioStore((s) => s.followAlongTarget);
  const pendingQuiz      = useScenarioStore((s) => s.pendingQuiz);
  const results          = useScenarioStore((s) => s.results);

  const sessionId           = useScenarioStore((s) => s.sessionId);
  const startSessionStore   = useScenarioStore((s) => s.startSession);
  const pauseAction         = useScenarioStore((s) => s.pause);
  const resumeAction        = useScenarioStore((s) => s.resume);
  const sectionEndAction    = useScenarioStore((s) => s.sectionEnd);
  const skipPlaybackAction  = useScenarioStore((s) => s.skipPlayback);
  const enterFollowAlong    = useScenarioStore((s) => s.enterFollowAlong);
  const passFollowAlong     = useScenarioStore((s) => s.passFollowAlong);
  const skipFollowAlong     = useScenarioStore((s) => s.skipFollowAlong);
  const enterChildQuestion  = useScenarioStore((s) => s.enterChildQuestion);
  const exitChildQuestion   = useScenarioStore((s) => s.exitChildQuestion);
  const startQuiz           = useScenarioStore((s) => s.startQuiz);
  const finishQuiz          = useScenarioStore((s) => s.finishQuiz);
  const cancelQuizAction    = useScenarioStore((s) => s.cancelQuiz);
  const nextSectionAction   = useScenarioStore((s) => s.nextSection);
  const prevSectionAction   = useScenarioStore((s) => s.prevSection);
  const resetStore          = useScenarioStore((s) => s.reset);

  const ws = useGlossWS();

  // ── 초기 로딩: 책 + 퀴즈 prefetch + 세션 시작 ────────────────────────────
  useEffect(() => {
    if (!bookId) return;
    Promise.all([
      fetchBookDetail(bookId),
      fetchBookQuizzes(bookId).catch(() => [] as QuizOut[]),  // 퀴즈 없어도 학습은 가능
    ])
      .then(async ([b, qs]) => {
        setBook(b);
        const map = new Map<number, QuizOut>();
        qs.forEach((q) => map.set(q.section_order, q));
        setQuizzes(map);

        // 실 세션 발급 시도 (로그인된 경우만). 실패하면 메모리 전용 stub로 진행.
        let sid: string;
        try {
          const { session_id } = await apiStartSession(bookId);
          sid = session_id;
        } catch {
          sid = (crypto as { randomUUID?: () => string }).randomUUID?.() ?? "stub";
        }
        startSessionStore({
          bookId,
          sessionId: sid,
          totalSections: b.sections.length,
        });
      })
      .catch(() => setFetchError("책 정보를 불러오지 못했습니다."));
    return () => resetStore();
  }, [bookId, startSessionStore, resetStore]);

  const currentSection = book?.sections[sectionIdx];

  // ── 이미지 트리거: 글로스 재생 시점에 맞춰 이미지 인덱스 전환 ──────────────
  const [imgIdx, setImgIdx] = useState(0);

  useEffect(() => {
    setImgIdx(0);
  }, [sectionIdx]);

  useEffect(() => {
    if (!currentSection?.image_triggers || !ws.currentClip) return;
    const triggers = currentSection.image_triggers;
    for (let i = 1; i < triggers.length; i++) {
      if (triggers[i] && ws.currentClip.gloss === triggers[i]) {
        setImgIdx(i);
        break;
      }
    }
  }, [ws.currentClip, currentSection]);

  const activeImageUrl = (() => {
    const urls = currentSection?.image_urls;
    if (urls && urls.length > 0) return urls[imgIdx] ?? urls[0];
    return currentSection?.image_url ?? null;
  })();

  // ── ws.done → store.sectionEnd 자동 전이 ────────────────────────────────
  useEffect(() => {
    if (mode === "STORY_PLAYING" && ws.status === "done") {
      sectionEndAction();
    }
  }, [mode, ws.status, sectionEndAction]);

  // ── "빠르게 완료" 대기: 토큰이 도착하면 즉시 SECTION_DONE 전이 ─────────────
  // 따라해보기 기능은 ws.tokens가 있어야 가능하므로, sendText 직후 tokens 도착을 대기한다.
  const skipPendingRef = useRef(false);
  useEffect(() => {
    if (!skipPendingRef.current) return;
    if (ws.tokens.length === 0) return;
    if (mode !== "STORY_PLAYING" && mode !== "PAUSED") return;
    skipPendingRef.current = false;
    ws.pause();
    skipPlaybackAction();
  }, [ws.tokens.length, mode, ws, skipPlaybackAction]);

  // ── COMPLETED → 백엔드 세션 마감 (한 번만) ───────────────────────────────
  const completedSentRef = useRef(false);
  useEffect(() => {
    if (mode !== "COMPLETED" || completedSentRef.current) return;
    if (!sessionId || !localStorage.getItem("access_token")) return;
    completedSentRef.current = true;
    const correctCount = results.filter((r) => r.quizCorrect === true).length;
    const avg = totalSections > 0 ? correctCount / totalSections : 0;
    apiCompleteSession(sessionId, avg).catch(() => { /* 저장 실패 무시 */ });
  }, [mode, sessionId, results, totalSections]);

  // ── 핸들러 ────────────────────────────────────────────────────────────
  const handlePlay = () => {
    if (!currentSection) return;
    if (ws.status === "closed" || ws.status === "error") ws.reconnect();
    ws.sendText(currentSection.text);
  };

  const handlePause = () => {
    ws.pause();
    pauseAction(ws.currentIndex);
  };

  const handleResume = () => {
    ws.resume();
    resumeAction();
  };

  // ⏩ 아바타 재생을 건너뛰고 따라해보기/퀴즈로 즉시 진입.
  // 이미 토큰이 있으면 곧장 전이, 없으면 sendText 후 effect에서 토큰 도착 대기.
  const handleSkipPlayback = () => {
    if (!currentSection) return;
    if (ws.tokens.length > 0) {
      ws.pause();
      skipPlaybackAction();
      return;
    }
    skipPendingRef.current = true;
    if (ws.status === "closed" || ws.status === "error") ws.reconnect();
    if (ws.status === "idle" || ws.status === "done") {
      ws.sendText(currentSection.text);
    }
  };

  const handleEnterFollow = () => {
    // TODO: 고유명사 제외 정책 — BookSection.proper_nouns 도입 후 필터링
    const target = ws.tokens[0];
    if (!target) {
      // 토큰 없으면 곧장 스킵
      skipFollowAlong();
      return;
    }
    enterFollowAlong(target);
  };

  // 세션이 백엔드 발급(UUID v4)인지 확인. 메모리 stub은 발급 실패 시 같은 형식이지만,
  // localStorage access_token이 없으면 persist 호출 자체를 건너뛴다.
  const persistEnabled = () =>
    !!sessionId && !!localStorage.getItem("access_token");

  const persistSection = (payload: {
    follow_along_passed?: boolean | null;
    quiz_correct?: boolean | null;
    quiz_attempts?: number;
  }) => {
    if (!persistEnabled() || !currentSection) return;
    saveSectionResult(sessionId!, {
      section_order: currentSection.order,
      ...payload,
    }).catch(() => { /* 저장 실패는 학습 흐름에 영향 없음 */ });
  };

  const handleEnterChildQ = () => enterChildQuestion();

  const handleChildAnswer = (answer: string) => {
    // 답변 텍스트를 글로스로 변환해 아바타가 재생
    ws.sendText(answer);
  };

  const handleExitChildQ = () => {
    exitChildQuestion();   // → PAUSED
    // 사용자가 "재생"을 다시 눌러 섹션 재시작 (위 파일 헤더 참조)
  };

  const handleStartQuiz = () => {
    if (!currentSection) return;
    const quiz = quizzes.get(currentSection.order);
    if (!quiz) return;   // 퀴즈 없으면 버튼이 안 보이므로 도달 X (방어)
    startQuiz(quiz);
  };

  const handleQuizComplete = (r: { correct: boolean | null; attempts: number }) => {
    finishQuiz(r);
    persistSection({ quiz_correct: r.correct, quiz_attempts: r.attempts });
  };

  const handleNextSection = () => {
    nextSectionAction();
  };

  // 이전 섹션으로 돌아가기 — WS 토큰/클립 상태도 함께 초기화해야 새 섹션 본문이 깨끗이 재생됨
  const handlePrevSection = () => {
    if (sectionIdx <= 0) return;
    ws.reset();
    skipPendingRef.current = false;
    prevSectionAction();
  };

  // ── 파생 값 ───────────────────────────────────────────────────────────
  const currentResult = results.find((r) => r.sectionIdx === sectionIdx);
  const quizDoneForCurrent =
    !!currentResult && (currentResult.quizCorrect !== null || currentResult.quizAttempts > 0);
  const followDoneForCurrent =
    !!currentResult && currentResult.followAlongPassed !== null;
  // 이 섹션에 퀴즈가 존재하는가 (책마다 일부 섹션에만 사전 생성될 수 있음)
  const hasQuizForCurrent =
    !!currentSection && quizzes.has(currentSection.order);
  // 다음 섹션으로 넘어갈 수 있는 조건: 퀴즈 풀었거나, 애초에 퀴즈가 없는 섹션
  const canAdvance = quizDoneForCurrent || !hasQuizForCurrent;

  const scoreColor = (s: number) =>
    s >= 0.75 ? "#10b981" : s >= 0.5 ? "#f59e0b" : "#ef4444";

  const overallAvg = useMemo(() => {
    // 퀴즈가 존재하는 섹션만 분모로 사용
    const quizSectionsCount = quizzes.size;
    const correctCount = results.filter((r) => r.quizCorrect === true).length;
    return quizSectionsCount > 0 ? correctCount / quizSectionsCount : 0;
  }, [results, quizzes]);

  // ── 에러/로딩 ─────────────────────────────────────────────────────────
  if (fetchError) {
    return (
      <div className="page-error">
        <p>{fetchError}</p>
        <button onClick={() => navigate("/books")}>목록으로</button>
      </div>
    );
  }
  if (!book) return <div className="page-loading">불러오는 중…</div>;

  // ── 완료 요약 ─────────────────────────────────────────────────────────
  if (mode === "COMPLETED") {
    return (
      <div className="read-page">
        <div className="summary-card">
          <h2>🎉 {book.title} 완료!</h2>
          <ul className="summary-list">
            {book.sections.map((sec, i) => {
              const r = results.find((x) => x.sectionIdx === i);
              const sectionHasQuiz = quizzes.has(sec.order);
              return (
                <li key={i} className="summary-item">
                  <span className="summary-section-name">{sec.title ?? `섹션 ${i + 1}`}</span>
                  <span className="summary-score">
                    {!sectionHasQuiz
                      ? "— 퀴즈 없음"
                      : r?.quizCorrect === true
                        ? "✅ 정답"
                        : r?.quizCorrect === false
                          ? `❌ ${r.quizAttempts}회 시도`
                          : "정답보기"}
                  </span>
                </li>
              );
            })}
          </ul>
          <div className="summary-overall">
            <span>전체 정답률</span>
            <span style={{ color: scoreColor(overallAvg) }}>
              {Math.round(overallAvg * 100)}%
            </span>
          </div>
          <button className="btn-play" onClick={() => navigate("/books")}>목록으로</button>
        </div>
      </div>
    );
  }

  const isPlaying = ws.status === "streaming";
  const canPlay   = ws.status === "idle" || ws.status === "done";
  const wsError   = ws.status === "error";

  // ── 메인 렌더 ─────────────────────────────────────────────────────────
  return (
    <div className="read-page">
      {/* 헤더 */}
      <header className="read-header">
        <button className="btn-back" onClick={() => navigate("/books")}>← 목록</button>
        <h1 className="read-title">{book.title}</h1>
        <span className="section-counter">{sectionIdx + 1} / {book.sections.length}</span>
        <button className="btn-avatar-change" onClick={() => navigate("/avatar-select")}>아바타 변경</button>
      </header>

      {/* 진행 점 */}
      <div className="section-progress-row">
        {book.sections.map((_, i) => {
          const r = results.find((x) => x.sectionIdx === i);
          const color =
            r?.quizCorrect === true ? "#10b981"
            : r?.quizCorrect === false ? "#ef4444"
            : r ? "#f59e0b"
            : undefined;
          return (
            <div
              key={i}
              className={`section-progress-dot ${i === sectionIdx ? "active" : ""}`}
              style={color ? { background: color, opacity: 1 } : {}}
            />
          );
        })}
      </div>

      <div className="read-body">
        {/* 좌측 컬럼: 이미지+아바타 패널 + 글로스 바 */}
        <div className="left-col">
          <section className="avatar-section">
            {/* 동화 이미지 */}
            {activeImageUrl && (
              <img className="section-image" src={activeImageUrl} alt="" aria-hidden />
            )}

            {/* 하단 그라데이션 오버레이 */}
            <div className="avatar-gradient" />

            {/* 아바타 — 하단 중앙 */}
            <div className="avatar-overlay-wrap">
              <div className="avatar-scene-wrap">
                <AvatarScene
                  clip={ws.currentClip}
                  status={ws.status}
                  currentIndex={ws.currentIndex}
                  total={ws.total}
                  frozen={mode !== "STORY_PLAYING"}
                  avatarUrl={avatarUrl}
                />
              </div>
            </div>

            {wsError && (
              <div className="ws-error avatar-ws-error">
                <span>⚠ {ws.errorMsg || "WS 오류"}</span>
                <button onClick={ws.reconnect}>재연결</button>
              </div>
            )}
          </section>

          {/* 글로스 바 — 이미지 아래 */}
          <div className="gloss-bar">
            <div className="gloss-overlay">
              {ws.status === "connecting" && (
                <span className="overlay-tag connecting">연결 중…</span>
              )}
              {ws.status === "streaming" && ws.currentClip && (
                <>
                  <span className={`overlay-tag ${ws.currentClip.is_fallback ? "fallback" : "matched"}`}>
                    {ws.currentClip.gloss}
                  </span>
                  <span className="overlay-progress">
                    {ws.currentIndex + 1} / {ws.total}
                  </span>
                </>
              )}
              {ws.status === "done" && (
                <span className="overlay-tag done">완료</span>
              )}
            </div>
            {ws.tokens.length > 0 && (
              <div className="gloss-tokens">
                {ws.tokens.map((t, i) => (
                  <span
                    key={i}
                    className={`token ${i === ws.currentIndex && ws.status === "streaming" ? "active" : ""}`}
                  >
                    {t}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* 텍스트 + 컨트롤 + 모드별 패널 */}
        <section className="text-section">
          {currentSection?.title && <h2 className="section-title">{currentSection.title}</h2>}
          <p className="section-text">{currentSection?.text}</p>

          {/* STORY_PLAYING: 수어로 보기 / 재생 중 / 일시정지 / 넘어가기 */}
          {mode === "STORY_PLAYING" && (
            <div className="read-controls">
              {sectionIdx >= 1 && (
                <button
                  className="btn-prev-section"
                  onClick={handlePrevSection}
                  title="이전 섹션 내용으로 돌아가요."
                >
                  ← 이전 섹션
                </button>
              )}

              <button
                className="btn-play"
                onClick={handlePlay}
                disabled={!canPlay && !wsError && !isPlaying}
              >
                {isPlaying && !ws.paused
                  ? "▶ 재생 중…"
                  : ws.status === "connecting"
                    ? "연결 중…"
                    : "▶ 수어로 보기"}
              </button>

              {isPlaying && (
                <button className="btn-pause" onClick={handlePause}>⏸ 일시정지</button>
              )}

              <button
                className="btn-skip-playback"
                onClick={handleSkipPlayback}
                disabled={skipPendingRef.current}
                title="아바타 수어 생성을 건너뛰고 따라해보기/퀴즈로 바로 이동합니다."
              >
                {skipPendingRef.current ? "⏳ 준비 중…" : "⏩ 넘어가기"}
              </button>
            </div>
          )}

          {/* PAUSED: 재개 / 질문하기 / 넘어가기 */}
          {mode === "PAUSED" && (
            <div className="read-controls">
              {sectionIdx >= 1 && (
                <button
                  className="btn-prev-section"
                  onClick={handlePrevSection}
                  title="이전 섹션 내용으로 돌아가요."
                >
                  ← 이전 섹션
                </button>
              )}
              <button className="btn-pause" onClick={handleResume}>▶ 재개</button>
              <button className="btn-practice" onClick={handleEnterChildQ}>
                ❓ 질문하기
              </button>
              <button
                className="btn-skip-playback"
                onClick={handleSkipPlayback}
                disabled={skipPendingRef.current}
              >
                {skipPendingRef.current ? "⏳ 준비 중…" : "⏩ 넘어가기"}
              </button>
            </div>
          )}

          {/* CHILD_QUESTION 패널 */}
          {mode === "CHILD_QUESTION" && currentSection && (
            <ChildQuestionPanel
              storyContext={book.sections.map((s, i) => {
                const head = s.title ? `${i + 1}. ${s.title}` : `${i + 1}.`;
                return `[${head}]\n${s.text}`;
              }).join("\n\n")}
              onAnswer={handleChildAnswer}
              onExit={handleExitChildQ}
              answerGlossTokens={ws.tokens}
            />
          )}

          {/* FOLLOW_ALONG 패널 */}
          {mode === "FOLLOW_ALONG" && followAlongTarget && (
            <FollowAlongPanel
              targetGloss={followAlongTarget}
              onPass={() => {
                passFollowAlong();
                persistSection({ follow_along_passed: true });
              }}
              onSkip={() => {
                skipFollowAlong();
                persistSection({ follow_along_passed: false });
              }}
            />
          )}

          {/* QUIZ 패널 */}
          {mode === "QUIZ" && pendingQuiz && (
            <QuizPanel
              quiz={pendingQuiz}
              onComplete={handleQuizComplete}
              onCancel={cancelQuizAction}
            />
          )}

          {/* SECTION_DONE: 단계별 카드로 명확하게 구분 */}
          {mode === "SECTION_DONE" && (
            <div className="section-done">
              {sectionIdx >= 1 && (
                <button
                  className="btn-prev-section btn-prev-section-inline"
                  onClick={handlePrevSection}
                  title="이전 섹션 내용으로 돌아가요."
                >
                  ← 이전 섹션으로
                </button>
              )}

              {/* 1단계 — 따라해보기 (선택) */}
              {!followDoneForCurrent && ws.tokens.length > 0 && (
                <div className="step-card step-optional">
                  <div className="step-head">
                    <span className="step-badge">선택</span>
                    <span className="step-title">🤟 따라해보기</span>
                  </div>
                  <p className="step-desc">
                    아바타가 보여준 수어 중 하나를 직접 따라해 봐요.
                  </p>
                  <button className="btn-practice" onClick={handleEnterFollow}>
                    따라해보기 시작
                  </button>
                </div>
              )}

              {/* 2단계 — 퀴즈 (필수) */}
              {hasQuizForCurrent && !quizDoneForCurrent && (
                <div className="step-card step-required">
                  <div className="step-head">
                    <span className="step-badge step-badge-required">필수</span>
                    <span className="step-title">🧠 퀴즈 풀기</span>
                  </div>
                  <p className="step-desc">
                    퀴즈를 풀어야 <strong>다음 섹션</strong>으로 넘어갈 수 있어요.
                  </p>
                  <button className="btn-solve-cta" onClick={handleStartQuiz}>
                    퀴즈 풀기 시작
                  </button>
                </div>
              )}

              {/* 3단계 — 다음 섹션으로 진행 */}
              {canAdvance && (
                <div className="step-card step-advance">
                  <div className="step-head">
                    <span className="step-title">
                      {sectionIdx + 1 < totalSections ? "다음 섹션으로" : "학습 마치기"}
                    </span>
                  </div>
                  {quizDoneForCurrent && (
                    <p className="step-desc">잘 했어요! 다음 단계로 이동할 수 있어요.</p>
                  )}
                  <button className="btn-play" onClick={handleNextSection}>
                    {sectionIdx + 1 < totalSections ? "다음 섹션 →" : "📊 결과 보기"}
                  </button>
                </div>
              )}

              {!hasQuizForCurrent && (
                <p className="section-note">이 섹션은 퀴즈가 준비되지 않았어요.</p>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
