/**
 * QASection — 섹션 이해도 Q&A (멀티 라운드 루프)
 *
 * 흐름 (최대 2 라운드):
 *   Round 1: T5 질문 생성 → 아동 답변 → T5 평가
 *     - score >= 0.6 → 결과 표시
 *     - score < 0.6  → Round 2: 힌트 포함 후속 질문 생성 → 아동 답변 → T5 평가 → 결과 표시
 *   결과: 라운드별 점수 + 평균 점수 → onComplete(avgScore) 콜백
 */
import { useState } from "react";
import {
  generateQuestion,
  evaluateAnswer,
  generateFollowup,
  type EvaluateResponse,
} from "../../api/qa";

type Phase =
  | "idle"
  | "loading_q"
  | "answering"
  | "loading_eval"
  | "loading_followup"
  | "result"
  | "error";

interface RoundRecord {
  question: string;
  answer: string;
  score: number;
  feedback: string;
  hint: string;
}

interface QASectionProps {
  sectionText: string;
  sectionTitle?: string;
  onComplete?: (avgScore: number) => void;
}

const FOLLOWUP_THRESHOLD = 0.6; // 이 미만이면 라운드 2 진입
const MAX_ROUNDS = 2;

export default function QASection({ sectionText, sectionTitle, onComplete }: QASectionProps) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [round, setRound] = useState(1);
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [rounds, setRounds] = useState<RoundRecord[]>([]);
  const [errorMsg, setErrorMsg] = useState("");

  // ── 초기 질문 생성 ──────────────────────────────────────────────────────────
  const handleStart = async () => {
    setPhase("loading_q");
    setErrorMsg("");
    setRounds([]);
    setRound(1);
    try {
      const res = await generateQuestion(sectionText);
      setCurrentQuestion(res.question);
      setAnswer("");
      setPhase("answering");
    } catch {
      setErrorMsg("질문 생성에 실패했습니다. AI 엔진을 확인하세요.");
      setPhase("error");
    }
  };

  // ── 답변 제출 + 평가 ────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (!answer.trim()) return;
    setPhase("loading_eval");
    try {
      const evalRes: EvaluateResponse = await evaluateAnswer(
        currentQuestion,
        sectionText,
        answer
      );

      const record: RoundRecord = {
        question: currentQuestion,
        answer,
        score: evalRes.score,
        feedback: evalRes.feedback,
        hint: evalRes.correct_hint,
      };
      const newRounds = [...rounds, record];
      setRounds(newRounds);

      const shouldContinue =
        evalRes.score < FOLLOWUP_THRESHOLD && round < MAX_ROUNDS;

      if (shouldContinue) {
        // 라운드 2: 후속 질문 생성
        setPhase("loading_followup");
        const followRes = await generateFollowup(
          sectionText,
          currentQuestion,
          answer,
          evalRes.feedback
        );
        setCurrentQuestion(followRes.question);
        setAnswer("");
        setRound(2);
        setPhase("answering");
      } else {
        // 최종 결과
        const avg =
          newRounds.reduce((s, r) => s + r.score, 0) / newRounds.length;
        onComplete?.(avg);
        setPhase("result");
      }
    } catch {
      setErrorMsg("답변 평가에 실패했습니다.");
      setPhase("error");
    }
  };

  // ── 리셋 ───────────────────────────────────────────────────────────────────
  const handleReset = () => {
    setPhase("idle");
    setRound(1);
    setCurrentQuestion("");
    setAnswer("");
    setRounds([]);
    setErrorMsg("");
  };

  // ── 평균 점수 계산 ──────────────────────────────────────────────────────────
  const avgScore = rounds.length
    ? rounds.reduce((s, r) => s + r.score, 0) / rounds.length
    : 0;

  const scoreColor = (s: number) =>
    s >= 0.75 ? "#10b981" : s >= 0.5 ? "#f59e0b" : "#ef4444";

  return (
    <div className="qa-section">
      <div className="qa-header">
        <span className="qa-icon">🧠</span>
        <span className="qa-title">이해도 확인</span>
        {sectionTitle && <span className="qa-subtitle">— {sectionTitle}</span>}
      </div>

      {/* ── idle ── */}
      {phase === "idle" && (
        <button className="btn-ask" onClick={handleStart}>
          질문 받기
        </button>
      )}

      {/* ── 질문 생성 로딩 ── */}
      {(phase === "loading_q" || phase === "loading_followup") && (
        <div className="qa-loading">
          <div className="spinner" />
          <span>
            {phase === "loading_q"
              ? "질문 생성 중… (pko-T5)"
              : "후속 질문 생성 중…"}
          </span>
        </div>
      )}

      {/* ── 답변 입력 ── */}
      {(phase === "answering" || phase === "loading_eval") && (
        <div className="qa-answering">
          {round === 2 && (
            <div className="qa-round-badge">💡 힌트 포함 질문 (라운드 2)</div>
          )}
          <p className="qa-question">{currentQuestion}</p>
          <textarea
            className="qa-input"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            placeholder="답을 입력하세요…"
            rows={3}
            disabled={phase === "loading_eval"}
          />
          <div className="qa-actions">
            <button
              className="btn-submit"
              onClick={handleSubmit}
              disabled={!answer.trim() || phase === "loading_eval"}
            >
              {phase === "loading_eval" ? "평가 중…" : "제출"}
            </button>
            <button className="btn-skip" onClick={handleReset}>
              건너뛰기
            </button>
          </div>
        </div>
      )}

      {/* ── 최종 결과 ── */}
      {phase === "result" && (
        <div className="qa-result">
          {/* 라운드별 Q&A 요약 */}
          {rounds.map((r, i) => (
            <div key={i} className="qa-round-summary">
              <p className="qa-question-recap">
                <strong>Q{i + 1}:</strong> {r.question}
              </p>
              <p className="qa-answer-recap">내 답: {r.answer}</p>

              <div className="score-wrap">
                <div className="score-bar-bg">
                  <div
                    className="score-bar"
                    style={{
                      width: `${Math.round(r.score * 100)}%`,
                      background: scoreColor(r.score),
                    }}
                  />
                </div>
                <span className="score-label" style={{ color: scoreColor(r.score) }}>
                  {Math.round(r.score * 100)}점
                </span>
              </div>

              <p className="qa-feedback">{r.feedback}</p>

              <details className="qa-hint-wrap">
                <summary>정답 힌트 보기</summary>
                <p className="qa-hint">{r.hint}</p>
              </details>
            </div>
          ))}

          {/* 평균 최종 점수 */}
          {rounds.length > 1 && (
            <div className="qa-final-score">
              <span>최종 평균 점수</span>
              <span
                className="qa-final-score-value"
                style={{ color: scoreColor(avgScore) }}
              >
                {Math.round(avgScore * 100)}점
              </span>
            </div>
          )}

          <button className="btn-retry" onClick={handleReset}>
            다시 질문 받기
          </button>
        </div>
      )}

      {/* ── 에러 ── */}
      {phase === "error" && (
        <div className="qa-error">
          <p>⚠ {errorMsg}</p>
          <button onClick={handleReset}>다시 시도</button>
        </div>
      )}
    </div>
  );
}
