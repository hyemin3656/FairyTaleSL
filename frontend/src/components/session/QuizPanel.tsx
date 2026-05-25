/**
 * QuizPanel — 저장된 퀴즈 (주 모드, 섹션 종료마다 필수)
 *
 * 흐름:
 *   - "풀기" 선택: 답안 입력 → /quizzes/{id}/check
 *     · 정답 → revealed (정답 표시) → 사용자가 "다음으로" 클릭 시 onComplete({correct: true})
 *     · 오답 → attempts++. MAX(2) 도달 시 정답 공개, "다음으로" 클릭 시 onComplete({correct: false})
 *   - "정답보기" 선택: 정답 공개 → "다음으로" 클릭 시 onComplete({correct: null, attempts: 0})
 *
 * UX 원칙:
 *   revealed 상태에 도달해도 즉시 부모로 알리지 않는다. 사용자가 정답을 확인하고
 *   "다음으로" 버튼을 누를 때 onComplete가 호출되어 패널이 닫힌다.
 */
import { useCallback, useRef, useState } from "react";
import WebcamCapture from "../webcam/WebcamCapture";
import type { TsnPrediction } from "../webcam/WebcamCapture";
import { checkQuizAnswer } from "../../api/quizzes";
import type { QuizOut } from "../../api/quizzes";

const MAX_RETRIES = 2;   // config.yaml::quiz.max_retries
const RECOGNITION_THRESHOLD = 0.5;   // ChildQuestionPanel과 동일 임계값

type Phase =
  | "choosing"         // 풀기/정답보기 선택
  | "answering"        // 입력
  | "submitting"       // /check 호출 중
  | "revealed";        // 정답 공개 (성공 or 마지막 실패 or 정답보기 선택)

type Outcome = "correct" | "wrong-final" | "revealed-only";
type InputMode = "sign" | "keyboard";

interface Props {
  quiz: QuizOut;
  onComplete: (r: { correct: boolean | null; attempts: number }) => void;
  onCancel?: () => void;
}

export default function QuizPanel({ quiz, onComplete, onCancel }: Props) {
  const [phase, setPhase] = useState<Phase>("choosing");
  const [inputMode, setInputMode] = useState<InputMode>("sign");
  const [typedAnswer, setTypedAnswer] = useState("");
  const [signLabels, setSignLabels] = useState<string[]>([]);
  const [attempts, setAttempts] = useState(0);
  const [lastResult, setLastResult] = useState<"correct" | "wrong" | null>(null);
  const [revealedAnswer, setRevealedAnswer] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const lastAppendedRef = useRef<string>("");

  // 현재 모드에서 제출할 답 문자열
  const currentAnswer = inputMode === "sign"
    ? signLabels.join(" ")
    : typedAnswer.trim();

  const handleSignPrediction = useCallback((pred: TsnPrediction) => {
    if (phase === "submitting") return;
    if (inputMode !== "sign") return;
    if (pred.is_dummy) return;
    if (pred.confidence < RECOGNITION_THRESHOLD) return;
    if (!pred.gloss) return;
    if (pred.gloss === lastAppendedRef.current) return;
    lastAppendedRef.current = pred.gloss;
    setSignLabels((arr) => [...arr, pred.gloss]);
  }, [phase, inputMode]);

  const handleClear = () => {
    if (inputMode === "sign") {
      setSignLabels([]);
      lastAppendedRef.current = "";
    } else {
      setTypedAnswer("");
    }
  };

  const handleSolve = () => {
    setPhase("answering");
  };

  const handleRevealOnly = async () => {
    setPhase("submitting");
    try {
      const res = await checkQuizAnswer(quiz.id, "", true);
      setRevealedAnswer(res.expected_answer);
      setOutcome("revealed-only");
      setPhase("revealed");
    } catch (e) {
      setErrorMsg((e as Error).message);
      setPhase("answering");
    }
  };

  const handleSubmit = async () => {
    if (!currentAnswer || phase === "submitting") return;
    const nextAttempts = attempts + 1;
    const isFinalAttempt = nextAttempts >= MAX_RETRIES;
    setPhase("submitting");
    setErrorMsg("");
    try {
      const res = await checkQuizAnswer(quiz.id, currentAnswer, isFinalAttempt);
      setAttempts(nextAttempts);
      if (res.correct) {
        setRevealedAnswer(res.expected_answer);
        setLastResult("correct");
        setOutcome("correct");
        setPhase("revealed");
      } else if (isFinalAttempt) {
        // 2회 연속 오답 → 정답 공개 + "다음으로" 대기
        setRevealedAnswer(res.expected_answer);
        setLastResult("wrong");
        setOutcome("wrong-final");
        setPhase("revealed");
      } else {
        // 1회차 오답 → 한 번 더 시도 (입력 초기화)
        setLastResult("wrong");
        setTypedAnswer("");
        setSignLabels([]);
        lastAppendedRef.current = "";
        setPhase("answering");
      }
    } catch (e) {
      setErrorMsg((e as Error).message);
      setPhase("answering");
    }
  };

  const handleAcknowledge = () => {
    if (outcome === "correct") {
      onComplete({ correct: true, attempts });
    } else if (outcome === "wrong-final") {
      onComplete({ correct: false, attempts });
    } else if (outcome === "revealed-only") {
      onComplete({ correct: null, attempts: 0 });
    }
  };

  return (
    <div className="quiz-panel">
      <div className="quiz-panel-header">
        {onCancel && phase !== "revealed" && phase !== "submitting" && (
          <button
            className="btn-quiz-back"
            onClick={onCancel}
            title="퀴즈를 종료하고 섹션 화면으로 돌아가요"
          >
            ← 뒤로가기
          </button>
        )}
        <span className="quiz-panel-icon">🧠</span>
        <div className="quiz-panel-titles">
          <span className="quiz-panel-title">퀴즈</span>
          <span className="quiz-panel-sub">정답을 맞히면 다음 섹션으로 넘어갈 수 있어요</span>
        </div>
      </div>

      <div className="quiz-question-box">
        <span className="quiz-question-label">Q.</span>
        <p className="quiz-question">{quiz.question}</p>
      </div>

      {phase === "choosing" && (
        <div className="quiz-choice-grid">
          <button className="quiz-choice quiz-choice-primary" onClick={handleSolve}>
            <span className="quiz-choice-emoji">✍️</span>
            <span className="quiz-choice-title">풀기</span>
            <span className="quiz-choice-desc">직접 답을 입력해요 (2회 시도)</span>
          </button>
          <button className="quiz-choice quiz-choice-secondary" onClick={handleRevealOnly}>
            <span className="quiz-choice-emoji">👀</span>
            <span className="quiz-choice-title">정답 보기</span>
            <span className="quiz-choice-desc">바로 정답을 확인해요</span>
          </button>
        </div>
      )}

      {(phase === "answering" || phase === "submitting") && (
        <div className="quiz-answer-block">
          {/* 입력 모드 탭 — 질문하기와 동일 UI 패턴 */}
          <div className="cq-mode-tabs" role="tablist">
            <button
              role="tab"
              aria-selected={inputMode === "sign"}
              className={`cq-mode-tab ${inputMode === "sign" ? "active" : ""}`}
              onClick={() => setInputMode("sign")}
              disabled={phase === "submitting"}
            >
              🤟 수어로 답하기
            </button>
            <button
              role="tab"
              aria-selected={inputMode === "keyboard"}
              className={`cq-mode-tab ${inputMode === "keyboard" ? "active" : ""}`}
              onClick={() => setInputMode("keyboard")}
              disabled={phase === "submitting"}
            >
              ⌨️ 키보드로 답하기
            </button>
          </div>

          {/* 본문 — 모드별 입력 영역 */}
          {inputMode === "sign" ? (
            <>
              <WebcamCapture onPrediction={handleSignPrediction} mirrored />
              <div className="cq-input-card">
                <div className="cq-input-label">인식된 답 (수어)</div>
                <div className="cq-tokens">
                  {signLabels.length === 0 ? (
                    <span className="cq-placeholder">손으로 수어를 보여주면 여기에 쌓입니다…</span>
                  ) : (
                    signLabels.map((l, i) => <span key={i} className="cq-token">{l}</span>)
                  )}
                </div>
                {signLabels.length > 0 && (
                  <button className="btn-cq-clear" onClick={handleClear} disabled={phase === "submitting"}>
                    🗑 지우기
                  </button>
                )}
              </div>
            </>
          ) : (
            <div className="cq-input-card">
              <label className="cq-input-label" htmlFor="quiz-typed">키보드로 답 쓰기</label>
              <input
                id="quiz-typed"
                className="quiz-input"
                value={typedAnswer}
                onChange={(e) => setTypedAnswer(e.target.value)}
                placeholder="답을 입력하세요"
                disabled={phase === "submitting"}
                onKeyDown={(e) => { if (e.key === "Enter") handleSubmit(); }}
                autoFocus
              />
              {typedAnswer.length > 0 && (
                <button className="btn-cq-clear" onClick={handleClear} disabled={phase === "submitting"}>
                  🗑 지우기
                </button>
              )}
            </div>
          )}

          {/* 제출 액션 + 메타 */}
          <div className="cq-send-row">
            <div className="cq-preview">
              <span className="cq-preview-label">제출할 답:</span>
              <span className="cq-preview-text">
                {currentAnswer || <em className="cq-preview-empty">아직 비어있어요</em>}
              </span>
            </div>
            <button
              className="btn-cq-send"
              onClick={handleSubmit}
              disabled={!currentAnswer || phase === "submitting"}
            >
              {phase === "submitting" ? "확인 중…" : "✓ 제출"}
            </button>
          </div>

          <div className="quiz-attempts">시도 {attempts}/{MAX_RETRIES}</div>
          {lastResult === "wrong" && attempts < MAX_RETRIES && (
            <div className="feedback hint">
              😅 아쉽지만 틀렸어요. 한 번 더 시도해보세요.
            </div>
          )}
          {errorMsg && <div className="feedback error">⚠ {errorMsg}</div>}
        </div>
      )}

      {phase === "revealed" && (
        <div className={`quiz-result ${outcome === "correct" ? "success" : "reveal"}`}>
          {outcome === "correct" && (
            <>
              <div className="quiz-result-badge">✅ 정답!</div>
              <p className="quiz-result-text">정답은 <strong>{revealedAnswer}</strong> 였어요.</p>
            </>
          )}
          {outcome === "wrong-final" && (
            <>
              <div className="quiz-result-badge quiz-result-badge-reveal">
                😢 2회 모두 틀렸어요
              </div>
              <p className="quiz-result-text">
                정답은 <strong>{revealedAnswer ?? "—"}</strong> 였어요. 괜찮아요, 다음 섹션에서 또 도전해봐요!
              </p>
            </>
          )}
          {outcome === "revealed-only" && (
            <>
              <div className="quiz-result-badge quiz-result-badge-reveal">📖 정답 공개</div>
              <p className="quiz-result-text">정답: <strong>{revealedAnswer ?? "—"}</strong></p>
            </>
          )}
          <button className="btn-acknowledge" onClick={handleAcknowledge}>
            다음으로 →
          </button>
        </div>
      )}
    </div>
  );
}
