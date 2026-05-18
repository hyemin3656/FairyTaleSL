/**
 * ChildQuestionPanel — 아이가 질문하기 (부 모드)
 *
 * 두 가지 입력 모드:
 *   🤟 수어    : 웹캠 인식 → top-1 라벨이 임계값(0.5) 이상이면 누적
 *   ⌨️ 키보드  : 보호자가 텍스트로 직접 입력
 *
 * 버튼 구분:
 *   "✉ 질문 보내기" → 현재 입력된 질문을 Gemini로 전송 (= 이전 "질문 끝")
 *   "✕ 마치고 돌아가기" → 질문 모드 자체를 종료하고 동화로 복귀 (= 이전 "질문 마치기")
 *
 * 흐름:
 *   1) 질문 보내기 → POST /qa/child → Gemini 응답 → onAnswer(text)
 *      → 부모는 useGlossWS.sendText(text)로 글로스 변환 + 아바타 재생
 *   2) 마치고 돌아가기 → onExit() → scenarioStore.exitChildQuestion()
 */
import { useCallback, useRef, useState } from "react";
import WebcamCapture from "../webcam/WebcamCapture";
import type { TsnPrediction } from "../webcam/WebcamCapture";
import { askChildQuestion } from "../../api/qa";

const RECOGNITION_THRESHOLD = 0.5;

interface Props {
  storyContext: string;        // 현재 섹션 본문 (Gemini에 전달)
  onAnswer: (text: string) => void;
  onExit: () => void;
  answerGlossTokens?: string[];   // 답변을 글로스로 변환한 토큰 (아바타가 재생 중인 단어들)
}

type Phase = "idle" | "submitting" | "answered" | "error";
type InputMode = "sign" | "keyboard";

export default function ChildQuestionPanel({ storyContext, onAnswer, onExit, answerGlossTokens = [] }: Props) {
  const [inputMode, setInputMode] = useState<InputMode>("sign");
  const [labels, setLabels] = useState<string[]>([]);
  const [typedText, setTypedText] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [answer, setAnswer] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const lastAppendedRef = useRef<string>("");

  const handlePrediction = useCallback((preds: TsnPrediction[]) => {
    if (phase === "submitting") return;
    if (inputMode !== "sign") return;
    if (preds.length === 0) return;
    const m = new Map<string, number>();
    preds.forEach((p) => m.set(p.label, (m.get(p.label) ?? 0) + p.score));
    let topLabel = "", topScore = -Infinity;
    m.forEach((score, label) => {
      if (score > topScore) { topScore = score; topLabel = label; }
    });
    if (topScore < RECOGNITION_THRESHOLD) return;
    if (topLabel === lastAppendedRef.current) return;
    lastAppendedRef.current = topLabel;
    setLabels((arr) => [...arr, topLabel]);
  }, [phase, inputMode]);

  const handleClear = () => {
    if (inputMode === "sign") {
      setLabels([]);
      lastAppendedRef.current = "";
    } else {
      setTypedText("");
    }
  };

  // 현재 모드에 따라 보낼 질문 문자열 결정
  const currentQuestion = inputMode === "sign"
    ? labels.join(" ")
    : typedText.trim();

  const canSend = currentQuestion.length > 0 && phase !== "submitting";

  const handleSubmit = async () => {
    if (!canSend) return;
    setPhase("submitting");
    setErrorMsg("");
    try {
      const res = await askChildQuestion(currentQuestion, storyContext);
      setAnswer(res.answer);
      setPhase("answered");
      onAnswer(res.answer);
    } catch (e) {
      setErrorMsg((e as Error).message || "응답 실패");
      setPhase("error");
    }
  };

  const handleRetry = () => {
    setPhase("idle");
    setErrorMsg("");
  };

  return (
    <div className="cq-panel">
      {/* 헤더 — 마치고 돌아가기 버튼이 명확히 보이도록 분리 */}
      <div className="cq-header">
        <div className="cq-header-title">
          <span className="cq-header-icon">❓</span>
          <div className="cq-header-text">
            <span className="cq-header-name">아이가 질문하기</span>
            <span className="cq-header-sub">동화에 대해 궁금한 점을 물어봐요</span>
          </div>
        </div>
        <button className="btn-cq-exit" onClick={onExit} title="질문 모드를 닫고 동화로 돌아가요">
          ✕ 마치고 돌아가기
        </button>
      </div>

      {/* 입력 모드 탭 */}
      <div className="cq-mode-tabs" role="tablist">
        <button
          role="tab"
          aria-selected={inputMode === "sign"}
          className={`cq-mode-tab ${inputMode === "sign" ? "active" : ""}`}
          onClick={() => setInputMode("sign")}
        >
          🤟 수어로 질문
        </button>
        <button
          role="tab"
          aria-selected={inputMode === "keyboard"}
          className={`cq-mode-tab ${inputMode === "keyboard" ? "active" : ""}`}
          onClick={() => setInputMode("keyboard")}
        >
          ⌨️ 키보드로 질문
        </button>
      </div>

      {/* 본문 — 모드별 입력 영역 */}
      {inputMode === "sign" ? (
        <>
          <WebcamCapture onPrediction={handlePrediction} mirrored />
          <div className="cq-input-card">
            <div className="cq-input-label">인식된 질문 (수어)</div>
            <div className="cq-tokens">
              {labels.length === 0 ? (
                <span className="cq-placeholder">손으로 수어를 보여주면 여기에 쌓입니다…</span>
              ) : (
                labels.map((l, i) => <span key={i} className="cq-token">{l}</span>)
              )}
            </div>
            {labels.length > 0 && (
              <button className="btn-cq-clear" onClick={handleClear}>🗑 지우기</button>
            )}
          </div>
        </>
      ) : (
        <div className="cq-input-card">
          <label className="cq-input-label" htmlFor="cq-typed">키보드로 질문 쓰기</label>
          <textarea
            id="cq-typed"
            className="cq-textarea"
            value={typedText}
            onChange={(e) => setTypedText(e.target.value)}
            placeholder="예) 호랑이는 왜 떡을 달라고 했어요?"
            rows={3}
            disabled={phase === "submitting"}
          />
          {typedText.length > 0 && (
            <button className="btn-cq-clear" onClick={handleClear}>🗑 지우기</button>
          )}
        </div>
      )}

      {/* 주 액션 — 질문 보내기 (현재 질문이 미리 보임) */}
      <div className="cq-send-row">
        <div className="cq-preview">
          <span className="cq-preview-label">보낼 질문:</span>
          <span className="cq-preview-text">
            {currentQuestion || <em className="cq-preview-empty">아직 비어있어요</em>}
          </span>
        </div>
        <button
          className="btn-cq-send"
          onClick={handleSubmit}
          disabled={!canSend}
        >
          {phase === "submitting" ? "⏳ Gemini에 묻는 중…" : "✉ 질문 보내기"}
        </button>
      </div>

      {/* 응답 — 텍스트 + 수어(글로스 토큰) 둘 다 노출 */}
      {phase === "answered" && (
        <div className="cq-answer">
          <div className="cq-answer-label">📢 답변</div>

          {/* 1) 텍스트 */}
          <div className="cq-answer-section">
            <div className="cq-answer-section-label">📝 한국어 답변</div>
            <p className="cq-answer-text">{answer}</p>
          </div>

          {/* 2) 수어 — 글로스 토큰 + 아바타 재생 안내 */}
          <div className="cq-answer-section">
            <div className="cq-answer-section-label">🤟 수어 답변</div>
            {answerGlossTokens.length > 0 ? (
              <div className="cq-answer-gloss">
                {answerGlossTokens.map((t, i) => (
                  <span key={i} className="cq-answer-gloss-token">{t}</span>
                ))}
              </div>
            ) : (
              <p className="cq-answer-hint">아바타가 수어로 보여주는 중…</p>
            )}
            <p className="cq-answer-hint">왼쪽 아바타가 위 단어들을 순서대로 보여줘요.</p>
          </div>
        </div>
      )}

      {phase === "error" && (
        <div className="cq-error">
          <p className="cq-error-text">⚠ {errorMsg}</p>
          <button className="btn-cq-retry" onClick={handleRetry}>다시 시도</button>
        </div>
      )}
    </div>
  );
}
