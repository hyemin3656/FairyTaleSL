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
  const [pendingLabel, setPendingLabel] = useState<string | null>(null);
  const [pendingConf, setPendingConf] = useState<number>(0);
  const [typedText, setTypedText] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [answer, setAnswer] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string>("");

  // 방금 추가한 단어 — 사용자가 다른 수어를 보일 때까지 같은 예측은 무시한다.
  // 예: "직녀" 추가 후, 사이드카가 잔여 동작에서 또 "직녀"를 보내도 후보에 안 띄움.
  //     "누구" 등 다른 단어가 인식되어야 다시 어떤 예측이든 받기 시작.
  const muteLabelRef = useRef<string | null>(null);

  // 인식된 결과를 "후보(pending)"로만 보여주고, 사용자가 ✅ 추가하기를 눌러야 누적됨.
  const handlePrediction = useCallback((pred: TsnPrediction) => {
    if (phase === "submitting") return;
    if (inputMode !== "sign") return;
    if (pred.is_dummy) return;
    if (pred.confidence < RECOGNITION_THRESHOLD) return;
    if (!pred.gloss) return;
    // 방금 추가한 단어가 잔여 인식으로 다시 들어오면 무시
    if (muteLabelRef.current && pred.gloss === muteLabelRef.current) return;
    // 다른 단어가 들어왔으면 mute 해제 → 새 단어 인식 시작
    if (muteLabelRef.current && pred.gloss !== muteLabelRef.current) {
      muteLabelRef.current = null;
    }
    setPendingLabel(pred.gloss);
    setPendingConf(pred.confidence);
  }, [phase, inputMode]);

  const handleConfirmPending = () => {
    if (!pendingLabel) return;
    const added = pendingLabel;
    setLabels((arr) => [...arr, added]);
    setPendingLabel(null);
    setPendingConf(0);
    // 다음 인식 사이클을 위해 같은 단어 잠시 차단
    muteLabelRef.current = added;
  };

  const handleRejectPending = () => {
    setPendingLabel(null);
    setPendingConf(0);
    // 거절한 단어도 잠시 차단 — 사용자가 다시 시도하면 다른 자세로 인식되어야 의미 있음
    muteLabelRef.current = pendingLabel;
  };

  const handleClear = () => {
    if (inputMode === "sign") {
      setLabels([]);
      setPendingLabel(null);
      setPendingConf(0);
      muteLabelRef.current = null;
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
            <span className="cq-header-name">질문하기</span>
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

          {/* 후보(pending) — 인식된 수어가 의도한 단어가 맞으면 ✅ 추가하기 */}
          <div className="cq-pending-card">
            <div className="cq-pending-label">현재 인식된 수어</div>
            {pendingLabel ? (
              <>
                <div className="cq-pending-row">
                  <span className="cq-pending-word">{pendingLabel}</span>
                  <span className="cq-pending-conf">정확도 {Math.round(pendingConf * 100)}%</span>
                </div>
                <div className="cq-pending-actions">
                  <button className="btn-pending-add" onClick={handleConfirmPending}>
                    ✅ 이 단어 추가
                  </button>
                  <button className="btn-pending-skip" onClick={handleRejectPending}>
                    ✗ 다시 하기
                  </button>
                </div>
                <p className="cq-pending-hint">
                  맞으면 "추가", 아니면 "다시 하기"를 눌러 새로 수어를 보여줘요.
                </p>
              </>
            ) : (
              <p className="cq-placeholder">손으로 수어를 보여주면 여기에 단어가 나타나요.</p>
            )}
          </div>

          {/* 누적된 단어들 (= 만들고 있는 문장) */}
          <div className="cq-input-card">
            <div className="cq-input-label">만들어진 문장</div>
            <div className="cq-tokens">
              {labels.length === 0 ? (
                <span className="cq-placeholder">아직 추가된 단어가 없어요.</span>
              ) : (
                labels.map((l, i) => <span key={i} className="cq-token">{l}</span>)
              )}
            </div>
            {labels.length > 0 && (
              <button className="btn-cq-clear" onClick={handleClear}>🗑 전부 지우기</button>
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
