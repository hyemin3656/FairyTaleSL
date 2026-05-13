/**
 * ChildQuestionPanel — 아이가 질문하기 (부 모드)
 *
 * 흐름:
 *   1. 웹캠으로 수어 인식 → top-1 라벨이 임계값(0.5) 이상이면 직전 라벨과 다를 때 누적
 *   2. "질문 끝" 클릭 → 누적된 라벨을 공백으로 join → POST /qa/child
 *   3. Gemini 응답 텍스트를 onAnswer(text)로 부모에 전달
 *      → 부모는 useGlossWS.sendText(text)로 글로스 변환 + 아바타 재생
 *   4. "질문 마치기" 클릭 → onExit()
 *      → 부모는 scenarioStore.exitChildQuestion() + ws.resume() (토큰 인덱스 보존)
 *
 * 누적 정책:
 *   - top-1 점수 ≥ 0.5
 *   - 직전 누적 라벨과 동일하면 무시 (1초마다 같은 라벨 폭주 방지)
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
}

type Phase = "collecting" | "submitting" | "answered" | "error";

export default function ChildQuestionPanel({ storyContext, onAnswer, onExit }: Props) {
  const [labels, setLabels] = useState<string[]>([]);
  const [phase, setPhase] = useState<Phase>("collecting");
  const [answer, setAnswer] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const lastAppendedRef = useRef<string>("");

  const handlePrediction = useCallback((preds: TsnPrediction[]) => {
    if (phase !== "collecting") return;
    if (preds.length === 0) return;
    // labelScores 합산 (중복 라벨 처리)
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
  }, [phase]);

  const handleClear = () => {
    setLabels([]);
    lastAppendedRef.current = "";
  };

  const handleSubmit = async () => {
    if (labels.length === 0) return;
    const question = labels.join(" ");
    setPhase("submitting");
    setErrorMsg("");
    try {
      const res = await askChildQuestion(question, storyContext);
      setAnswer(res.answer);
      setPhase("answered");
      onAnswer(res.answer);
    } catch (e) {
      setErrorMsg((e as Error).message);
      setPhase("error");
    }
  };

  return (
    <div className="child-question-panel">
      <div className="panel-header">
        <span className="panel-icon">❓</span>
        <span className="panel-title">아이가 질문하기</span>
        <button className="btn-exit" onClick={onExit}>질문 마치기</button>
      </div>

      <WebcamCapture onPrediction={handlePrediction} mirrored />

      <div className="accumulated-card">
        <div className="acc-label">인식된 질문 (수어)</div>
        <div className="acc-tokens">
          {labels.length === 0 ? (
            <span className="acc-placeholder">손으로 수어를 보여주면 여기에 쌓입니다…</span>
          ) : (
            labels.map((l, i) => <span key={i} className="acc-token">{l}</span>)
          )}
        </div>
        {labels.length > 0 && (
          <button className="btn-clear" onClick={handleClear}>지우기</button>
        )}
      </div>

      <div className="panel-actions">
        <button
          className="btn-submit"
          onClick={handleSubmit}
          disabled={labels.length === 0 || phase === "submitting"}
        >
          {phase === "submitting" ? "Gemini에 묻는 중…" : "질문 끝"}
        </button>
      </div>

      {phase === "answered" && (
        <div className="answer-card">
          <div className="answer-label">📢 답변</div>
          <p className="answer-text">{answer}</p>
          <p className="answer-hint">아바타가 수어로 보여줍니다…</p>
        </div>
      )}

      {phase === "error" && (
        <div className="answer-card error">
          <p>⚠ {errorMsg || "응답 실패"}</p>
        </div>
      )}
    </div>
  );
}
