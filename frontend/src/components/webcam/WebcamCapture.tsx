/**
 * WebcamCapture — 사이드카 기반 수어 인식 미리보기
 *
 * 변경 (2026-06-05):
 *   - 이전: 브라우저가 카메라 점유 + MediaPipe Hands 처리 → /ws/recognition
 *   - 이후: Python 사이드카(:8002)가 카메라 점유 + MediaPipe Holistic + CNN1D 추론
 *           브라우저는 JPEG 미리보기(<img>)와 예측 결과만 받음
 *
 * 부모 컴포넌트 인터페이스(onPrediction)는 그대로 유지 → FollowAlongPanel,
 * ChildQuestionPanel, QuizPanel은 변경 없음.
 */
import { useEffect, useRef } from "react";
import { useRecognitionWS } from "../../hooks/useRecognitionWS";

export interface RecognitionPrediction {
  gloss: string;
  confidence: number;
  is_dummy?: boolean;
}

interface WebcamCaptureProps {
  onPrediction?: (pred: RecognitionPrediction) => void;
  mirrored?: boolean;   // 사이드카가 이미 미러링하므로 시각적 효과만 (기본 true)
}

export default function WebcamCapture({
  onPrediction,
  mirrored: _mirrored = true,
}: WebcamCaptureProps) {
  const { status, result, previewBlobUrl, errorMsg, reconnect } = useRecognitionWS();
  const onPredRef = useRef(onPrediction);

  // onPrediction 콜백 ref 동기화
  useEffect(() => { onPredRef.current = onPrediction; }, [onPrediction]);

  // 예측 결과를 부모 콜백으로 전달
  useEffect(() => {
    if (result) {
      onPredRef.current?.({
        gloss: result.gloss,
        confidence: result.confidence,
        is_dummy: result.is_dummy,
      });
    }
  }, [result]);

  const confidencePct = result ? Math.round(result.confidence * 100) : 0;
  const barColor =
    confidencePct >= 60 ? "#10b981" : confidencePct >= 35 ? "#f59e0b" : "#ef4444";

  const statusBadge = status === "active" ? "🟢 인식 중"
                    : status === "idle"   ? "⚪ 대기"
                    : status === "connecting" ? "🟡 연결 중…"
                    : status === "error"  ? "🔴 오류"
                    : "⚫ 끊김";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={previewWrapStyle}>
        {previewBlobUrl ? (
          <img src={previewBlobUrl} alt="webcam preview" style={previewImgStyle} />
        ) : (
          <div style={previewPlaceholderStyle}>
            {status === "connecting" || status === "closed"
              ? "수어 인식 서비스에 연결 중…"
              : status === "error"
                ? "수어 인식 서비스에 연결할 수 없어요"
                : "카메라 영상을 기다리는 중…"}
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <span style={{ fontSize: 12, color: "#94a3b8" }}>{statusBadge}</span>
        {(status === "error" || status === "closed") && (
          <button onClick={reconnect} style={btnStyle}>다시 연결</button>
        )}
      </div>

      {errorMsg && status === "error" && (
        <div style={errorStyle}>
          {errorMsg}
          <div style={{ marginTop: 4, fontSize: 12, color: "#64748b" }}>
            터미널에서 <code>cd local_sign_service && ./.venv/bin/uvicorn main:app --port 8002</code> 실행 후 다시 연결을 눌러주세요.
          </div>
        </div>
      )}

      {result && (
        <div style={predBoxStyle}>
          <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>인식 결과 (CNN1D)</div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 18, fontWeight: 700 }}>
            <span>{result.gloss}</span>
            <span style={{ color: barColor }}>{confidencePct}%</span>
          </div>
          <div style={{ marginTop: 6, height: 6, background: "#e2e8f0", borderRadius: 3 }}>
            <div style={{
              width: `${confidencePct}%`,
              height: "100%",
              background: barColor,
              borderRadius: 3,
              transition: "width 0.2s",
            }} />
          </div>
          {result.inference_ms != null && (
            <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 4 }}>
              추론 {result.inference_ms.toFixed(1)}ms
              {result.seg_type === "window" ? " · sequence" : ""}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── 스타일 ───────────────────────────────────────────────────────────────────
const previewWrapStyle: React.CSSProperties = {
  width: "100%",
  maxWidth: 480,
  aspectRatio: "4 / 3",
  background: "#000",
  borderRadius: 12,
  overflow: "hidden",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};
const previewImgStyle: React.CSSProperties = {
  width: "100%",
  height: "100%",
  objectFit: "cover",
  display: "block",
};
const previewPlaceholderStyle: React.CSSProperties = {
  color: "#94a3b8",
  fontSize: 13,
  textAlign: "center",
  padding: 16,
};
const btnStyle: React.CSSProperties = {
  padding: "6px 12px",
  background: "#4f46e5",
  color: "#fff",
  fontSize: 13,
  fontWeight: 600,
  borderRadius: 8,
  border: "none",
  cursor: "pointer",
};
const errorStyle: React.CSSProperties = {
  padding: "8px 12px",
  background: "#fef2f2",
  border: "1px solid #fecaca",
  color: "#b91c1c",
  borderRadius: 8,
  fontSize: 13,
};
const predBoxStyle: React.CSSProperties = {
  padding: 12,
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
};

// 기존 import 호환용 타입 re-export
export type { RecognitionPrediction as TsnPrediction };
