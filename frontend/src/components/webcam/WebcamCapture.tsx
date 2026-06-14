/**
 * WebcamCapture — 사이드카 기반 수어 인식 미리보기
 *
 * UX:
 *   - 기본 상태: 카메라가 꺼진 placeholder + "🎥 카메라 켜기" 버튼
 *   - 켜기 누르면 → 사이드카(:8002)에 WS 연결 + JPEG 미리보기 + 인식 결과 시작
 *   - "끄기" 누르면 → WS 끊고 placeholder로 복귀
 *
 * 부모 컴포넌트 인터페이스(onPrediction)는 그대로 유지 → FollowAlongPanel,
 * ChildQuestionPanel, QuizPanel은 변경 없음.
 */
import { useEffect, useRef, useState } from "react";
import { useRecognitionWS } from "../../hooks/useRecognitionWS";

export interface RecognitionPrediction {
  gloss: string;
  confidence: number;
  is_dummy?: boolean;
}

interface WebcamCaptureProps {
  onPrediction?: (pred: RecognitionPrediction) => void;
  mirrored?: boolean;   // 사이드카가 이미 미러링하므로 시각적 효과만
  // 인식 모드 (기본: "gloss" — 한 segment당 1단어)
  // "sequence": sliding window 연속 인식 → 손을 들고 있는 동안 여러 단어 연속 출력
  recognitionMode?: "gloss" | "sequence";
}

export default function WebcamCapture(props: WebcamCaptureProps) {
  const [cameraOn, setCameraOn] = useState(false);

  if (!cameraOn) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={previewWrapStyle}>
          <div style={previewPlaceholderStyle}>
            <span style={{ fontSize: 42 }}>🎥</span>
            <span style={{ fontSize: 14, fontWeight: 600 }}>카메라가 꺼져 있어요</span>
            <span style={{ fontSize: 12, color: "#94a3b8" }}>
              수어 인식을 시작하려면 카메라를 켜주세요
            </span>
          </div>
        </div>

        <button onClick={() => setCameraOn(true)} style={primaryBtnStyle}>
          🎥 카메라 켜기
        </button>
      </div>
    );
  }

  return <CameraView {...props} onTurnOff={() => setCameraOn(false)} />;
}

// ── 실제 카메라/인식 뷰 — cameraOn=true일 때만 마운트되어 WS 연결 ───────────
function CameraView({
  onPrediction,
  onTurnOff,
  recognitionMode,
}: WebcamCaptureProps & { onTurnOff: () => void }) {
  const { status, result, previewBlobUrl, errorMsg, reconnect, setMode, mode } = useRecognitionWS();
  const onPredRef = useRef(onPrediction);

  useEffect(() => { onPredRef.current = onPrediction; }, [onPrediction]);

  // 인식 모드 변경 — props.recognitionMode와 사이드카 mode 동기화
  useEffect(() => {
    if (!recognitionMode) return;
    if (status !== "idle" && status !== "active") return;   // WS ready 후에만 전송
    if (mode === recognitionMode) return;
    setMode(recognitionMode);
  }, [recognitionMode, mode, status, setMode]);

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
            {status === "connecting"
              ? "수어 인식 서비스에 연결 중…"
              : status === "error" || status === "closed"
                ? "수어 인식 서비스에 연결할 수 없어요"
                : "카메라 영상을 기다리는 중…"}
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <button onClick={onTurnOff} style={dangerBtnStyle}>
          ⏹ 카메라 끄기
        </button>
        <span style={{ fontSize: 12, color: "#94a3b8" }}>{statusBadge}</span>
        {(status === "error" || status === "closed") && (
          <button onClick={reconnect} style={ghostBtnStyle}>다시 연결</button>
        )}
      </div>

      {errorMsg && status === "error" && (
        <div style={errorStyle}>
          {errorMsg}
          <div style={{ marginTop: 4, fontSize: 12, color: "#64748b" }}>
            터미널에서 <code>cd local_sign_service && ./.venv/bin/uvicorn main:app --port 8002</code> 실행 후 "다시 연결"을 눌러주세요.
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
  background: "#0f172a",
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
  color: "#cbd5e1",
  fontSize: 13,
  textAlign: "center",
  padding: 16,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: 6,
};
const primaryBtnStyle: React.CSSProperties = {
  alignSelf: "flex-start",
  padding: "10px 20px",
  background: "#4f46e5",
  color: "#fff",
  fontSize: 14,
  fontWeight: 700,
  borderRadius: 10,
  border: "none",
  cursor: "pointer",
};
const dangerBtnStyle: React.CSSProperties = {
  padding: "8px 14px",
  background: "#ef4444",
  color: "#fff",
  fontSize: 13,
  fontWeight: 600,
  borderRadius: 8,
  border: "none",
  cursor: "pointer",
};
const ghostBtnStyle: React.CSSProperties = {
  padding: "6px 12px",
  background: "#fff",
  color: "#4f46e5",
  fontSize: 13,
  fontWeight: 600,
  borderRadius: 8,
  border: "1px solid #c7d2fe",
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
