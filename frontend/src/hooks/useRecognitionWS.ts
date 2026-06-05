/**
 * useRecognitionWS — 수어 인식 WebSocket 훅 (사이드카 모드)
 *
 * 구조 변경 (2026-06-05):
 *   - 이전: 브라우저가 MediaPipe로 landmark 추출 → /ws/recognition으로 전송
 *   - 이후: 별도 Python 사이드카(:8002)가 카메라 점유 + MediaPipe + CNN1D 추론까지 수행
 *           브라우저는 /ws/sign으로 JPEG 미리보기 + 예측 결과만 받는다
 *
 * 사이드카 메시지 스키마:
 *   { type: "ready",      mode, running, fps }
 *   { type: "state",      mode, running, fps }
 *   { type: "preview",    jpeg_b64: "..." }
 *   { type: "prediction", gloss, confidence, is_dummy, seg_type, inference_ms }
 *   { type: "idle" }
 *   { type: "error",      message }
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export interface RecognitionResult {
  gloss: string;
  confidence: number;
  is_dummy?: boolean;
  seg_type?: "final" | "window";
  inference_ms?: number;
}

export type RecognitionStatus = "connecting" | "idle" | "active" | "error" | "closed";
export type RecognitionMode = "gloss" | "sequence";

// 사이드카 URL — ENV로 override 가능 (.env: VITE_SIGN_WS_URL=ws://localhost:8002/ws/sign)
const DEFAULT_SIGN_WS = "ws://localhost:8002/ws/sign";

export function useRecognitionWS() {
  const wsRef = useRef<WebSocket | null>(null);

  const [status, setStatus] = useState<RecognitionStatus>("connecting");
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [previewBlobUrl, setPreviewBlobUrl] = useState<string | null>(null);
  const [mode, setMode] = useState<RecognitionMode>("gloss");
  const [errorMsg, setErrorMsg] = useState("");

  const WS_URL = useMemo(() => {
    const env = (import.meta as { env?: Record<string, string> }).env;
    return env?.VITE_SIGN_WS_URL ?? DEFAULT_SIGN_WS;
  }, []);

  // base64 JPEG → Blob URL 변환은 비용이 크므로 직접 data:URL 사용
  const applyPreview = useCallback((b64: string) => {
    setPreviewBlobUrl(`data:image/jpeg;base64,${b64}`);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setStatus("connecting");

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setStatus("idle");

    ws.onmessage = (event) => {
      let msg: { type: string; [k: string]: unknown };
      try {
        msg = JSON.parse(event.data as string);
      } catch {
        return;
      }
      switch (msg.type) {
        case "ready":
        case "state":
          setMode((msg.mode as RecognitionMode) ?? "gloss");
          setStatus("idle");
          break;
        case "preview":
          applyPreview(msg.jpeg_b64 as string);
          break;
        case "prediction":
          setStatus("active");
          setResult({
            gloss: msg.gloss as string,
            confidence: msg.confidence as number,
            is_dummy: msg.is_dummy as boolean | undefined,
            seg_type: msg.seg_type as "final" | "window" | undefined,
            inference_ms: msg.inference_ms as number | undefined,
          });
          break;
        case "idle":
          setStatus("idle");
          break;
        case "error":
          setStatus("error");
          setErrorMsg((msg.message as string) ?? "사이드카 오류");
          break;
      }
    };

    ws.onerror = () => {
      setStatus("error");
      setErrorMsg("사이드카(:8002)에 연결할 수 없어요. 수어 인식 서비스를 실행해주세요.");
    };

    ws.onclose = () => setStatus("closed");
  }, [WS_URL, applyPreview]);

  useEffect(() => {
    connect();
    return () => { wsRef.current?.close(); };
  }, [connect]);

  const setRecognitionMode = useCallback((m: RecognitionMode) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "set_mode", mode: m }));
  }, []);

  return {
    status,
    result,
    previewBlobUrl,
    mode,
    errorMsg,
    setMode: setRecognitionMode,
    reconnect: connect,
  };
}
