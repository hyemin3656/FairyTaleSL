/**
 * useRecognitionWS — 수어 인식 WebSocket 훅
 *
 * MediaPipe 랜드마크를 서버로 전송하고 인식 결과를 수신.
 * 연속 프레임 전송 시 쓰로틀링(SEND_INTERVAL_MS)으로 서버 부하 제어.
 */
import { useCallback, useEffect, useRef, useState } from "react";

export interface HandLandmark { x: number; y: number; z: number; }
export interface HandData { landmarks: HandLandmark[]; handedness: string; }

export interface RecognitionResult {
  gloss: string;
  confidence: number;
  is_dummy?: boolean;
  is_random_init?: boolean;
}

export type RecognitionStatus = "connecting" | "idle" | "active" | "error" | "closed";

const SEND_INTERVAL_MS = 150; // ~6fps 전송

export function useRecognitionWS() {
  const wsRef = useRef<WebSocket | null>(null);
  const lastSendRef = useRef<number>(0);

  const [status, setStatus] = useState<RecognitionStatus>("connecting");
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [errorMsg, setErrorMsg] = useState("");

  const WS_URL = `ws://${window.location.host}/ws/recognition`;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setStatus("connecting");

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setStatus("idle");

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data as string);
      if (msg.type === "result") {
        setStatus("active");
        setResult({
          gloss: msg.gloss,
          confidence: msg.confidence,
          is_dummy: msg.is_dummy,
          is_random_init: msg.is_random_init,
        });
      } else if (msg.type === "idle") {
        setStatus("idle");
        setResult(null);
      } else if (msg.type === "error") {
        setStatus("error");
        setErrorMsg(msg.message);
      }
    };

    ws.onerror = () => {
      setStatus("error");
      setErrorMsg("WebSocket 연결 오류");
    };

    ws.onclose = () => setStatus("closed");
  }, [WS_URL]);

  useEffect(() => {
    connect();
    return () => { wsRef.current?.close(); };
  }, [connect]);

  // throttle은 WebcamCapture에서 이미 처리 — 여기서는 즉시 전송
  const sendLandmarks = useCallback((hands: HandData[]) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    ws.send(JSON.stringify({
      type: "landmarks",
      num_hands: hands.length,
      hands: hands.map((h) => ({
        landmarks: h.landmarks,
        handedness: h.handedness,
      })),
    }));
  }, []);

  return { status, result, errorMsg, sendLandmarks, reconnect: connect };
}
