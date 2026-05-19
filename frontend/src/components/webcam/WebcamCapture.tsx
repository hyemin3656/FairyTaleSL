/**
 * WebcamCapture — MediaPipe HandLandmarker 기반 수어 인식 컴포넌트
 *
 * 흐름:
 *   웹캠 → requestAnimationFrame → MediaPipe HandLandmarker
 *   → WebSocket /ws/recognition → 인식 결과 콜백
 *
 * TSN(RGB 프레임) 방식에서 ST-GCN(키포인트) 방식으로 변경.
 * 프레임 버퍼는 백엔드가 누적하므로 프론트엔드는 매 프레임 현재 손 좌표만 전송.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  HandLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";
import { useRecognitionWS } from "../../hooks/useRecognitionWS";

export interface RecognitionPrediction {
  gloss: string;
  confidence: number;
  is_dummy?: boolean;
}

interface WebcamCaptureProps {
  onPrediction?: (pred: RecognitionPrediction) => void;
  mirrored?: boolean;
}

// MediaPipe WASM 경로 (CDN — 패키지 버전과 일치)
const MEDIAPIPE_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// 전송 주기 (~10fps — 추론 부하와 실시간성 균형)
const SEND_INTERVAL_MS = 100;

export default function WebcamCapture({
  onPrediction,
  mirrored = true,
}: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const rafRef = useRef<number | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const lastSendRef = useRef<number>(0);
  const onPredRef = useRef(onPrediction);

  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const { status, result, sendLandmarks } = useRecognitionWS();

  // onPrediction 콜백 ref 동기화
  useEffect(() => { onPredRef.current = onPrediction; }, [onPrediction]);

  // 인식 결과 → 부모 콜백 전달
  useEffect(() => {
    if (result) onPredRef.current?.(result);
  }, [result]);

  // MediaPipe HandLandmarker 초기화 (최초 1회)
  const initLandmarker = useCallback(async () => {
    if (landmarkerRef.current) return;
    setLoading(true);
    try {
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
      landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.4,
        minHandPresenceConfidence: 0.4,
        minTrackingConfidence: 0.4,
      });
    } catch (e) {
      setErrorMsg(`MediaPipe 로드 실패: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  // 프레임 처리 루프
  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !landmarker || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const now = performance.now();
    const results = landmarker.detectForVideo(video, now);

    if (results.landmarks.length > 0 && now - lastSendRef.current > SEND_INTERVAL_MS) {
      lastSendRef.current = now;
      const hands = results.landmarks.map((lm, i) => ({
        landmarks: lm.map((p) => ({ x: p.x, y: p.y, z: p.z })),
        handedness: results.handednesses[i]?.[0]?.categoryName ?? "Unknown",
      }));
      sendLandmarks(hands);
    } else if (results.landmarks.length === 0 && now - lastSendRef.current > 500) {
      // 손이 없으면 0.5초마다 idle 신호
      lastSendRef.current = now;
      sendLandmarks([]);
    }

    rafRef.current = requestAnimationFrame(processFrame);
  }, [sendLandmarks]);

  const startWebcam = useCallback(async () => {
    setErrorMsg("");
    await initLandmarker();
    if (!landmarkerRef.current) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: 30 },
        audio: false,
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setRunning(true);
      rafRef.current = requestAnimationFrame(processFrame);
    } catch (e) {
      setErrorMsg(`카메라 접근 실패: ${(e as Error).message}`);
    }
  }, [initLandmarker, processFrame]);

  const stopWebcam = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setRunning(false);
  }, []);

  useEffect(() => () => stopWebcam(), [stopWebcam]);

  const confidencePct = result ? Math.round(result.confidence * 100) : 0;
  const barColor =
    confidencePct >= 60 ? "#10b981" : confidencePct >= 35 ? "#f59e0b" : "#ef4444";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        style={{
          width: "100%",
          maxWidth: 480,
          background: "#000",
          borderRadius: 12,
          transform: mirrored ? "scaleX(-1)" : "none",
        }}
      />

      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        {!running ? (
          <button onClick={startWebcam} disabled={loading} style={btnStyle}>
            {loading ? "로딩 중…" : "카메라 시작"}
          </button>
        ) : (
          <button onClick={stopWebcam} style={{ ...btnStyle, background: "#ef4444" }}>
            카메라 정지
          </button>
        )}
        <span style={{ fontSize: 12, color: "#94a3b8" }}>
          WS: {status}
        </span>
      </div>

      {errorMsg && (
        <div style={errorStyle}>{errorMsg}</div>
      )}

      {result && (
        <div style={predBoxStyle}>
          <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>인식 결과 (ST-GCN)</div>
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
          {result.is_dummy && (
            <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 4 }}>
              * 가중치 미로드 — 학습 후 weights/stgcn_best.pt 배치 필요
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── 스타일 ───────────────────────────────────────────────────────────────────
const btnStyle: React.CSSProperties = {
  padding: "8px 16px",
  background: "#4f46e5",
  color: "#fff",
  fontSize: 14,
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
