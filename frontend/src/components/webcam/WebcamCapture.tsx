/**
 * WebcamCapture — 웹캠 + TSN(MMAction2) 프레임 기반 인식
 *
 * 30fps 캡처 → 3초 sliding window → 1초마다 25프레임 균등 샘플링하여
 * AI Engine /predict_frames 호출. 결과는 onPrediction 콜백으로 부모에 전달.
 */
import { useCallback, useEffect, useRef, useState } from "react";

export interface TsnPrediction {
  class_id: number;
  label: string;
  score: number;
}

interface WebcamCaptureProps {
  onPrediction?: (preds: TsnPrediction[]) => void;
  mirrored?: boolean;
  endpoint?: string;
}

const FPS = 30;
const CAPTURE_INTERVAL_MS = 1000 / FPS;
const WINDOW_MS = 3000;
const NUM_CLIPS = 25;
const PREDICT_INTERVAL_MS = 1000;
const DEFAULT_ENDPOINT = "http://localhost:8001/predict_frames";

export default function WebcamCapture({
  onPrediction,
  mirrored = true,
  endpoint = DEFAULT_ENDPOINT,
}: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameBufferRef = useRef<{ time: number; frame: string }[]>([]);
  const captureTimerRef = useRef<number | null>(null);
  const predictTimerRef = useRef<number | null>(null);
  const isPredictingRef = useRef(false);
  const onPredictionRef = useRef(onPrediction);

  const [running, setRunning] = useState(false);
  const [predictions, setPredictions] = useState<TsnPrediction[]>([]);
  const [errorMsg, setErrorMsg] = useState<string>("");

  // 콜백 ref 동기화 (closure 문제 회피)
  useEffect(() => { onPredictionRef.current = onPrediction; }, [onPrediction]);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;
    if (!canvasRef.current) canvasRef.current = document.createElement("canvas");
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const frame = canvas.toDataURL("image/jpeg", 0.8);
    const now = Date.now();
    frameBufferRef.current.push({ time: now, frame });
    frameBufferRef.current = frameBufferRef.current.filter(
      (f) => now - f.time <= WINDOW_MS
    );
  }, []);

  const uniformSample = (
    buffer: { time: number; frame: string }[],
    n: number
  ): string[] => {
    const total = buffer.length;
    if (total <= n) return buffer.map((f) => f.frame);
    const out: string[] = [];
    for (let i = 0; i < n; i++) {
      out.push(buffer[Math.floor((i * total) / n)].frame);
    }
    return out;
  };

  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { frameRate: FPS, width: 640, height: 480 },
        audio: false,
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setRunning(true);
      setErrorMsg("");

      captureTimerRef.current = window.setInterval(captureFrame, CAPTURE_INTERVAL_MS);
      predictTimerRef.current = window.setInterval(async () => {
        if (isPredictingRef.current) return;
        const buffer = frameBufferRef.current;
        if (buffer.length < NUM_CLIPS) return;
        const frames = uniformSample(buffer, NUM_CLIPS);
        try {
          isPredictingRef.current = true;
          const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frames, topk: 67 }),
          });
          if (!res.ok) {
            setErrorMsg(`서버 오류 ${res.status}`);
            return;
          }
          const data: { predictions: TsnPrediction[] } = await res.json();
          setPredictions(data.predictions);
          onPredictionRef.current?.(data.predictions);
        } catch (e) {
          setErrorMsg(`요청 실패: ${(e as Error).message}`);
        } finally {
          isPredictingRef.current = false;
        }
      }, PREDICT_INTERVAL_MS);
    } catch (e) {
      setErrorMsg(`카메라 접근 실패: ${(e as Error).message}`);
    }
  }, [captureFrame, endpoint]);

  const stopWebcam = useCallback(() => {
    if (captureTimerRef.current !== null) clearInterval(captureTimerRef.current);
    if (predictTimerRef.current !== null) clearInterval(predictTimerRef.current);
    captureTimerRef.current = null;
    predictTimerRef.current = null;
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    frameBufferRef.current = [];
    setRunning(false);
    setPredictions([]);
  }, []);

  // 언마운트 정리
  useEffect(() => stopWebcam, [stopWebcam]);

  return (
    <div className="webcam-wrap" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <video
        ref={videoRef}
        className="webcam-video"
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
      <div style={{ display: "flex", gap: 8 }}>
        {!running ? (
          <button onClick={startWebcam} style={btnStyle}>
            카메라 시작
          </button>
        ) : (
          <button onClick={stopWebcam} style={{ ...btnStyle, background: "#ef4444" }}>
            카메라 정지
          </button>
        )}
      </div>
      {errorMsg && <div style={errorStyle}>{errorMsg}</div>}
      {predictions.length > 0 && (
        <div style={predBoxStyle}>
          <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>인식 결과</div>
          {predictions.map((p) => (
            <div key={p.class_id} style={predRowStyle}>
              <span>{p.label}</span>
              <span style={{ color: "#4f46e5", fontWeight: 600 }}>
                {(p.score * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

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
const predRowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  fontSize: 14,
  padding: "4px 0",
};
