/**
 * WebcamCapture — 웹캠 스트림 + MediaPipe HandLandmarker
 *
 * MediaPipe tasks-vision HandLandmarker로 손 랜드마크를 추출하고
 * onLandmarks 콜백으로 전달. 캔버스에 랜드마크 오버레이 렌더링.
 *
 * 모델/WASM은 CDN에서 로드 (초기 1회, 이후 캐시됨).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  HandLandmarker,
  FilesetResolver,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import type { HandData } from "../../hooks/useRecognitionWS";

interface WebcamCaptureProps {
  onLandmarks: (hands: HandData[]) => void;
  mirrored?: boolean;
}

const WASM_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// 연결선 색상
const CONN_COLOR = "rgba(99,102,241,0.8)";
const POINT_COLOR = "#f59e0b";

// 손 연결 인덱스 (MediaPipe 21개 랜드마크)
const HAND_CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

type InitState = "idle" | "loading" | "ready" | "error";

export default function WebcamCapture({
  onLandmarks,
  mirrored = true,
}: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const rafRef = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);

  const [initState, setInitState] = useState<InitState>("idle");
  const [camError, setCamError] = useState("");

  // ── MediaPipe 초기화 ───────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setInitState("loading");

    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_CDN);
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        if (cancelled) { landmarker.close(); return; }
        landmarkerRef.current = landmarker;
        setInitState("ready");
      } catch {
        if (!cancelled) setInitState("error");
      }
    })();

    return () => { cancelled = true; };
  }, []);

  // ── 카메라 시작 ───────────────────────────────────────────
  useEffect(() => {
    if (initState !== "ready") return;

    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } })
      .then((stream) => {
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      })
      .catch((e) => {
        setCamError(`카메라 접근 실패: ${e.message}`);
      });

    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      cancelAnimationFrame(rafRef.current);
    };
  }, [initState]);

  // ── 프레임 루프 ───────────────────────────────────────────
  const drawLandmarks = useCallback(
    (result: HandLandmarkerResult, w: number, h: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, w, h);

      result.landmarks.forEach((lm) => {
        // 연결선
        ctx.strokeStyle = CONN_COLOR;
        ctx.lineWidth = 2;
        HAND_CONNECTIONS.forEach(([a, b]) => {
          const pA = lm[a], pB = lm[b];
          ctx.beginPath();
          ctx.moveTo(pA.x * w, pA.y * h);
          ctx.lineTo(pB.x * w, pB.y * h);
          ctx.stroke();
        });
        // 점
        ctx.fillStyle = POINT_COLOR;
        lm.forEach((p) => {
          ctx.beginPath();
          ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
          ctx.fill();
        });
      });
    },
    []
  );

  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !landmarker || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const w = video.videoWidth;
    const h = video.videoHeight;

    // 캔버스 크기 동기화
    if (canvasRef.current) {
      canvasRef.current.width = w;
      canvasRef.current.height = h;
    }

    const result = landmarker.detectForVideo(video, performance.now());
    drawLandmarks(result, w, h);

    // 랜드마크 콜백
    if (result.landmarks.length > 0) {
      const hands: HandData[] = result.landmarks.map((lm, i) => ({
        landmarks: lm.map((p) => ({ x: p.x, y: p.y, z: p.z })),
        handedness: result.handedness[i]?.[0]?.categoryName ?? "Unknown",
      }));
      onLandmarks(hands);
    } else {
      onLandmarks([]);
    }

    rafRef.current = requestAnimationFrame(processFrame);
  }, [drawLandmarks, onLandmarks]);

  const handleVideoPlay = useCallback(() => {
    rafRef.current = requestAnimationFrame(processFrame);
  }, [processFrame]);

  // ── 렌더 ─────────────────────────────────────────────────
  if (initState === "idle" || initState === "loading") {
    return (
      <div className="webcam-loading">
        <div className="spinner" />
        <span>MediaPipe 모델 로드 중…</span>
      </div>
    );
  }

  if (initState === "error") {
    return (
      <div className="webcam-error">MediaPipe 초기화 실패. 네트워크를 확인하세요.</div>
    );
  }

  if (camError) {
    return <div className="webcam-error">{camError}</div>;
  }

  return (
    <div
      className="webcam-wrap"
      style={{ transform: mirrored ? "scaleX(-1)" : "none" }}
    >
      <video
        ref={videoRef}
        className="webcam-video"
        playsInline
        muted
        onPlay={handleVideoPlay}
      />
      <canvas ref={canvasRef} className="webcam-canvas" />
    </div>
  );
}
