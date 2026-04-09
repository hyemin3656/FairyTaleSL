/**
 * AvatarPlayer — 글로스 모션 클립 시퀀스 플레이어 (더미 UI)
 *
 * 실제 3D 렌더링 없이, 현재 재생 중인 글로스를 시각적으로 표시.
 * Three.js/R3F 통합은 Step 3에서 진행.
 */
import { useEffect, useRef, useState } from "react";
import type { MotionClip } from "../../api/gloss";

interface AvatarPlayerProps {
  clips: MotionClip[];
  playing: boolean;
  onComplete?: () => void;
}

export default function AvatarPlayer({
  clips,
  playing,
  onComplete,
}: AvatarPlayerProps) {
  const [currentIdx, setCurrentIdx] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // playing 시작 시 클립 순서대로 재생
  useEffect(() => {
    if (!playing || clips.length === 0) return;
    setCurrentIdx(0);
  }, [playing, clips]);

  useEffect(() => {
    if (!playing || clips.length === 0) return;

    const clip = clips[currentIdx];
    if (!clip) {
      onComplete?.();
      return;
    }

    timerRef.current = setTimeout(() => {
      if (currentIdx < clips.length - 1) {
        setCurrentIdx((i) => i + 1);
      } else {
        onComplete?.();
      }
    }, clip.duration_sec * 1000);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [playing, currentIdx, clips, onComplete]);

  const current = clips[currentIdx];
  const progress =
    clips.length > 0 ? ((currentIdx + 1) / clips.length) * 100 : 0;

  return (
    <div className="avatar-player">
      {/* 아바타 플레이스홀더 */}
      <div className="avatar-viewport">
        <div className="avatar-dummy">
          <div className="avatar-figure">🧏</div>
          {playing && current ? (
            <div className="avatar-caption">
              <span
                className={`gloss-badge ${current.is_fallback ? "fallback" : "matched"}`}
              >
                {current.gloss}
              </span>
              <span className="emotion-label">{current.emotion_label}</span>
            </div>
          ) : (
            <div className="avatar-idle">수어 아바타 (더미)</div>
          )}
        </div>
      </div>

      {/* 진행 바 */}
      {playing && clips.length > 0 && (
        <div className="clip-progress">
          <div
            className="clip-progress-bar"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* 글로스 토큰 목록 */}
      {clips.length > 0 && (
        <div className="gloss-tokens">
          {clips.map((clip, i) => (
            <span
              key={i}
              className={`token ${i === currentIdx && playing ? "active" : ""} ${clip.is_fallback ? "fallback" : "matched"}`}
            >
              {clip.gloss}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
