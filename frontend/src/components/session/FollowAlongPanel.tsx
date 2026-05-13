/**
 * FollowAlongPanel — 섹션 종료 후 따라하기 (부 모드)
 *
 * - 목표 글로스에 대해 웹캠 인식 → 정확도 게이지
 * - 임계값(0.5) 도달 시 자동 onPass()
 * - 사용자가 직접 스킵하면 onSkip()
 *
 * 기존 SignPracticePage 로직을 컴포넌트화. 페이지(스탠드얼론) 흐름과 시나리오
 * 내장 흐름에서 같은 UI를 재사용한다.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import WebcamCapture from "../webcam/WebcamCapture";
import type { TsnPrediction } from "../webcam/WebcamCapture";

const SUCCESS_THRESHOLD = 0.5;   // config.yaml::recognition.success_threshold 와 동일

interface Props {
  targetGloss: string;
  onPass: () => void;
  onSkip: () => void;
}

export default function FollowAlongPanel({ targetGloss, onPass, onSkip }: Props) {
  const [preds, setPreds] = useState<TsnPrediction[]>([]);
  const passedRef = useRef(false);

  const handlePrediction = useCallback((p: TsnPrediction[]) => {
    setPreds(p);
  }, []);

  // 중복 라벨 합산 (class_labels.json에 같은 라벨이 두 class_id에 매핑된 경우)
  const labelScores = useMemo(() => {
    const m = new Map<string, number>();
    preds.forEach((p) => m.set(p.label, (m.get(p.label) ?? 0) + p.score));
    return m;
  }, [preds]);

  const topLabel = useMemo<{ label: string; score: number } | null>(() => {
    let best: { label: string; score: number } | null = null;
    labelScores.forEach((score, label) => {
      if (!best || score > best.score) best = { label, score };
    });
    return best;
  }, [labelScores]);

  const targetScore = labelScores.get(targetGloss) ?? 0;
  const isMatch =
    !!topLabel && topLabel.label === targetGloss && targetScore >= SUCCESS_THRESHOLD;

  // 통과 시 자동 진행 (한 번만, "✅ 잘 했어요!" 표시 후 0.8초 뒤)
  useEffect(() => {
    if (isMatch && !passedRef.current) {
      passedRef.current = true;
      const t = setTimeout(onPass, 800);
      return () => clearTimeout(t);
    }
  }, [isMatch, onPass]);

  const barColor =
    targetScore >= SUCCESS_THRESHOLD ? "#10b981" : targetScore >= 0.3 ? "#f59e0b" : "#ef4444";

  return (
    <div className="follow-along-panel">
      <div className="panel-header">
        <span className="panel-icon">🤟</span>
        <span className="panel-title">따라해보기</span>
      </div>

      <div className="target-card">
        <span className="target-label">목표 수어</span>
        <span className="target-gloss">{targetGloss}</span>
      </div>

      <WebcamCapture onPrediction={handlePrediction} mirrored />

      <div className="recognition-card">
        <span className="recognition-label">인식된 수어</span>
        {topLabel ? (
          <>
            <span className={`recognition-gloss ${isMatch ? "match" : "no-match"}`}>
              {topLabel.label}
            </span>
            <div className="conf-bar-wrap">
              <div
                className="conf-bar"
                style={{ width: `${targetScore * 100}%`, background: barColor }}
              />
            </div>
            <span className="conf-label">정확도 {Math.round(targetScore * 100)}%</span>
          </>
        ) : (
          <span className="recognition-idle">카메라를 시작하고 손을 보여주세요</span>
        )}
      </div>

      {isMatch && (
        <div className="feedback success">
          ✅ 잘 했어요! <strong>{targetGloss}</strong> 수어가 맞습니다.
        </div>
      )}
      {topLabel && !isMatch && targetScore >= 0.3 && (
        <div className="feedback hint">🤔 비슷해요! 조금 더 정확하게 해보세요.</div>
      )}

      <div className="panel-actions">
        <button className="btn-skip" onClick={onSkip}>건너뛰기</button>
      </div>
    </div>
  );
}
