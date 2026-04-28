/**
 * SignPracticePage — 수어 따라하기 연습 페이지
 *
 * URL: /practice?gloss=토끼&bookId=xxx&section=1
 *
 * 흐름:
 *   1. URL 파라미터에서 목표 글로스 수신
 *   2. 웹캠 + MediaPipe로 손 랜드마크 추출
 *   3. WS /ws/recognition 으로 랜드마크 전송
 *   4. 서버 응답과 목표 글로스 비교 → 정확도 피드백
 *   5. 성공 시 책 읽기 페이지로 복귀
 */
import { useCallback, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import WebcamCapture from "../components/webcam/WebcamCapture";
import type { TsnPrediction } from "../components/webcam/WebcamCapture";

const SUCCESS_THRESHOLD = 0.5;  // TSN softmax 점수 임계값

export default function SignPracticePage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const targetGloss = searchParams.get("gloss") ?? "토끼";
  const bookId      = searchParams.get("bookId");
  const section     = searchParams.get("section") ?? "0";

  const [preds, setPreds] = useState<TsnPrediction[]>([]);

  const handlePrediction = useCallback((p: TsnPrediction[]) => {
    setPreds(p);
  }, []);

  // class_labels.json의 중복 라벨("좋다"=21,65 / "부탁"=37,57) 보정 — 같은 라벨 점수 합산
  const labelScores = useMemo(() => {
    const m = new Map<string, number>();
    preds.forEach((p) => m.set(p.label, (m.get(p.label) ?? 0) + p.score));
    return m;
  }, [preds]);

  const topLabel = useMemo<{ label: string; score: number } | null>(() => {
    let bestLabel: string | null = null;
    let bestScore = -Infinity;
    labelScores.forEach((score, label) => {
      if (score > bestScore) {
        bestScore = score;
        bestLabel = label;
      }
    });
    return bestLabel === null ? null : { label: bestLabel, score: bestScore };
  }, [labelScores]);

  const targetScore = labelScores.get(targetGloss) ?? 0;

  const isMatch =
    !!topLabel && topLabel.label === targetGloss && targetScore >= SUCCESS_THRESHOLD;

  const handleBack = () => {
    if (bookId) {
      navigate(`/books/${bookId}?section=${section}`);
    } else {
      navigate("/books");
    }
  };

  const barColor =
    targetScore >= SUCCESS_THRESHOLD ? "#10b981" : targetScore >= 0.3 ? "#f59e0b" : "#ef4444";

  return (
    <div className="practice-page">
      <header className="practice-header">
        <button className="btn-back" onClick={handleBack}>
          ← 돌아가기
        </button>
        <h1 className="practice-title">수어 따라하기</h1>
      </header>

      <div className="practice-body">
        <section className="practice-cam-section">
          <div className="cam-label">📷 내 손</div>
          <WebcamCapture onPrediction={handlePrediction} mirrored />
        </section>

        <section className="practice-result-section">
          <div className="target-card">
            <span className="target-label">목표 수어</span>
            <span className="target-gloss">{targetGloss}</span>
          </div>

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
                <span className="conf-label">
                  정확도 {Math.round(targetScore * 100)}%
                </span>
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
            <div className="feedback hint">
              🤔 비슷해요! 조금 더 정확하게 해보세요.
            </div>
          )}

          <button className="btn-done" onClick={handleBack}>
            완료하고 돌아가기
          </button>
        </section>
      </div>
    </div>
  );
}
