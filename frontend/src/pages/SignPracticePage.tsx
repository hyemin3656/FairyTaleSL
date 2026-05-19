/**
 * SignPracticePage — 수어 따라하기 연습 페이지 (ST-GCN 버전)
 *
 * URL: /practice?gloss=안녕&bookId=xxx&section=1
 *
 * 흐름:
 *   1. URL 파라미터에서 목표 글로스 수신
 *   2. WebcamCapture (MediaPipe HandLandmarker) → WS /ws/recognition 전송
 *   3. ST-GCN 인식 결과와 목표 글로스 비교 → 정확도 피드백
 *   4. 성공 시 책 읽기 페이지로 복귀
 */
import { useCallback, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import WebcamCapture from "../components/webcam/WebcamCapture";
import type { RecognitionPrediction } from "../components/webcam/WebcamCapture";

const SUCCESS_THRESHOLD = 0.55;  // ST-GCN softmax 점수 임계값

export default function SignPracticePage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const targetGloss = searchParams.get("gloss") ?? "안녕";
  const bookId      = searchParams.get("bookId");
  const section     = searchParams.get("section") ?? "0";

  const [pred, setPred] = useState<RecognitionPrediction | null>(null);

  const handlePrediction = useCallback((p: RecognitionPrediction) => {
    setPred(p);
  }, []);

  const confidence  = pred?.confidence ?? 0;
  const topGloss    = pred?.gloss ?? null;
  const isMatch     = topGloss === targetGloss && confidence >= SUCCESS_THRESHOLD;

  const handleBack = () => {
    if (bookId) {
      navigate(`/books/${bookId}?section=${section}`);
    } else {
      navigate("/books");
    }
  };

  const barColor =
    confidence >= SUCCESS_THRESHOLD ? "#10b981" : confidence >= 0.3 ? "#f59e0b" : "#ef4444";

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
            {topGloss ? (
              <>
                <span className={`recognition-gloss ${isMatch ? "match" : "no-match"}`}>
                  {topGloss}
                </span>
                <div className="conf-bar-wrap">
                  <div
                    className="conf-bar"
                    style={{ width: `${confidence * 100}%`, background: barColor }}
                  />
                </div>
                <span className="conf-label">
                  정확도 {Math.round(confidence * 100)}%
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
          {topGloss && !isMatch && confidence >= 0.3 && (
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
