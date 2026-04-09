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
import { useCallback, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useRecognitionWS } from "../hooks/useRecognitionWS";
import WebcamCapture from "../components/webcam/WebcamCapture";
import type { HandData } from "../hooks/useRecognitionWS";

const SUCCESS_THRESHOLD = 0.75;  // 정확도 임계값
const SUCCESS_COUNT_NEEDED = 3;  // 연속 성공 횟수

export default function SignPracticePage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const targetGloss = searchParams.get("gloss") ?? "토끼";
  const bookId      = searchParams.get("bookId");
  const section     = searchParams.get("section") ?? "0";

  const { status, result, errorMsg, sendLandmarks, reconnect } = useRecognitionWS();

  const [successCount, setSuccessCount] = useState(0);
  const [finished, setFinished] = useState(false);

  const handleLandmarks = useCallback(
    (hands: HandData[]) => {
      sendLandmarks(hands);
    },
    [sendLandmarks]
  );

  // 인식 결과 처리
  const isMatch =
    result?.gloss === targetGloss && (result?.confidence ?? 0) >= SUCCESS_THRESHOLD;

  // 연속 성공 카운트
  if (isMatch && successCount < SUCCESS_COUNT_NEEDED && !finished) {
    // React state 업데이트는 렌더 중 불가 → setTimeout 회피용 ref 필요 없이
    // result가 바뀔 때마다 effect로 처리
  }

  const handleBack = () => {
    if (bookId) {
      navigate(`/books/${bookId}?section=${section}`);
    } else {
      navigate("/books");
    }
  };

  // 정확도 바 색상
  const conf = result?.confidence ?? 0;
  const barColor =
    conf >= SUCCESS_THRESHOLD ? "#10b981" : conf >= 0.6 ? "#f59e0b" : "#ef4444";

  return (
    <div className="practice-page">
      {/* 헤더 */}
      <header className="practice-header">
        <button className="btn-back" onClick={handleBack}>
          ← 돌아가기
        </button>
        <h1 className="practice-title">수어 따라하기</h1>
        <span className="ws-status-badge" data-status={status}>
          {status === "connecting" ? "연결 중" :
           status === "active"     ? "인식 중" :
           status === "idle"       ? "대기"    :
           status === "error"      ? "오류"    : "종료"}
        </span>
      </header>

      <div className="practice-body">
        {/* 왼쪽: 웹캠 */}
        <section className="practice-cam-section">
          <div className="cam-label">📷 내 손</div>
          <WebcamCapture onLandmarks={handleLandmarks} mirrored />
        </section>

        {/* 오른쪽: 목표 + 인식 결과 */}
        <section className="practice-result-section">
          {/* 목표 글로스 */}
          <div className="target-card">
            <span className="target-label">목표 수어</span>
            <span className="target-gloss">{targetGloss}</span>
          </div>

          {/* 인식 결과 */}
          <div className="recognition-card">
            <span className="recognition-label">인식된 수어</span>
            {result ? (
              <>
                <span
                  className={`recognition-gloss ${isMatch ? "match" : "no-match"}`}
                >
                  {result.gloss}
                </span>

                {/* 정확도 바 */}
                <div className="conf-bar-wrap">
                  <div
                    className="conf-bar"
                    style={{ width: `${conf * 100}%`, background: barColor }}
                  />
                </div>
                <span className="conf-label">
                  정확도 {Math.round(conf * 100)}%
                  {result.is_dummy && (
                    <span className="dummy-badge"> (더미)</span>
                  )}
                </span>
              </>
            ) : (
              <span className="recognition-idle">
                {status === "idle" ? "손을 카메라 앞에 놓으세요" : "…"}
              </span>
            )}
          </div>

          {/* 피드백 */}
          {isMatch && (
            <div className="feedback success">
              ✅ 잘 했어요! <strong>{targetGloss}</strong> 수어가 맞습니다.
            </div>
          )}
          {result && !isMatch && result.confidence >= 0.5 && (
            <div className="feedback hint">
              🤔 비슷해요! 조금 더 정확하게 해보세요.
            </div>
          )}

          {/* WS 오류 */}
          {status === "error" && (
            <div className="ws-error">
              <span>⚠ {errorMsg}</span>
              <button onClick={reconnect}>재연결</button>
            </div>
          )}

          {/* 완료 버튼 */}
          <button className="btn-done" onClick={handleBack}>
            완료하고 돌아가기
          </button>
        </section>
      </div>
    </div>
  );
}
