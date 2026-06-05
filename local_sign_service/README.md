# local_sign_service — 수어 인식 사이드카

FairyTaleSL의 4번째 프로세스. 카메라를 직접 점유해서 **MediaPipe Holistic + CNN1D 추론**을 클라이언트 머신에서 완결합니다. 키포인트는 외부 서버로 나가지 않습니다.

```
[웹캠] ──→ [사이드카 :8002] ──WS(JPEG preview + 예측)──→ [브라우저 :5173]
                │
                └─ MediaPipe Holistic → RealtimeSegmenter → CNN1DRealtimeRecognizer
```

- **포트**: `8002`
- **장치**: CPU (1D-CNN 순수 추론 ~10ms — GPU 불필요)
- **인식 모드**: 기본 `gloss-level` (손 보임 → 사라짐 = 1단어 출력) / 옵션 `sequence-level` (sliding window 연속 출력)
- **모델/추론 코드 출처**: `origin/hyemin` 브랜치 (`webcam_cnn1d_realtime.py`)

---

## 폴더 구조

```
local_sign_service/
├── main.py              # FastAPI 앱 (WebSocket /ws/sign)
├── recognizer.py        # hyemin의 3 클래스 재패키징 (import-friendly)
├── _hyemin_realtime.py  # 원본 코드 사본 (참고용 — main.py는 사용 안 함)
├── requirements.txt     # 사이드카 전용 venv 의존성
├── checkpoints/         # CNN1D 가중치 (gitignored)
│   └── best_acc_top1_epoch_36.pth
├── configs/             # MMAction2 config (필요 시)
├── src/
│   └── class_labels.json
└── README.md
```

---

## 설치 (1회)

### 1. 사이드카 venv

```bash
cd local_sign_service
python3.11 -m venv .venv          # Python 3.10 / 3.11 권장
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. mmaction2 설치

사이드카는 hyemin이 사용한 MMAction2 fork를 그대로 씁니다. 워크스페이스 루트(`PSYcho/`)와 같은 레벨에 `mmaction2/`를 두는 게 가장 간단합니다.

```bash
cd ..   # /Users/.../PSYcho 의 부모로 이동 (또는 PSYcho/ 안에 둬도 됨)
git clone -b experiment/fairytalesl https://github.com/hyemin3656/mmaction2.git
cd mmaction2
pip install -v -e .   # ← 위 사이드카 venv 활성 상태로
```

> `recognizer.py`는 `local_sign_service/../mmaction2/` 경로를 자동으로 `sys.path`에 추가합니다. 다른 위치에 두려면 `PYTHONPATH`로 지정.

### 3. 모델 체크포인트 다운로드

[hyemin 브랜치 README](https://github.com/hyemin3656/FairyTaleSL/tree/hyemin#checkpoint)의 링크에서 `best_acc_top1_epoch_36.pth`를 받아 다음 위치에 배치:

```
local_sign_service/checkpoints/best_acc_top1_epoch_36.pth
```

> 다른 경로에 두려면 환경변수 `SIGN_CHECKPOINT=/path/to/file.pth` 지정.

---

## 실행

### 기본 실행 (CPU, 카메라 0, gloss 모드)

```bash
cd local_sign_service
source .venv/bin/activate
./.venv/bin/uvicorn main:app --port 8002
```

서버가 뜨면 `http://localhost:8002/health`로 확인:

```json
{"status":"ok","service":"local_sign_service","mode":"gloss","running":true,"fps":30}
```

### 옵션 (환경변수)

| ENV | 기본값 | 설명 |
|---|---|---|
| `SIGN_DEVICE` | `cpu` | `cpu` / `cuda:0` |
| `SIGN_CAMERA` | `0` | 카메라 인덱스 |
| `SIGN_FPS` | `30` | 캡처 FPS |
| `SIGN_WIDTH` | `640` | 캡처 너비 |
| `SIGN_HEIGHT` | `480` | 캡처 높이 |
| `SIGN_PREVIEW_FPS` | `10` | 브라우저 미리보기 fps (throttle) |
| `SIGN_JPEG_QUALITY` | `70` | 미리보기 JPEG 품질 (낮을수록 작아짐) |
| `SIGN_MODE` | `gloss` | `gloss` / `sequence` |
| `SIGN_TOP1_THRESHOLD` | `0.5` | top1 score 미만 결과 무시 |
| `SIGN_CONFIG` | `<mmaction2 cnn1d config>` | MMAction2 config 경로 |
| `SIGN_CHECKPOINT` | `./checkpoints/best_acc_top1_epoch_36.pth` | 모델 가중치 |
| `SIGN_LABEL_MAP` | `./src/class_labels.json` | class_id ↔ gloss 매핑 |

예시 (Apple Silicon Mac에서 카메라 1번 + 미리보기 8fps):

```bash
SIGN_CAMERA=1 SIGN_PREVIEW_FPS=8 ./.venv/bin/uvicorn main:app --port 8002
```

---

## 동작 확인

브라우저에서:

1. **백엔드(8000) + AI 엔진(8001) + 프론트(5173) + 사이드카(8002)** 모두 실행
2. 동화책 진입 → 어느 섹션에서든 따라해보기 / 질문하기 / 퀴즈 답안 진입
3. `WebcamCapture` 컴포넌트가 사이드카 미리보기 + 인식 결과를 표시
4. 손을 보였다가 내리면 → 한 단어 인식 → 부모 패널의 `onPrediction(pred)` 콜백 호출

문제 시 사이드카 콘솔 로그 확인:

```
[INFO] camera opened idx=0 640x480@30
[INFO] worker started
[INFO] sidecar ready
```

---

## WebSocket 프로토콜 (`/ws/sign`)

서버 → 클라이언트:

```ts
{ type: "ready",      mode: "gloss", running: true, fps: 30 }
{ type: "state",      mode: "sequence", ... }                          // 모드 변경 후
{ type: "preview",    jpeg_b64: "/9j/4AAQ..." }                        // ~10fps
{ type: "prediction", gloss: "직녀", confidence: 0.87,
                      is_dummy: false, seg_type: "final",
                      inference_ms: 9.2 }
{ type: "idle" }
{ type: "error",      message: "…" }
```

클라이언트 → 서버:

```ts
{ type: "set_mode", mode: "gloss" | "sequence" }
{ type: "ping" }   // → { type: "pong" }
```

---

## 통합 효과 — main의 수어 인식이 다음과 같이 동작합니다

| 영역 | 컴포넌트 | 동작 |
|---|---|---|
| **따라해보기** | [FollowAlongPanel.tsx](../frontend/src/components/session/FollowAlongPanel.tsx) | 사이드카에서 단어 인식 → confidence 0.5 이상이면 자동 통과 |
| **질문하기 (수어 모드)** | [ChildQuestionPanel.tsx](../frontend/src/components/session/ChildQuestionPanel.tsx) | 인식된 단어가 pending 카드에 표시 → 사용자가 "✅ 추가" 누르면 누적 |
| **퀴즈 답안 (수어 모드)** | [QuizPanel.tsx](../frontend/src/components/session/QuizPanel.tsx) | 단어 누적 → 제출 → 백엔드 정규화 매칭 |

`onPrediction(pred)` 콜백 시그니처가 그대로라 **세 패널은 코드 변경이 없습니다**.

---

## 성능 / Latency

| 단계 | 비용 |
|---|---|
| 카메라 캡처 | ~33ms (30fps) |
| MediaPipe Holistic | ~30–50ms (CPU complexity=1) |
| Segmenter 판정 | <1ms |
| CNN1D 추론 (CPU) | **~10ms** (박혜민 측정값) |
| JPEG 인코딩 + WS 전송 | ~3ms (640×480, q=70) |
| **end-to-end** | **손 사라짐 → 화면 표시까지 보통 100ms 이내** |

전체 파이프라인이 모두 localhost에서 동작 — 네트워크 RTT 없음.

---

## 알려진 제약 / TODO

- macOS에서 사이드카가 카메라를 점유하면 다른 앱(예: Zoom, Teams)에서 카메라가 사용 중으로 표시될 수 있습니다.
- 모델 체크포인트는 학습 셋팅에 따라 라벨이 달라질 수 있으니 `src/class_labels.json`이 학습 시점의 라벨과 동기화돼 있는지 확인 필요.
- 첫 호출 시 MediaPipe 모델 로드 + 첫 추론에 ~1초 지연 (워밍업). 사이드카 시작 시 dummy frame 한 번 돌리는 워밍업 추가 가능 (TODO).
