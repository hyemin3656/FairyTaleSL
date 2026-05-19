# FairyTaleSL

동화 텍스트를 한국수어(KSL)로 교육하는 웹 서비스입니다.  
ST-GCN(Spatial-Temporal Graph CNN) 기반 수어 인식으로 사용자가 카메라 앞에서 수어를 따라하면 실시간으로 인식·피드백합니다.

---

## 서비스 구성

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Nginx | 80 | 리버스 프록시 (진입점) |
| Frontend | 5173 | Vite + React (MediaPipe HandLandmarker) |
| Backend | 8000 | FastAPI + WebSocket `/ws/recognition` |
| AI Engine | 8001 | ST-GCN 수어 인식 (67 한국어 클래스) |
| PostgreSQL | - | 내부 전용 |

---

## 수어 인식 파이프라인

```
브라우저 웹캠
  → MediaPipe HandLandmarker (손 랜드마크 21점 × 최대 2손)
  → WebSocket /ws/recognition (100ms 주기 전송)
  → Backend 프레임 버퍼 (최대 105프레임 누적)
  → AI Engine POST /predict
  → ST-GCN (67 한국어 클래스 분류)
  → 인식 결과 반환
```

### 인식 대상 단어 (67개)
안녕, 무엇, 고기, 비빔밥, 기쁘다, 취미, 나, 영화, 얼굴, 보다, 공부하다, 다시, 몇, 받다, 버스, 너, 휴대폰, 걷다, 서울 등 67개 한국어 단어

---

## 실행 방법

### 방법 1 — Docker (권장)

#### 사전 준비
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치

```bash
git clone https://github.com/hyemin3656/FairyTaleSL.git
cd FairyTaleSL

# 환경변수 설정
cp .env.example .env
# .env 파일에서 SECRET_KEY, LLM_API_KEY 값 설정

docker compose up --build
```

초기 데이터 시딩 (최초 1회):
```bash
docker compose exec backend python scripts/seed.py
```

접속: `http://localhost`

---

### 방법 2 — 로컬 직접 실행 (개발/GPU 서버)

> AI Engine은 GPU 서버에서 실행하는 경우에 적합합니다.

#### 사전 준비

1. **학습 데이터 준비** (Google Drive에서 다운로드 후 변환)

```bash
pip install gdown
mkdir -p ~/pose_align && cd ~/pose_align

# holistic_results (train/val/test) 다운로드
gdown --folder https://drive.google.com/drive/folders/1UvOW9TJA62yQBDEF2LGvyLRnajNHOGnm -O .

# (T,V,4) 포맷 → ST-GCN (T,225) 포맷 변환
cd /path/to/FairyTaleSL
python ai_engine/pipelines/convert_holistic_to_flat.py \
    --src_root ~/pose_align \
    --out_dir  ai_engine/data/keypoints
```

2. **ST-GCN 학습**

```bash
# PyTorch (CUDA) 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python ai_engine/training/train_stgcn.py \
    --keypoints_dir ai_engine/data/keypoints \
    --split_csv     split_result.csv \
    --label_map     ai_engine/data/keypoints/label_map.txt \
    --class_pickle  class_label.p \
    --out_dir       weights/stgcn_baseline \
    --epochs 100 --batch_size 256 --device cuda
```

3. **가중치 배포**

```bash
mkdir -p ai_engine/weights
cp weights/stgcn_baseline/stgcn_best.pt ai_engine/weights/stgcn_best.pt
```

#### 서비스 실행 (터미널 3개)

**Terminal 1 — AI Engine**
```bash
cd ai_engine
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 — Backend**
```bash
cd backend
AI_ENGINE_URL=http://localhost:8001 uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> `AI_ENGINE_URL` 환경변수 필수 — 기본값이 Docker 전용 `http://ai_engine:8001`

**Terminal 3 — Frontend**
```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

접속: `http://localhost:5173`

---

## 학습 결과 (현재 베이스라인)

| | 샘플 수 | Top-1 정확도 | Top-3 정확도 |
|--|--------|------------|------------|
| Train | 978 | 36.9% | 61.8% |
| Val | 133 | 25.6% | 43.6% |
| Test | 118 | 24.6% | 50.0% |

- 학습 환경: NVIDIA RTX A5000, 100 epochs, batch_size=256
- 클래스당 약 18개 샘플로 학습된 베이스라인 — 데이터 증가 시 정확도 개선 가능

---

## 주요 파일 구조

```
FairyTaleSL/
├── ai_engine/
│   ├── main.py                          # FastAPI (포트 8001)
│   ├── models/stgcn.py                  # ST-GCN 모델 정의
│   ├── routers/predict.py               # POST /predict (수어 추론)
│   ├── training/
│   │   ├── train_stgcn.py               # 학습 스크립트
│   │   └── dataset.py                   # KSLKeypointDataset
│   ├── pipelines/
│   │   └── convert_holistic_to_flat.py  # 데이터 포맷 변환기
│   ├── tsn/class_labels.json            # 67개 한국어 클래스 레이블
│   └── weights/stgcn_best.pt            # 학습된 가중치 (gitignore)
├── backend/
│   └── routers/recognition_ws.py        # WebSocket 프레임 버퍼 누적
├── frontend/
│   └── src/
│       ├── components/webcam/WebcamCapture.tsx  # MediaPipe HandLandmarker
│       ├── hooks/useRecognitionWS.ts            # WebSocket 훅
│       └── pages/SignPracticePage.tsx           # 수어 따라하기 페이지
└── weights/                             # 학습 가중치 (gitignore)
```

---

## gitignore 대상 (커밋 제외)

| 항목 | 이유 |
|------|------|
| `weights/`, `*.pt` | 대용량 바이너리 |
| `ai_engine/weights/` | 대용량 바이너리 |
| `ai_engine/data/` | 대용량 학습 데이터 |
| `.env` | 시크릿 키 포함 |
| `node_modules/` | 패키지 캐시 |

---

## 컨테이너 종료 (Docker 사용 시)

```bash
docker compose down

# DB 포함 전체 초기화
docker compose down -v
```
