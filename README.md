# FairyTaleSL

한국 전래동화를 한국수어(KSL)로 배우는 웹 서비스입니다.  
3D 아바타가 동화 본문을 수어로 재생하고, 아이가 직접 따라하거나 Gemini에게 질문하고, 퀴즈로 학습을 마무리합니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 수어 재생 | 동화 섹션 텍스트를 글로스로 변환 → 3D VRM 아바타가 수어 동작을 순서대로 재생 |
| 일시정지 / 재개 | 재생 중 일시정지 시 아바타 동작 freeze, WebSocket 위치 보존 |
| 질문하기 | 수어 또는 키보드로 질문 입력 → Gemini 2.5 Flash 답변 → 아바타가 수어로 답변 재생 |
| 따라해보기 | 아바타가 보여준 수어 단어를 웹캠으로 직접 따라하면 ST-GCN이 인식·피드백 |
| 퀴즈 | 섹션별 OX·객관식 퀴즈로 이해도 확인 |
| 섹션 삽화 | 각 섹션 상단에 AI 생성 삽화 표시 (Imagen API) |
| 학습 이력 | 세션별 따라하기 통과 여부, 퀴즈 정답률 저장 |

---

## 서비스 구성

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Nginx | 80 | 리버스 프록시 (진입점) |
| Frontend | 5173 | Vite + React + React Three Fiber (3D 아바타) |
| Backend | 8000 | FastAPI + WebSocket (글로스 스트리밍, 세션 관리) |
| AI Engine | 8001 | ST-GCN 수어 인식 (67 한국어 클래스) |
| PostgreSQL | — | 내부 전용 |

---

## 수어 재생 파이프라인

```
동화 텍스트
  → kiwipiepy 형태소 분석 → KSL 어순 변환 → 글로스 시퀀스
  → WebSocket /ws/gloss (글로스별 MotionClip 스트리밍)
  → 프론트엔드: animation_data(사전 베이크 bone rotation) 수신
  → React Three Fiber: VRM 아바타 본 직접 구동 (30fps)
```

- 글로스 수: **6,376개** (kiwipiepy 기반 추출)
- 수어 영상: **6,156개** (AI Hub KSL 데이터셋)
- 키포인트: MediaPipe Holistic 225차원 (손·포즈 통합)
- 애니메이션: step6_bake_anim.py로 사전 베이크된 bone rotation JSON

---

## 수어 인식 파이프라인 (따라해보기)

```
브라우저 웹캠
  → MediaPipe HandLandmarker (손 랜드마크 21점 × 최대 2손)
  → WebSocket /ws/recognition (100ms 주기 전송)
  → Backend 프레임 버퍼 (최대 105프레임 누적)
  → AI Engine POST /predict
  → ST-GCN (67 한국어 클래스 분류)
  → 인식 결과 반환
```

---

## 동화 목록 (13편)

토끼와 거북이, 흥부와 놀부, 콩쥐팥쥐, 금도끼 은도끼, 심청전, 단군신화, 선녀와 나무꾼, 해님달님, 팥죽할머니와 호랑이, 견우와 직녀, 장화홍련, 혹부리 영감, 별을 찾아서

---

## 실행 방법

### 방법 1 — Docker (권장)

```bash
git clone https://github.com/hyemin3656/FairyTaleSL.git
cd FairyTaleSL

# 환경변수 설정
cp .env.example .env
# .env: SECRET_KEY, GEMINI_API_KEY 값 입력

docker compose up --build
```

초기 데이터 시딩 (최초 1회):
```bash
docker compose exec backend alembic upgrade head
docker compose exec backend python scripts/seed.py
```

접속: `http://localhost`

---

### 방법 2 — 로컬 직접 실행 (개발)

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

**Terminal 3 — Frontend**
```bash
cd frontend
npm install
npm run dev
```

접속: `http://localhost:5173`

---

## 섹션 삽화 생성 (Imagen API 결제 후)

```bash
# .env에 GEMINI_API_KEY 설정 후
cd data_pipeline
python generate_section_images.py
# backend/static/images/sections/{tale_id}_{order}.png 저장
```

---

## 주요 파일 구조

```
FairyTaleSL/
├── backend/
│   ├── routers/
│   │   ├── ws.py                    # WebSocket 글로스 스트리밍 (pause/resume)
│   │   ├── books.py                 # 동화 목록/상세 API
│   │   └── qa.py                    # Gemini 질문 응답 API
│   ├── services/gloss_service.py    # kiwipiepy KSL 어순 변환 + MotionClip 조립
│   ├── models/book.py               # Book, BookSection (image_url 포함)
│   ├── scripts/seed.py              # 13권 동화 + 글로스 모션 시딩
│   ├── alembic/versions/            # DB 마이그레이션
│   └── static/images/sections/     # 섹션 삽화 저장 위치
├── frontend/
│   └── src/
│       ├── pages/BookReadPage.tsx          # 학습 시나리오 메인 페이지
│       ├── components/avatar/AvatarScene.tsx  # VRM 아바타 + bone 구동
│       ├── components/session/
│       │   ├── ChildQuestionPanel.tsx      # 수어/키보드 질문 입력
│       │   ├── FollowAlongPanel.tsx        # 따라해보기
│       │   └── QuizPanel.tsx              # 퀴즈
│       └── stores/scenarioStore.ts        # Zustand 학습 상태 머신
├── ai_engine/
│   ├── models/stgcn.py              # ST-GCN 모델 정의
│   └── routers/predict.py           # POST /predict (수어 추론)
└── data_pipeline/
    ├── sign_generation/             # 글로스 추출 → 키포인트 → Motion DB 파이프라인
    └── generate_section_images.py   # Imagen API 섹션 삽화 생성
```

---

## gitignore 대상

| 항목 | 이유 |
|------|------|
| `*.pt`, `weights/` | 대용량 모델 가중치 |
| `ai_engine/data/` | 대용량 학습 데이터 |
| `data_pipeline/raw_data/` | AI Hub 원본 영상 |
| `data_pipeline/sign_generation/data/motion_db.sqlite` | 대용량 Motion DB |
| `.env` | 시크릿 키 포함 |
| `node_modules/` | 패키지 캐시 |

---

## 컨테이너 종료

```bash
docker compose down

# DB 포함 전체 초기화
docker compose down -v
```
