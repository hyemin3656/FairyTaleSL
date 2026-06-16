# FairyTaleSL

한국 전래동화·세계명작을 **한국수어(KSL)** 로 배우는 웹 서비스입니다.  
3D VRM 아바타가 동화 본문을 수어로 재생하고, 아이가 직접 따라하거나 Gemini에게 질문하고, 퀴즈로 학습을 마무리합니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 수어 재생 | 동화 섹션 텍스트 → 글로스 변환 → 3D VRM 아바타가 수어 동작 순서대로 재생 |
| 감정 표정 | 문장 단위 감정 분류 결과를 아바타 표정·눈 깜빡임으로 실시간 반영 (문장 내 모든 글로스에 동일 감정 적용) |
| 일시정지 / 재개 | 재생 중 일시정지 시 아바타 동작 freeze, WebSocket 위치 보존 |
| 자막 ON/OFF | 글로스 토큰 바 표시 여부를 헤더에서 토글 |
| 아바타 선택 | 3종 VRM 아바타(성준·동순·혜미) 중 선택, 모델별 스케일 독립 적용 |
| 질문하기 | 수어 또는 키보드로 질문 입력 → Gemini 2.5 Flash 답변 → 아바타가 수어로 답변 재생 |
| 따라해보기 | 아바타가 보여준 수어 단어를 웹캠으로 직접 따라하면 ST-GCN이 인식·피드백 |
| 퀴즈 | 섹션별 OX·객관식 퀴즈로 이해도 확인 |
| 섹션 삽화 | 각 섹션 장면에 맞는 삽화 자동 전환 |
| 학습 이력 | 세션별 따라하기 통과 여부, 퀴즈 정답률 저장 |

---

## 서비스 구성

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Nginx | 80 | 리버스 프록시 (진입점) |
| Frontend | 5173 | Vite + React + React Three Fiber (3D 아바타) |
| Backend | 8000 | FastAPI + WebSocket (글로스 스트리밍, 세션 관리) |
| AI Engine | 8001 | ST-GCN 수어 인식 (질문 생성·평가) |
| Local Sign Service | 8002 | CNN1D 수어 인식 사이드카 (카메라 점유 + 실시간 추론) |
| PostgreSQL | — | 내부 전용 |

---

## 수어 생성 파이프라인

```
fairy_tales_structured.json (사전 큐레이팅된 ksl_glosses)
  → BookSection.sign_text 에 글로스 시퀀스 저장
  → 재생 시: sign_text → WebSocket /ws/gloss
  → 글로스별 MotionClip 스트리밍 (키포인트 + 감정 레이블)
  → React Three Fiber: VRM 아바타 bone 직접 구동 (15fps)
  → 미등록 글로스: 유의어 대체(_SYNONYM_MAP) → idle 유지 (fallback 동작 없음)
```

> **질문하기(QA)**: Gemini 답변 → RunYourAI KSL 재작성(국립국어원 어휘 제한) → 글로스 스트리밍

| 항목 | 수치 |
|------|------|
| 동화 편수 | **20편** (전래동화 13편 + 세계명작 5편 + 창작 2편) |
| 글로스 수 | **6,418개** (국립국어원 KSL 데이터셋) |
| 수어 영상 | **6,282개** (AI Hub KSL 데이터셋) |
| 키포인트 차원 | 225차원 (MediaPipe Holistic: 손 42점 + 포즈 33점) |
| 동화 커버리지 | **100%** (직접 매칭 + 유의어 대체, fallback 동작 제거) |
| 평균 재생 시간 | 글로스당 **3.76초** (평균 56프레임 @ 15fps) |

---

## 감정 분류 파이프라인

```
동화 문장
  → Claude (claude-haiku-4-5) 배치 분류 — 5클래스
  → fairy_tales_structured.json에 문장별 emotion 저장
  → 수어 재생 시 문장 텍스트로 감정 조회
  → 해당 문장의 모든 글로스 클립에 동일 감정 적용
  → VRM ExpressionManager로 아바타 표정 실시간 전환
```

| 감정 | 비율 | 기준 |
|------|------|------|
| 기쁨 | 37.8% | 행복, 화해, 승리, 긍정적 결말 |
| 중립 | 26.1% | 배경 서술, 평범한 행동 묘사 |
| 놀람 | 20.0% | 예상 밖 사건, 충격, 긴장 |
| 슬픔 | 9.5% | 이별, 상실, 절망 |
| 분노 | 6.7% | 위협, 갈등, 억울함 |

> **문장 단위 분류**: "행복하지 않다" 같은 부정 표현도 문장 전체 맥락으로 판단하여 올바른 감정 적용

---

## 수어 인식 파이프라인 (따라해보기)

```
Local Sign Service(:8002) — 사이드카 프로세스
  → 카메라 단독 점유 (브라우저 getUserMedia 충돌 회피)
  → MediaPipe Holistic: pose 33점 + 손 21점×2
  → 손 검출 비율 기반 세그먼트 자동 분리
  → CNN1D 실시간 추론 (hyemin 모델, 로컬 CPU)
  → WebSocket /ws/sign 으로 브라우저에 push
      preview  : JPEG 미리보기 (~10fps)
      prediction: {gloss, confidence}
```

---

## 동화 목록 (20편)

**전래동화 (13편)**  
토끼와 거북이, 흥부와 놀부, 콩쥐팥쥐, 금도끼 은도끼, 심청전, 단군신화,  
선녀와 나무꾼, 해님달님, 팥죽할머니와 호랑이, 견우와 직녀, 장화홍련, 혹부리 영감, 별을 찾아서

**세계명작 (7편)**  
개미와 베짱이, 빨간 모자, 신데렐라, 아기 돼지 삼형제, 인어공주, 벌거벗은 임금님, 사계절 친구들

---

## 실행 방법

### Docker (권장)

```bash
git clone https://github.com/hyemin3656/FairyTaleSL.git
cd FairyTaleSL

cp .env.example .env
# .env: SECRET_KEY, LLM_API_KEY (RunYourAI) 값 입력

docker compose up --build
```

초기 데이터 시딩 (최초 1회):
```bash
docker compose exec backend alembic upgrade head
docker compose exec backend python scripts/seed.py
```

접속: `http://localhost`

### 로컬 개발 (Docker 없이) — 4개 터미널

로컬에서 코드 변경하며 개발할 때는 4개 서비스를 각각 별도 터미널에서 실행합니다.  
모든 명령은 프로젝트 루트(`PSYcho/`)에서 시작합니다.

**Terminal 1 — Backend (:8000)**
```bash
lsof -ti :8000 | xargs kill -9       # 기존 프로세스 정리
cd backend && ./.venv/bin/uvicorn main:app --port 8000 --reload
```

**Terminal 2 — AI Engine (:8001)**
```bash
lsof -ti :8001 | xargs kill -9
cd ai_engine && ./.venv/bin/uvicorn main:app --port 8001 --reload
```

**Terminal 3 — Frontend (:5173)**
```bash
lsof -ti :5173 | xargs kill -9
cd frontend && npm run dev
```

**Terminal 4 — Local Sign Service (:8002, 수어 인식 사이드카)**
```bash
lsof -ti :8002 | xargs kill -9
cd local_sign_service && ./venv/bin/uvicorn main:app --port 8002
```

브라우저 접속: `http://localhost:5173`

> **기동 순서 권장**: 사이드카(8002) → AI 엔진(8001) → 백엔드(8000) → 프론트(5173)  
> 모델 로딩이 무거운 사이드카·AI 엔진을 먼저 띄우면 안정적입니다.

#### 사전 준비 — 1회만 실행

```bash
# 1. 각 venv 의존성 설치
cd backend && python -m venv .venv && ./.venv/bin/pip install -r requirements.txt && cd ..
cd ai_engine && python -m venv .venv && ./.venv/bin/pip install -r requirements.txt && cd ..
cd local_sign_service && python3.11 -m venv venv && ./venv/bin/pip install -r requirements.txt && cd ..
cd frontend && npm install && cd ..

# 2. backend 로컬 .env 생성 (Docker 호스트명 → localhost 오버라이드)
cat > backend/.env << 'EOF'
DATABASE_URL=postgresql+asyncpg://psycho:changeme@localhost:5432/psycho_db
AI_ENGINE_URL=http://localhost:8001
EOF

# 3. DB 마이그레이션 + 시드 (로컬 PostgreSQL 필요)
cd backend && ./.venv/bin/alembic upgrade head
./.venv/bin/python scripts/seed.py

# 4. motion_db.sqlite 받기 (Git LFS)
brew install git-lfs && git lfs install
git lfs pull

# 5. 루트 .env 작성 (LLM_API_KEY, GEMINI_API_KEY 등 입력)
cp .env.example .env
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
│   ├── services/gloss_service.py    # kiwipiepy KSL 어순 변환 + 감정 조회 + MotionClip 조립
│   ├── models/book.py               # Book, BookSection (sign_text 포함)
│   ├── alembic/versions/0008_add_section_sign_text.py
│   └── scripts/seed.py              # 20편 동화(JSON 단일 소스, ksl_glosses) + 글로스 모션 시딩
├── frontend/
│   └── src/
│       ├── pages/BookReadPage.tsx             # 학습 시나리오 메인 페이지
│       ├── components/avatar/AvatarScene.tsx  # VRM 아바타 + bone 구동 + 표정/눈깜빡임
│       ├── components/session/
│       │   ├── ChildQuestionPanel.tsx         # 수어/키보드 질문 입력
│       │   ├── FollowAlongPanel.tsx           # 따라해보기
│       │   └── QuizPanel.tsx                 # 퀴즈
│       └── stores/
│           ├── scenarioStore.ts              # Zustand 학습 상태 머신
│           └── avatarStore.ts               # 아바타 선택 + 모델별 스케일
├── ai_engine/
│   ├── models/stgcn.py              # ST-GCN 모델 정의
│   └── routers/predict.py           # POST /predict (수어 추론)
└── data_pipeline/
    └── sign_generation/             # 글로스 추출 → 키포인트 → Motion DB 파이프라인
        ├── step1_extract_gloss.py      # kiwipiepy 형태소 분석 + KSL 어순 변환
        ├── step3_extract_keypoints.py  # MediaPipe 키포인트 추출
        ├── step3c_reextract.py         # generic placeholder 재추출
        ├── step4b_emotion_runyourai.py # Claude 문장 감정 분류
        └── step5_motion_db.py          # SQLite Motion DB 시딩
```

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| Frontend | React, TypeScript, Vite, React Three Fiber, @pixiv/three-vrm, Zustand |
| Backend | FastAPI, SQLAlchemy, asyncpg, WebSocket, kiwipiepy |
| AI | MediaPipe Holistic, ST-GCN, Claude (감정 분류), Gemini 2.5 Flash (QA) |
| DB | PostgreSQL (서비스 데이터), SQLite (Motion DB) |
| Infra | Docker Compose, Nginx |
