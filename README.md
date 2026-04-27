# FairyTaleSL

동화 텍스트를 한국수어(KSL)로 변환해 3D 아바타로 교육하는 웹 서비스입니다.

---

## 실행 방법

### 사전 준비

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치 필요

### 1. 저장소 클론

```bash
git clone https://github.com/hyemin3656/FairyTaleSL.git
cd FairyTaleSL
```

### 2. 환경변수 파일 생성

```bash
cp .env.example .env
```

`.env` 파일을 열어 아래 값을 팀 공유 값으로 교체하세요.

| 항목 | 설명 |
|------|------|
| `SECRET_KEY` | JWT 서명용 32바이트 랜덤 문자열 |
| `LLM_API_KEY` | OpenAI 또는 Gemini API 키 |

### 3. Docker 컨테이너 실행

```bash
docker compose up --build
```

최초 실행 시 이미지 빌드로 수 분 소요됩니다.

### 4. 초기 데이터 시딩 (최초 1회)

컨테이너가 모두 뜬 후 별도 터미널에서:

```bash
docker compose exec backend python scripts/seed.py
```

### 5. 웹 브라우저 접속

```
http://localhost
```

---

## 서비스 구성

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Nginx | 80 | 리버스 프록시 (진입점) |
| Frontend | 5173 | Vite + React |
| Backend | 8000 | FastAPI + WebSocket |
| PostgreSQL | - | 내부 전용 |

---

## 컨테이너 종료

```bash
docker compose down
```

DB 데이터까지 초기화하려면:

```bash
docker compose down -v
```
