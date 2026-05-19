"""
AI Engine — FastAPI 서비스 (포트 8001)

엔드포인트:
  GET  /health
  POST /predict   — ST-GCN + CTC 수어 인식
  POST /qa/question  — pko-T5 질문 생성
  POST /qa/evaluate  — pko-T5 답변 평가
"""
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 루트 .env 를 우선 로드 — Gemini API 키 등 (ai_engine/.env 가 있으면 그것이 override)
_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
if _ROOT_ENV.exists():
    load_dotenv(_ROOT_ENV)
load_dotenv()   # ai_engine/.env (있으면)

from routers import predict, qa


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ST-GCN classify 모드 초기화 (67 한국어 클래스)
    from models.stgcn import get_model, load_weights
    from routers.predict import VOCAB, WEIGHTS_PATH
    get_model(vocab=VOCAB, device="cpu", mode="classify")
    load_weights(WEIGHTS_PATH)
    print(f"[AI Engine] ST-GCN ready (classify, {len(VOCAB)} classes).")
    # T5 백그라운드 로드 예약
    from models.t5_qa import preload as t5_preload
    t5_preload()
    yield


app = FastAPI(
    title="PSYcho AI Engine",
    description="ST-GCN + CTC 수어 인식 / pko-T5 Q&A",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(qa.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai_engine"}
