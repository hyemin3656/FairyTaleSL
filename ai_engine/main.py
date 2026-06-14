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

from routers import qa

try:
    from routers import predict as _predict_router
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    print("[AI Engine] torch/numpy 없음 — /predict 비활성화, QA만 동작")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _HAS_TORCH:
        try:
            from models.stgcn import get_model, load_weights
            from routers.predict import VOCAB, WEIGHTS_PATH
            get_model(vocab=VOCAB, device="cpu", mode="classify")
            load_weights(WEIGHTS_PATH)
            print(f"[AI Engine] ST-GCN ready ({len(VOCAB)} classes).")
        except Exception as e:
            print(f"[AI Engine] ST-GCN 로드 실패 (무시): {e}")
    yield


app = FastAPI(title="PSYcho AI Engine", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if _HAS_TORCH:
    app.include_router(_predict_router.router)
app.include_router(qa.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai_engine"}
