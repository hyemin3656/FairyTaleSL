"""
AI Engine FastAPI service on port 8001.

Active endpoints:
  GET  /health
  POST /qa/question
  POST /qa/evaluate
  POST /qa/followup
  POST /qa/child
"""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
if _ROOT_ENV.exists():
    load_dotenv(_ROOT_ENV)
load_dotenv()

from routers import qa

app = FastAPI(title="PSYcho AI Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai_engine"}
