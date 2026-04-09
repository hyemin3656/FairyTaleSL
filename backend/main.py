from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import settings
from routers import books, motions, gloss, ws, recognition_ws, qa, auth, sessions

app = FastAPI(
    title="PSYcho API",
    description="수어 기반 동화책 교육 플랫폼 API",
    version="0.1.0",
)

ALLOW_ORIGINS = (
    ["*"]
    if settings.APP_ENV == "development"
    else ["http://localhost:5173", "http://localhost"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 (더미 GLB, 커버 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(books.router, prefix="/api/v1")
app.include_router(motions.router, prefix="/api/v1")
app.include_router(gloss.router, prefix="/api/v1")
app.include_router(ws.router)              # /ws/gloss
app.include_router(recognition_ws.router)  # /ws/recognition
app.include_router(qa.router, prefix="/api/v1")        # /api/v1/qa/*
app.include_router(auth.router, prefix="/api/v1")      # /api/v1/auth/*
app.include_router(sessions.router, prefix="/api/v1")  # /api/v1/sessions/*


@app.get("/health")
async def health():
    return {"status": "ok"}
