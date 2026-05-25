from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://psycho:changeme@postgres:5432/psycho_db"

    # JWT
    SECRET_KEY: str = "changeme-32-bytes-secret-key-here"
    ALGORITHM: str = "HS256"
    # 아이용 학습 앱이라 보안 민감도가 낮고, 1시간 만료 시 매번 재로그인이 불편해
    # 30일(43200분)로 확장. 그 사이 사용자가 명시적으로 로그아웃하지 않으면 세션 유지.
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30

    # LLM
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    # AI Engine
    AI_ENGINE_URL: str = "http://ai_engine:8001"

    # SMTP
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""

    # App
    APP_ENV: str = "development"


settings = Settings()
