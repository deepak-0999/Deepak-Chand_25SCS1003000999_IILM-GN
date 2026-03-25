"""
Configuration — loaded from environment or .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    anthropic_api_key: str = Field(..., description="Your Anthropic API key")
    model: str = Field(default="claude-sonnet-4-20250514", description="Claude model to use")
    max_sessions: int = Field(default=500, description="Max concurrent in-memory sessions")
    session_ttl_minutes: int = Field(default=60, description="Session expiry in minutes")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5500"],
        description="Allowed CORS origins",
    )
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
