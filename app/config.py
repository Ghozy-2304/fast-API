import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    app_name: str = Field("FastAPI AI Chatbot", alias="APP_NAME")
    app_env: str = Field("development", alias="APP_ENV")
    debug: bool = Field(True, alias="DEBUG")
    port: int = Field(8000, alias="PORT")
    
    allowed_origins_raw: str = Field(
        "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,https://jhic-20-production.up.railway.app",
        alias="ALLOWED_ORIGINS"
    )
    internal_api_key: Optional[str] = Field(None, alias="INTERNAL_API_KEY")
    
    # OpenRouter Credentials & Config (Gemini 3.8 Flash)
    openrouter_api_key: str = Field("", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field("google/gemini-3.8-flash", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_system_instruction_file: str = Field("system_instructions.txt", alias="OPENROUTER_SYSTEM_INSTRUCTION_FILE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def system_instructions(self) -> str:
        file_path = self.openrouter_system_instruction_file
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except Exception:
                pass
        return "You are a helpful and polite AI assistant."

    @property
    def allowed_origins(self) -> List[str]:
        if not self.allowed_origins_raw:
            return ["*"] if self.app_env != "production" else []
        origins = []
        for origin in self.allowed_origins_raw.split(","):
            cleaned = origin.strip()
            if not cleaned:
                continue
            normalized = cleaned.rstrip("/") if cleaned != "*" else "*"
            if normalized not in origins:
                origins.append(normalized)
        return origins

settings = Settings()
