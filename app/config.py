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
    
    # OpenAI Credentials & Config (Responses API)
    openai_api_key: str = Field("sk-your-openai-api-key-here", alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-5.5", alias="OPENAI_MODEL") # Gunakan model terbaru yang mendukung Responses API (gpt-4o, gpt-5.5, dll)
    openai_system_instruction_file: Optional[str] = Field("system_instructions.txt", alias="OPENAI_SYSTEM_INSTRUCTION_FILE")
    openai_system_instructions_raw: Optional[str] = Field(None, alias="OPENAI_SYSTEM_INSTRUCTIONS")

    @property
    def openai_system_instructions(self) -> str:
        file_path = self.openai_system_instruction_file
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except Exception:
                pass
        
        if self.openai_system_instructions_raw and self.openai_system_instructions_raw.strip():
            return self.openai_system_instructions_raw.strip()
            
        return "You are a helpful and polite AI assistant."

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

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
