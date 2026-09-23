import os
import logging
from fastapi import FastAPI, HTTPException, status, Depends, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.schemas import ChatRequest, ChatResponse, ConversationCreateResponse, HealthResponse
from app.services import OpenRouterService, get_openrouter_service

logger = logging.getLogger("api.main")

# Inisialisasi Rate Limiter (slowapi)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.app_name,
    description="Backend API untuk Chatbot AI berbasis FastAPI dan OpenRouter (Gemini 3.8 Flash)",
    version="2.2.0-openrouter",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Registrasi exception handler untuk Rate Limit
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Proteksi API Key internal
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if settings.internal_api_key and settings.internal_api_key.strip():
        if not api_key or api_key != settings.internal_api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

# Keamanan CORS
allowed_origins = settings.allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True if "*" not in allowed_origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
        version="2.2.0-openrouter"
    )

@app.get("/demo", include_in_schema=False)
@app.get("/chat", include_in_schema=False)
async def serve_demo_page():
    """
    Endpoint utilitas untuk melayani halaman testing frontend chatbot satu halaman (index.html).
    Dapat diakses melalui http://127.0.0.1:8000/demo atau http://127.0.0.1:8000/chat
    """
    demo_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "index.html")
    if os.path.exists(demo_file):
        return FileResponse(demo_file)
    return {"message": "Frontend demo file not found."}

@app.post("/api/v1/conversations", response_model=ConversationCreateResponse, status_code=status.HTTP_201_CREATED, tags=["Conversations"], dependencies=[Depends(verify_api_key)])
@limiter.limit("15/minute")
async def create_new_conversation(request: Request, service: OpenRouterService = Depends(get_openrouter_service)):
    """
    Endpoint untuk membuat sesi conversation baru.
    """
    try:
        conv_id = await service.create_conversation()
        return ConversationCreateResponse(
            conversation_id=conv_id,
            message="Sesi percakapan berhasil dibuat"
        )
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Terjadi kesalahan internal: {str(e)}")

@app.post("/api/v1/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK, tags=["Chat"], dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def chat_endpoint(
    request: Request,
    payload: ChatRequest, 
    service: OpenRouterService = Depends(get_openrouter_service)
):
    """
    Endpoint utama berinteraksi dengan AI menggunakan OpenRouter API.
    """
    try:
        output_text, response_id, conv_id = await service.chat_with_ai(
            message=payload.message,
            conversation_id=payload.conversation_id,
            previous_response_id=payload.previous_response_id
        )
        return ChatResponse(
            response=output_text,
            response_id=response_id,
            conversation_id=conv_id
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unhandled error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Kesalahan pada layanan OpenRouter AI: {str(e)}")

