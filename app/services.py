import logging
import uuid
import asyncio
from typing import Tuple, Optional, Dict, List
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger("api.services")

# In-memory session store untuk menyimpan riwayat percakapan per conversation_id
_conversations_store: Dict[str, List[types.Content]] = {}

class GeminiService:
    def __init__(self, client: Optional[genai.Client] = None):
        self.client = client or genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        self.instructions = settings.system_instructions

    async def create_conversation(self) -> str:
        """
        Membuat sesi percakapan baru untuk Gemini.
        """
        try:
            conv_id = f"conv_{uuid.uuid4().hex[:16]}"
            _conversations_store[conv_id] = []
            logger.info(f"Berhasil membuat conversation baru: {conv_id}")
            return conv_id
        except Exception as e:
            logger.error(f"Error saat membuat conversation Gemini: {str(e)}")
            raise e

    async def chat_with_ai(
        self, 
        message: str, 
        conversation_id: Optional[str] = None,
        previous_response_id: Optional[str] = None
    ) -> Tuple[str, str, Optional[str]]:
        """
        Mengirim pesan menggunakan Google Gemini API.
        Mempertahankan riwayat obrolan berbasis conversation_id.
        """
        try:
            if not conversation_id:
                conversation_id = await self.create_conversation()
            elif conversation_id not in _conversations_store:
                _conversations_store[conversation_id] = []

            history = _conversations_store[conversation_id]

            # Susun daftar pesan (history + pesan baru)
            contents = list(history)
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)]
                )
            )

            config = types.GenerateContentConfig(
                system_instruction=self.instructions,
            )

            # Eksekusi request generate_content ke Gemini
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )
            )

            output_text = response.text or ""
            if not output_text.strip():
                raise RuntimeError("Gemini AI memproses permintaan tetapi tidak mengembalikan teks jawaban.")

            # Perbarui riwayat di memori
            history.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)]
                )
            )
            history.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=output_text.strip())]
                )
            )
            _conversations_store[conversation_id] = history

            response_id = f"resp_{uuid.uuid4().hex[:16]}"
            logger.info(f"Response {response_id} berhasil dibuat untuk conversation {conversation_id}.")

            return output_text.strip(), response_id, conversation_id

        except Exception as e:
            logger.error(f"Error saat komunikasi dengan Gemini API: {str(e)}")
            raise e

# Dependency provider untuk FastAPI
def get_gemini_service() -> GeminiService:
    return GeminiService()

# Backward compatibility alias
OpenAIService = GeminiService
get_openai_service = get_gemini_service

