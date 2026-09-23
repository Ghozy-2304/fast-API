import logging
import uuid
from typing import Tuple, Optional, Dict, List, AsyncGenerator
from openai import AsyncOpenAI, APIError, AuthenticationError
from app.config import settings

logger = logging.getLogger("api.services")

# In-memory session store untuk menyimpan riwayat percakapan per conversation_id
_conversations_store: Dict[str, List[Dict[str, str]]] = {}

class OpenRouterService:
    def __init__(self, client: Optional[AsyncOpenAI] = None):
        if not settings.openrouter_api_key or not settings.openrouter_api_key.strip():
            raise ValueError("OPENROUTER_API_KEY tidak ditemukan. Harap atur OPENROUTER_API_KEY di environment variables server (Railway).")
        
        self.client = client or AsyncOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://jhic-20-production.up.railway.app",
                "X-Title": "FastAPI AI Chatbot"
            }
        )
        self.model = settings.openrouter_model
        self.instructions = settings.system_instructions

    async def create_conversation(self) -> str:
        """
        Membuat sesi percakapan baru untuk OpenRouter.
        """
        try:
            conv_id = f"conv_{uuid.uuid4().hex[:16]}"
            _conversations_store[conv_id] = []
            logger.info(f"Berhasil membuat conversation baru: {conv_id}")
            return conv_id
        except Exception as e:
            logger.error(f"Error saat membuat conversation OpenRouter: {str(e)}")
            raise e

    async def chat_with_ai(
        self, 
        message: str, 
        conversation_id: Optional[str] = None,
        previous_response_id: Optional[str] = None
    ) -> Tuple[str, str, Optional[str]]:
        """
        Mengirim pesan menggunakan OpenRouter API (Non-Streaming).
        """
        try:
            if not conversation_id:
                conversation_id = await self.create_conversation()
            elif conversation_id not in _conversations_store:
                _conversations_store[conversation_id] = []

            history = _conversations_store[conversation_id]

            messages = [{"role": "system", "content": self.instructions}]
            messages.extend(history)
            messages.append({"role": "user", "content": message})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError("OpenRouter AI tidak mengembalikan teks jawaban.")

            output_text = response.choices[0].message.content.strip()

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": output_text})
            _conversations_store[conversation_id] = history

            response_id = f"resp_{uuid.uuid4().hex[:16]}"
            logger.info(f"Response {response_id} berhasil dibuat untuk conversation {conversation_id}.")

            return output_text, response_id, conversation_id

        except AuthenticationError as e:
            logger.error(f"Autentikasi OpenRouter gagal: {str(e)}")
            raise e
        except APIError as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Error saat komunikasi dengan OpenRouter API: {str(e)}")
            raise e

    async def chat_with_ai_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Mengirim pesan menggunakan OpenRouter API (Streaming kata per kata secara real-time).
        """
        try:
            if not conversation_id:
                conversation_id = await self.create_conversation()
            elif conversation_id not in _conversations_store:
                _conversations_store[conversation_id] = []

            history = _conversations_store[conversation_id]

            messages = [{"role": "system", "content": self.instructions}]
            messages.extend(history)
            messages.append({"role": "user", "content": message})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )

            collected_chunks = []
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_chunks.append(content)
                    yield content

            full_text = "".join(collected_chunks).strip()
            if full_text:
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": full_text})
                _conversations_store[conversation_id] = history

        except Exception as e:
            logger.error(f"Error saat streaming dari OpenRouter API: {str(e)}")
            raise e

# Dependency provider untuk FastAPI
def get_openrouter_service() -> OpenRouterService:
    return OpenRouterService()

# Backward compatibility aliases
GeminiService = OpenRouterService
get_gemini_service = get_openrouter_service
OpenAIService = OpenRouterService
get_openai_service = get_openrouter_service


