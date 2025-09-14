"""OpenAI API клиент."""
import logging
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import ConfigurationException

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Клиент для работы с OpenAI API."""

    def __init__(self) -> None:
        """Инициализация клиента."""
        try:
            self._embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                max_retries=3,
            )

            self._llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                max_retries=3,
            )

            self._client = OpenAI(api_key=settings.openai_api_key)

            logger.info(f"OpenAI клиент инициализирован: {settings.llm_model}")

        except Exception as e:
            logger.error(f"Ошибка инициализации OpenAI клиента: {e}")
            raise ConfigurationException(f"Не удалось инициализировать OpenAI клиент: {e}") from e

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Получить сервис эмбеддингов."""
        return self._embeddings

    @property
    def llm(self) -> ChatOpenAI:
        """Получить языковую модель."""
        return self._llm

    @property
    def client(self) -> OpenAI:
        """Получить базовый клиент OpenAI."""
        return self._client

    async def health_check(self) -> dict[str, Any]:
        """Проверить доступность OpenAI API."""
        try:
            # Простой тест API
            test_response = await self._embeddings.aembed_query("test")
            if test_response:
                return {
                    "status": "healthy",
                    "model": settings.llm_model,
                    "embedding_model": settings.embedding_model,
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Empty response from API",
                }
        except Exception as e:
            logger.error(f"Ошибка проверки OpenAI API: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }
