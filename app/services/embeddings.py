import asyncio
import logging
from typing import Any

import httpx
import tiktoken
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.exceptions import EmbeddingException

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """
    Сервис для генерации эмбеддингов.

    Обеспечивает создание векторных представлений текстов с использованием
    OpenAI API. Включает управление токенами, батчевую обработку документов
    и мониторинг состояния сервиса.

    Attributes
    ----------
        _embeddings: Клиент OpenAI для генерации эмбеддингов.
        _encoding: Токенизатор для подсчета токенов в тексте.

    """

    def __init__(self) -> None:
        """
        Инициализация сервиса эмбеддингов.

        Создает клиент OpenAI для генерации эмбеддингов и настраивает
        токенизатор для работы с текстами. Обрабатывает ошибки
        инициализации и логирует состояние сервиса.

        Raises
        ------
            EmbeddingException: При ошибке инициализации клиента OpenAI
                или токенизатора.

        """
        try:
            self._embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
                max_retries=3,
            )
            # Используем токенизатор cl100k_base для новых моделей OpenAI
            try:
                self._encoding = tiktoken.encoding_for_model(settings.embedding_model)
            except KeyError:
                # Fallback для новых моделей, которые используют cl100k_base
                self._encoding = tiktoken.get_encoding("cl100k_base")

            logger.info(f"Инициализирован сервис эмбеддингов: {settings.embedding_model}")
        except Exception as e:
            logger.error(f"Ошибка инициализации сервиса эмбеддингов: {e}")
            raise EmbeddingException(f"Не удалось инициализировать сервис эмбеддингов: {e}") from e

    async def embed_text(self, text: str) -> list[float]:
        """
        Генерация эмбеддинга для текста.

        Создает векторное представление входного текста с помощью OpenAI API.
        Автоматически обрезает текст до лимита токенов модели и выполняет
        валидацию входных данных.

        Args:
        ----
            text: Текст для создания эмбеддинга. Не должен быть пустым.

        Returns:
        -------
            Список чисел с плавающей точкой, представляющий векторное
            представление текста.

        Raises:
        ------
            EmbeddingException: При пустом тексте, ошибках API OpenAI
                или других проблемах генерации эмбеддинга.

        """
        if not text.strip():
            raise EmbeddingException("Пустой текст для генерации эмбеддинга")

        try:
            # Проверка длины текста (ограничение OpenAI)
            tokens_count = len(self._encoding.encode(text))
            if tokens_count > 8000:
                logger.warning(f"Текст содержит {tokens_count} токенов, обрезается до 8000")
                text = text[: 8000 * 4]  # Приблизительно 4 символа на токен

            # Генерация эмбеддинга в отдельном потоке
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self._embeddings.embed_query, text)

            logger.debug(f"Создан эмбеддинг размерностью {len(embedding)}")
            return embedding

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка при генерации эмбеддинга: {e}")
            raise EmbeddingException(f"Ошибка API OpenAI: {e.response.status_code}") from e
        except Exception as e:
            logger.error(f"Неожиданная ошибка при генерации эмбеддинга: {e}")
            raise EmbeddingException(f"Не удалось создать эмбеддинг: {e}") from e

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Генерация эмбеддингов для списка текстов.

        Эффективная батчевая обработка множественных текстов для создания
        векторных представлений. Автоматически фильтрует пустые тексты
        и обрабатывает ошибки.

        Args:
        ----
            texts: Список текстов для создания эмбеддингов. Пустые строки
                автоматически отфильтровываются.

        Returns:
        -------
            Список векторных представлений, где каждый элемент - это список
            чисел с плавающей точкой, соответствующий одному входному тексту.

        Raises:
        ------
            EmbeddingException: При отсутствии валидных текстов или ошибках
                генерации эмбеддингов.

        """
        if not texts:
            return []

        try:
            # Фильтрация пустых текстов
            valid_texts = [text.strip() for text in texts if text.strip()]
            if not valid_texts:
                raise EmbeddingException("Все тексты пустые")

            # Генерация эмбеддингов в отдельном потоке
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self._embeddings.embed_documents, valid_texts
            )

            logger.info(f"Создано {len(embeddings)} эмбеддингов")
            return embeddings

        except Exception as e:
            logger.error(f"Ошибка при генерации эмбеддингов для документов: {e}")
            raise EmbeddingException(f"Не удалось создать эмбеддинги: {e}") from e

    def count_tokens(self, text: str) -> int:
        """
        Подсчет количества токенов в тексте.

        Использует официальный токенизатор OpenAI для точного подсчета
        токенов в тексте. При ошибке возвращает приблизительную оценку.

        Args:
        ----
            text: Текст для анализа количества токенов.

        Returns:
        -------
            Количество токенов в тексте согласно модели токенизации OpenAI.

        """
        try:
            return len(self._encoding.encode(text))
        except Exception as e:
            logger.warning(f"Ошибка подсчета токенов: {e}")
            return len(text) // 4  # Приблизительная оценка

    async def health_check(self) -> dict[str, Any]:
        """
        Проверка здоровья сервиса эмбеддингов.

        Выполняет тестовую генерацию эмбеддинга для проверки
        работоспособности сервиса и доступности API OpenAI.

        Returns
        -------
            Словарь с информацией о состоянии сервиса, включающий:
            - status: статус сервиса ('healthy' или 'unhealthy')
            - model: используемая модель эмбеддингов
            - dimension: размерность векторов (при успехе)
            - error: описание ошибки (при неудаче)

        """
        try:
            # Тестовое создание эмбеддинга
            test_embedding = await self.embed_text("test")
            return {
                "status": "healthy",
                "model": settings.embedding_model,
                "dimension": len(test_embedding),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": settings.embedding_model,
                "error": str(e),
            }
