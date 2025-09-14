import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Настройки приложения.

    Класс для управления конфигурацией приложения с автоматической
    загрузкой настроек из переменных окружения и .env файла.

    Attributes
    ----------
        openai_api_key: Ключ API OpenAI для работы с языковыми моделями.
        chroma_persist_directory: Директория для хранения данных ChromaDB.
        collection_name: Название коллекции в векторной базе данных.
        embedding_model: Модель для создания эмбеддингов текста.
        embedding_dimension: Размерность векторных представлений.
        llm_model: Модель языковой модели для генерации ответов.
        llm_temperature: Температура для генерации (контроль случайности).
        llm_max_tokens: Максимальное количество токенов в ответе.
        similarity_threshold: Пороговое значение для семантического поиска.
        max_retrieved_docs: Максимальное количество документов для поиска.
        chunk_size: Размер чанков при разбиении документов.
        chunk_overlap: Перекрытие между соседними чанками.
        documents_directory: Директория с документами для индексации.
        log_level: Уровень логирования приложения.

    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI API
    openai_api_key: str = Field(..., description="Ключ API OpenAI")

    # ChromaDB
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", description="Директория для хранения ChromaDB"
    )
    collection_name: str = Field(
        default="company_knowledge", description="Название коллекции в ChromaDB"
    )

    # Embeddings
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Модель эмбеддингов OpenAI"
    )
    embedding_dimension: int = Field(default=1536, description="Размерность эмбеддингов")

    # LLM
    llm_model: str = Field(default="gpt-3.5-turbo", description="Модель языковой модели")
    llm_temperature: float = Field(default=0.1, description="Температура генерации")
    llm_max_tokens: int = Field(default=1000, description="Максимальное количество токенов")

    # Retrieval
    similarity_threshold: float = Field(
        default=0.3, description="Порог схожести для поиска документов"
    )
    max_retrieved_docs: int = Field(
        default=5, description="Максимальное количество извлекаемых документов"
    )
    chunk_size: int = Field(default=1000, description="Размер чанка документа")
    chunk_overlap: int = Field(default=200, description="Перекрытие между чанками")

    # API
    log_level: str = Field(default="INFO", description="Уровень логирования")
    api_version: str = Field(default="v1", description="Версия API")

    @property
    def documents_path(self) -> str:
        """Путь к директории с документами."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents")

    def model_dump_safe(self) -> dict[str, Any]:
        """Безопасный дамп настроек без секретных данных."""
        data = self.model_dump()
        data["openai_api_key"] = "***" if self.openai_api_key else None
        return data


settings = Settings()  # type: ignore
