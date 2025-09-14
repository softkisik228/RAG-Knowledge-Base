"""Pydantic схемы для API."""
from typing import Any

from pydantic import BaseModel, Field


# === Модели документов ===
class DocumentChunk(BaseModel):
    """
    Фрагмент документа.

    Представляет один фрагмент (чанк) документа с его содержимым,
    метаданными и оценкой релевантности для поискового запроса.

    Attributes
    ----------
        content: Текстовое содержимое фрагмента документа.
        metadata: Словарь с метаданными фрагмента (источник, дата и т.д.).
        score: Оценка релевантности фрагмента к поисковому запросу (0.0-1.0).

    """

    content: str = Field(..., description="Текст фрагмента")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Метаданные фрагмента")
    score: float = Field(..., description="Оценка релевантности", ge=0.0, le=1.0)


# === Модели запросов ===
class QueryRequest(BaseModel):
    """
    Запрос для поиска или Q&A.

    Модель для запросов пользователя к системе поиска или вопросно-ответной системе.
    Включает параметры для настройки качества и количества результатов поиска.

    Attributes
    ----------
        question: Вопрос или поисковый запрос пользователя.
        max_docs: Максимальное количество документов в результатах поиска.
        threshold: Минимальный порог схожести для включения документа в результаты.

    """

    question: str = Field(..., description="Вопрос пользователя", min_length=1, max_length=500)
    max_docs: int = Field(
        default=5, description="Максимальное количество документов для поиска", ge=1, le=10
    )
    threshold: float = Field(default=0.3, description="Порог схожести для поиска", ge=0.0, le=1.0)


# === Модели ответов ===
class SearchResponse(BaseModel):
    """
    Ответ семантического поиска.

    Результат поиска документов по семантическому сходству с запросом.
    Содержит найденные документы и информацию о поиске.

    Attributes
    ----------
        query: Исходный поисковый запрос пользователя.
        documents: Список найденных фрагментов документов с оценками релевантности.
        total_found: Общее количество найденных документов.

    """

    query: str = Field(..., description="Исходный запрос")
    documents: list[DocumentChunk] = Field(..., description="Найденные документы")
    total_found: int = Field(..., description="Общее количество найденных документов", ge=0)


class QAResponse(BaseModel):
    """
    Ответ системы вопрос-ответ.

    Результат обработки вопроса через RAG систему, включающий
    сгенерированный ответ и источники информации.

    Attributes
    ----------
        query: Исходный вопрос пользователя.
        answer: Сгенерированный языковой моделью ответ.
        sources: Список фрагментов документов, использованных для генерации ответа.
        confidence: Уверенность системы в правильности ответа.

    """

    query: str = Field(..., description="Исходный вопрос")
    answer: str = Field(..., description="Сгенерированный ответ")
    sources: list[DocumentChunk] = Field(..., description="Использованные источники")
    confidence: float = Field(..., description="Уверенность в ответе", ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """
    Ответ проверки здоровья сервиса.

    Информация о состоянии основных компонентов RAG системы
    для мониторинга и диагностики проблем.

    Attributes
    ----------
        status: Общий статус работоспособности сервиса.
        vector_db_status: Статус векторной базы данных.
        documents_count: Количество документов, загруженных в систему.
        embeddings_model: Название используемой модели для эмбеддингов.
        llm_model: Название используемой языковой модели.

    """

    status: str = Field(..., description="Статус сервиса")
    vector_db_status: str = Field(..., description="Статус векторной базы данных")
    documents_count: int = Field(..., description="Количество документов в базе", ge=0)
    embeddings_model: str = Field(..., description="Используемая модель эмбеддингов")
    llm_model: str = Field(..., description="Используемая языковая модель")


class ReloadResponse(BaseModel):
    """
    Ответ переиндексации документов.

    Результат операции переиндексации базы знаний с информацией
    о процессе обработки документов.

    Attributes
    ----------
        status: Статус завершения операции переиндексации.
        processed_files: Список имен файлов, которые были обработаны.
        total_chunks: Общее количество созданных текстовых фрагментов.
        message: Описательное сообщение о результате операции.

    """

    status: str = Field(..., description="Статус операции")
    processed_files: list[str] = Field(..., description="Список обработанных файлов")
    total_chunks: int = Field(..., description="Общее количество созданных чанков", ge=0)
    message: str = Field(..., description="Сообщение о результате операции")


class ErrorResponse(BaseModel):
    """
    Ответ с ошибкой.

    Стандартный формат для передачи информации об ошибках
    в API ответах.

    Attributes
    ----------
        error: Тип или код ошибки.
        message: Человекочитаемое описание ошибки.
        details: Дополнительные детали ошибки для отладки.

    """

    error: str = Field(..., description="Тип ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    details: dict[str, Any] = Field(default_factory=dict, description="Детали ошибки")
