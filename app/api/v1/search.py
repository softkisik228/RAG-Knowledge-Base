import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.exceptions import BaseAppException, LLMException, RetrievalException
from app.core.schemas import QAResponse, QueryRequest, ReloadResponse, SearchResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)

# Глобальный экземпляр RAG сервиса
rag_service: RAGService | None = None


async def get_rag_service() -> RAGService:
    """
    Получение экземпляра RAG сервиса.

    Создает глобальный экземпляр RAG сервиса при первом обращении
    или возвращает существующий.

    Returns
    -------
        RAGService: Экземпляр RAG сервиса.

    """
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service


@router.post("/query", response_model=QAResponse)
async def query_knowledge_base(request: QueryRequest) -> QAResponse:
    """
    Задать вопрос базе знаний компании.

    Обрабатывает пользовательский вопрос с помощью RAG системы,
    находит релевантные документы и генерирует ответ с помощью LLM.

    Args:
    ----
        request (QueryRequest): Запрос с вопросом и параметрами.

    Returns:
    -------
        QAResponse: Ответ с найденными источниками и сгенерированным текстом.

    Raises:
    ------
        HTTPException: При ошибке обработки запроса.

    """
    try:
        logger.info(f"Получен вопрос: '{request.question[:100]}...'")

        service = await get_rag_service()
        response = await service.answer_question(
            question=request.question,
            max_docs=request.max_docs,
            threshold=request.threshold,
        )

        return QAResponse(**response.model_dump())

    except LLMException as e:
        logger.error(f"Ошибка LLM: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ошибка языковой модели: {e.message}",
        ) from e
    except RetrievalException as e:
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ошибка поиска документов: {e.message}",
        ) from e
    except BaseAppException as e:
        logger.error(f"Ошибка приложения: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка: {e.message}",
        ) from e
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера",
        ) from e


@router.post("/similar", response_model=SearchResponse)
async def search_similar_documents(request: QueryRequest) -> SearchResponse:
    """
    Семантический поиск похожих документов.

    Находит документы, наиболее семантически близкие к запросу пользователя,
    без генерации ответа с помощью LLM.

    Args:
    ----
        request (QueryRequest): Запрос с текстом для поиска.

    Returns:
    -------
        SearchResponse: Список найденных документов с оценками релевантности.

    Raises:
    ------
        HTTPException: При ошибке поиска.

    """
    try:
        logger.info(f"Поиск документов для: '{request.question[:100]}...'")

        service = await get_rag_service()
        response = await service.search_documents(
            query=request.question,
            max_docs=request.max_docs,
            threshold=request.threshold,
        )

        return SearchResponse(**response.model_dump())

    except RetrievalException as e:
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ошибка поиска документов: {e.message}",
        ) from e
    except BaseAppException as e:
        logger.error(f"Ошибка приложения: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка: {e.message}",
        ) from e
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера",
        ) from e


@router.post("/reload", response_model=ReloadResponse)
async def reload_knowledge_base() -> ReloadResponse | JSONResponse:
    """
    Переиндексация базы знаний из документов.

    Перезагружает все документы из директории и создает новые эмбеддинги.
    Полезно после обновления документов или изменения конфигурации.

    Returns
    -------
        ReloadResponse | JSONResponse: Результат операции переиндексации.

    Raises
    ------
        HTTPException: При ошибке переиндексации.

    """
    try:
        logger.info("Запуск переиндексации базы знаний")

        service = await get_rag_service()
        result = await service.reload_knowledge_base()

        response = ReloadResponse(
            status=result["status"],
            processed_files=result["processed_files"],
            total_chunks=result["total_chunks"],
            message=result["message"],
        )

        status_code = status.HTTP_200_OK
        if result["status"] == "warning":
            status_code = status.HTTP_202_ACCEPTED
        elif result["status"] == "error":
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        return JSONResponse(
            status_code=status_code,
            content=response.model_dump(),
        )

    except BaseAppException as e:
        logger.error(f"Ошибка переиндексации: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка переиндексации: {e.message}",
        ) from e
    except Exception as e:
        logger.error(f"Неожиданная ошибка переиндексации: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка переиндексации базы знаний",
        ) from e


@router.get("/stats")
async def get_knowledge_base_stats() -> dict[str, Any]:
    """
    Получение статистики базы знаний.

    Возвращает подробную информацию о состоянии базы знаний,
    включая количество документов, конфигурацию и статус компонентов.

    Returns
    -------
        dict[str, Any]: Статистика системы и базы знаний.

    Raises
    ------
        HTTPException: При ошибке получения статистики.

    """
    try:
        service = await get_rag_service()
        stats = await service.get_knowledge_base_stats()
        return stats

    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка получения статистики",
        ) from e
