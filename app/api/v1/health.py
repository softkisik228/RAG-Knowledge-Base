import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.core.schemas import HealthResponse
from app.services.rag_service import RAGService

router = APIRouter(tags=["health"])
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


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Проверка здоровья сервиса и его компонентов.

    Выполняет базовую проверку состояния RAG системы, включая
    векторную базу данных и количество документов в базе знаний.

    Returns
    -------
        HealthResponse: Статус здоровья системы.

    Raises
    ------
        HTTPException: При критических ошибках системы.

    """
    try:
        service = await get_rag_service()
        health_data = await service.get_system_health()

        # Определение общего статуса
        is_healthy = health_data.get("status") == "healthy"

        # Извлечение информации о компонентах
        vector_db_status = "healthy"
        documents_count = 0

        if "vector_store" in health_data:
            vector_store_info = health_data["vector_store"]
            vector_db_status = vector_store_info.get("status", "unknown")

        if "collection" in health_data:
            collection_info = health_data["collection"]
            documents_count = collection_info.get("documents_count", 0)

        # Получение моделей из конфигурации
        config = health_data.get("configuration", {})
        embeddings_model = config.get("embedding_model", "unknown")
        llm_model = config.get("llm_model", "unknown")

        response = HealthResponse(
            status="healthy" if is_healthy else "unhealthy",
            vector_db_status=vector_db_status,
            documents_count=documents_count,
            embeddings_model=embeddings_model,
            llm_model=llm_model,
        )

        # Если система не здорова, возвращаем соответствующий HTTP статус
        if not is_healthy:
            logger.warning("Система не здорова", extra={"health_data": health_data})
            return response

        return response

    except Exception as e:
        logger.error(f"Критическая ошибка проверки здоровья: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис недоступен",
        ) from e


@router.get("/health/detailed")
async def detailed_health_check() -> dict[str, Any]:
    """
    Детальная проверка здоровья всех компонентов системы.

    Возвращает подробную информацию о состоянии векторной базы данных,
    сервиса эмбеддингов, конфигурации системы и директории документов.

    Returns
    -------
        dict[str, Any]: Подробная информация о состоянии системы.

    Raises
    ------
        HTTPException: При ошибке получения информации.

    """
    try:
        service = await get_rag_service()
        health_data = await service.get_system_health()

        # Добавление дополнительной информации
        health_data["timestamp"] = int(__import__("time").time())
        health_data["service_name"] = "RAG Knowledge Base"
        health_data["version"] = "1.0.0"

        return health_data

    except Exception as e:
        logger.error(f"Ошибка детальной проверки здоровья: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка получения информации о системе",
        ) from e


@router.get("/readiness")
async def readiness_check() -> dict[str, Any]:
    """
    Проверка готовности сервиса к обработке запросов.

    Проверяет критически важные компоненты: эмбеддинги, векторное хранилище,
    LLM и наличие загруженных документов.

    Returns
    -------
        dict[str, Any]: Статус готовности и информация о компонентах.

    """
    try:
        service = await get_rag_service()
        health_data = await service.get_system_health()

        # Проверяем критически важные компоненты
        is_ready = (
            health_data.get("embeddings", {}).get("status") == "healthy"
            and health_data.get("vector_store", {}).get("status") == "healthy"
            and health_data.get("llm_status") == "healthy"
            and health_data.get("collection", {}).get("documents_count", 0) > 0
        )

        status_text = "ready" if is_ready else "not_ready"

        return {
            "status": status_text,
            "ready": is_ready,
            "components": {
                "embeddings": health_data.get("embeddings", {}).get("status", "unknown"),
                "vector_store": health_data.get("vector_store", {}).get("status", "unknown"),
                "llm": health_data.get("llm_status", "unknown"),
                "documents_loaded": health_data.get("collection", {}).get("documents_count", 0) > 0,
            },
        }

    except Exception as e:
        logger.error(f"Ошибка проверки готовности: {e}")
        return {
            "status": "not_ready",
            "ready": False,
            "error": str(e),
        }


@router.get("/liveness")
async def liveness_check() -> dict[str, str]:
    """
    Базовая проверка жизнеспособности сервиса.

    Простая проверка того, что сервис запущен и может отвечать на запросы.

    Returns
    -------
        dict[str, str]: Статус жизнеспособности.

    """
    # Простая проверка - сервис запущен и может отвечать
    return {"status": "alive"}
