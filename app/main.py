import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.core.config import settings
from app.core.exceptions import BaseAppException
from app.services.rag_service import RAGService

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Глобальный экземпляр RAG сервиса
rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Управление жизненным циклом приложения.

    Выполняет инициализацию и завершение работы RAG сервиса при запуске
    и остановке приложения. Включает автоматическую индексацию документов
    при первом запуске.

    Args:
    ----
        app (FastAPI): Экземпляр FastAPI приложения.

    Yields:
    ------
        None: Управление передается основному приложению.

    """
    # Запуск
    logger.info("Запуск RAG сервиса...")

    try:
        # Инициализация RAG сервиса
        global rag_service
        rag_service = RAGService()

        # Проверка наличия документов и их автоиндексация
        stats = await rag_service.get_knowledge_base_stats()
        documents_count = stats.get("collection", {}).get("documents_count", 0)

        if documents_count == 0:
            logger.info("База знаний пуста, запуск автоиндексации...")

            # Проверяем наличие файлов документов
            documents_info = stats.get("documents_directory", {})
            if documents_info.get("files_count", 0) > 0:
                result = await rag_service.reload_knowledge_base()
                if result["status"] == "success":
                    logger.info(
                        f"Автоиндексация завершена: {result['total_chunks']} чанков из "
                        f"{len(result['processed_files'])} файлов"
                    )
                else:
                    logger.warning(
                        f"Автоиндексация завершена с предупреждениями: {result['message']}"
                    )
            else:
                logger.warning("Не найдено документов для индексации в директории")
        else:
            logger.info(f"База знаний содержит {documents_count} документов")

        # Проверка здоровья системы
        health = await rag_service.get_system_health()
        if health["status"] == "healthy":
            logger.info("RAG система готова к работе")
        else:
            logger.warning("RAG система запущена с предупреждениями")

    except Exception as e:
        logger.error(f"Ошибка инициализации сервиса: {e}")
        # Продолжаем запуск даже при ошибках инициализации

    yield

    # Завершение
    logger.info("Завершение работы RAG сервиса...")


def create_application() -> FastAPI:
    """
    Создание и настройка FastAPI приложения.

    Создает экземпляр FastAPI с полной конфигурацией, включая middleware,
    роутеры, обработчики исключений и документацию API.

    Returns
    -------
        FastAPI: Настроенный экземпляр приложения.

    """
    app = FastAPI(
        title="RAG Knowledge Base API",
        description="API для поиска по базе знаний компании с использованием RAG",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Настройка CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # В продакшене ограничить конкретными доменами
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Подключение роутеров
    app.include_router(v1_router, prefix="/api")

    # Глобальный обработчик исключений приложения
    @app.exception_handler(BaseAppException)
    async def app_exception_handler(request: Request, exc: BaseAppException) -> JSONResponse:
        """
        Глобальный обработчик кастомных исключений.

        Логирует и форматирует ответ для исключений типа BaseAppException.

        Args:
        ----
            request (Request): HTTP запрос.
            exc (BaseAppException): Кастомное исключение приложения.

        Returns:
        -------
            JSONResponse: JSON ответ с информацией об ошибке.

        """
        logger.error(f"Исключение приложения: {exc.message}", extra={"details": exc.details})

        return JSONResponse(
            status_code=500,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    # Обработчик общих исключений
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Обработчик неожиданных исключений.

        Логирует детали исключения и возвращает общий ответ об ошибке сервера.

        Args:
        ----
            request (Request): HTTP запрос.
            exc (Exception): Неожиданное исключение.

        Returns:
        -------
            JSONResponse: JSON ответ с общей информацией об ошибке.

        """
        logger.error(f"Неожиданное исключение: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "Внутренняя ошибка сервера",
                "details": {},
            },
        )

    # Базовый endpoint
    @app.get("/")
    async def root() -> dict[str, Any]:
        """
        Корневой endpoint с информацией о сервисе.

        Возвращает базовую информацию о статусе и доступных endpoint'ах API.

        Returns
        -------
            dict[str, Any]: Информация о сервисе и доступных endpoint'ах.

        """
        return {
            "service": "RAG Knowledge Base API",
            "version": "1.0.0",
            "status": "running",
            "docs_url": "/docs",
            "health_check": "/api/v1/health",
        }

    return app


# Создание экземпляра приложения
app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
