import logging
from typing import Any

from app.core.config import settings
from app.core.exceptions import RetrievalException
from app.core.schemas import DocumentChunk
from app.services.embeddings import EmbeddingsService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RetrieverService:
    """
    Сервис для семантического поиска документов.

    Обеспечивает интеллектуальный поиск документов на основе семантической
    близости, с поддержкой фильтрации по метаданным, обеспечения разнообразия
    результатов и поиска по источникам. Координирует работу сервисов
    эмбеддингов и векторного хранилища.

    Attributes
    ----------
        embeddings_service: Сервис для создания векторных представлений запросов.
        vector_store_service: Сервис для поиска в векторной базе данных.

    """

    def __init__(
        self, embeddings_service: EmbeddingsService, vector_store_service: VectorStoreService
    ) -> None:
        """
        Инициализация сервиса поиска.

        Настраивает зависимости для выполнения семантического поиска:
        сервис создания эмбеддингов и векторное хранилище документов.

        Args:
        ----
            embeddings_service: Сервис для генерации векторных представлений
                текстовых запросов.
            vector_store_service: Сервис для хранения и поиска документов
                в векторной базе данных.

        """
        self.embeddings_service = embeddings_service
        self.vector_store_service = vector_store_service

        logger.info("Инициализирован сервис поиска документов")

    async def search_similar_documents(
        self,
        query: str,
        max_docs: int | None = None,
        threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """
        Поиск документов, семантически похожих на запрос.

        Выполняет семантический поиск документов на основе векторного
        представления запроса. Поддерживает фильтрацию по релевантности
        и метаданным документов.

        Args:
        ----
            query: Текстовый поисковый запрос. Не может быть пустым.
            max_docs: Максимальное количество возвращаемых документов.
                Если None, используется значение из настроек.
            threshold: Порог семантической схожести от 0.0 до 1.0.
                Документы с меньшей схожестью исключаются.
            filters: Словарь фильтров для поиска по метаданным документов.

        Returns:
        -------
            Список объектов DocumentChunk, отсортированных по релевантности
            в убывающем порядке.

        Raises:
        ------
            RetrievalException: При пустом запросе или ошибках поиска
                в векторной базе данных.

        """
        if not query.strip():
            raise RetrievalException("Пустой поисковый запрос")

        # Использование настроек по умолчанию если не заданы
        max_docs = max_docs or settings.max_retrieved_docs
        threshold = threshold or settings.similarity_threshold

        try:
            logger.debug(f"Поиск документов для запроса: '{query[:50]}...'")

            # Генерация эмбеддинга запроса
            query_embedding = await self.embeddings_service.embed_text(query)

            # Поиск в векторном хранилище
            search_results = await self.vector_store_service.search_similar(
                query_embedding=query_embedding,
                limit=max_docs,
                threshold=threshold,
                where=filters,
            )

            # Преобразование результатов в DocumentChunk
            documents = []
            for i, doc_content in enumerate(search_results["documents"]):
                metadata = search_results["metadatas"][i] if search_results["metadatas"] else {}
                similarity_score = search_results["distances"][i]

                document_chunk = DocumentChunk(
                    content=doc_content,
                    metadata=metadata,
                    score=similarity_score,
                )
                documents.append(document_chunk)

            logger.info(f"Найдено {len(documents)} документов с порогом схожести {threshold}")

            return documents

        except Exception as e:
            logger.error(f"Ошибка поиска документов: {e}")
            raise RetrievalException(f"Не удалось выполнить поиск: {e}") from e

    async def search_by_metadata(
        self,
        query: str,
        metadata_filters: dict[str, Any],
        max_docs: int | None = None,
        threshold: float | None = None,
    ) -> list[DocumentChunk]:
        """
        Поиск документов с учетом фильтров метаданных.

        Удобный метод для семантического поиска с дополнительной
        фильтрацией по метаданным документов.

        Args:
        ----
            query: Текстовый поисковый запрос.
            metadata_filters: Словарь фильтров для поиска по метаданным.
            max_docs: Максимальное количество возвращаемых документов.
            threshold: Порог семантической схожести.

        Returns:
        -------
            Список объектов DocumentChunk, соответствующих критериям поиска.

        """
        return await self.search_similar_documents(
            query=query,
            max_docs=max_docs,
            threshold=threshold,
            filters=metadata_filters,
        )

    async def get_documents_by_source(
        self, source_filename: str, max_docs: int | None = None
    ) -> list[DocumentChunk]:
        """
        Получение всех документов из определенного источника.

        Извлекает все фрагменты документов, принадлежащие указанному файлу,
        независимо от семантической релевантности.

        Args:
        ----
            source_filename: Имя файла источника для поиска документов.
            max_docs: Максимальное количество возвращаемых документов.

        Returns:
        -------
            Список объектов DocumentChunk из указанного источника.

        Raises:
        ------
            RetrievalException: При ошибках поиска в векторной базе данных.

        """
        try:
            # Используем фиктивный запрос для получения всех документов источника
            search_results = await self.vector_store_service.search_similar(
                query_embedding=[0.0] * settings.embedding_dimension,  # Нулевой вектор
                limit=max_docs or 50,
                threshold=0.0,  # Очень низкий порог
                where={"filename": source_filename},
            )

            documents = []
            for i, doc_content in enumerate(search_results["documents"]):
                metadata = search_results["metadatas"][i] if search_results["metadatas"] else {}

                document_chunk = DocumentChunk(
                    content=doc_content,
                    metadata=metadata,
                    score=1.0,  # Максимальная релевантность для точного совпадения
                )
                documents.append(document_chunk)

            logger.info(f"Найдено {len(documents)} документов из источника '{source_filename}'")
            return documents

        except Exception as e:
            logger.error(f"Ошибка поиска по источнику {source_filename}: {e}")
            raise RetrievalException(f"Не удалось найти документы из источника: {e}") from e

    async def get_diverse_results(
        self,
        query: str,
        max_docs: int | None = None,
        diversity_threshold: float = 0.8,
    ) -> list[DocumentChunk]:
        """
        Получение разнообразных результатов поиска (избегание дублирования).

        Выполняет семантический поиск с последующей фильтрацией для обеспечения
        разнообразия результатов. Исключает слишком похожие документы для
        предоставления более широкого спектра информации.

        Args:
        ----
            query: Текстовый поисковый запрос.
            max_docs: Максимальное количество документов в итоговом результате.
            diversity_threshold: Порог разнообразия от 0.0 до 1.0. Документы
                с схожестью выше этого порога считаются дублирующими.

        Returns:
        -------
            Список разнообразных объектов DocumentChunk, отфильтрованных
            для минимизации дублирования информации.

        """
        # Сначала получаем больше результатов чем нужно
        extended_max = (max_docs or settings.max_retrieved_docs) * 2

        all_documents = await self.search_similar_documents(
            query=query,
            max_docs=extended_max,
            threshold=settings.similarity_threshold,
        )

        if not all_documents:
            return []

        # Фильтрация для обеспечения разнообразия
        diverse_documents = [all_documents[0]]  # Берем первый как базовый

        for candidate in all_documents[1:]:
            # Проверяем, насколько кандидат похож на уже выбранные
            is_diverse = True

            for selected in diverse_documents:
                # Простая проверка на основе содержимого (можно улучшить)
                similarity = self._calculate_text_similarity(candidate.content, selected.content)

                if similarity > diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                diverse_documents.append(candidate)

            # Останавливаемся когда достигли нужного количества
            if len(diverse_documents) >= (max_docs or settings.max_retrieved_docs):
                break

        logger.debug(
            f"Отфильтровано {len(diverse_documents)} разнообразных документов "
            f"из {len(all_documents)}"
        )

        return diverse_documents

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Простой расчет схожести текстов на основе пересечения слов.

        Использует коэффициент Жаккара для оценки лексического сходства
        между двумя текстами путем анализа пересечения множеств слов.

        Args:
        ----
            text1: Первый текст для сравнения.
            text2: Второй текст для сравнения.

        Returns:
        -------
            Коэффициент схожести от 0.0 до 1.0, где 1.0 означает
            полное совпадение словарного состава.

        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    async def get_retrieval_stats(self) -> dict[str, Any]:
        """
        Получение статистики работы сервиса поиска.

        Собирает комплексную информацию о состоянии всех компонентов
        поисковой системы: векторного хранилища, сервиса эмбеддингов
        и конфигурации поиска.

        Returns
        -------
            Словарь со статистикой поисковой системы, включающий:
            - vector_store: статистика векторного хранилища
            - embeddings_service: статус сервиса эмбеддингов
            - search_config: текущая конфигурация поиска

        """
        try:
            vector_stats = await self.vector_store_service.get_collection_stats()
            embeddings_health = await self.embeddings_service.health_check()

            return {
                "vector_store": vector_stats,
                "embeddings_service": embeddings_health,
                "search_config": {
                    "max_retrieved_docs": settings.max_retrieved_docs,
                    "similarity_threshold": settings.similarity_threshold,
                    "embedding_model": settings.embedding_model,
                },
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики поиска: {e}")
            return {"status": "error", "message": str(e)}
