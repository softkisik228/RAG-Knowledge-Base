import logging
import os
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.exceptions import VectorStoreException

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Сервис для работы с векторной базой данных.

    Обеспечивает операции с векторным хранилищем ChromaDB: добавление документов,
    семантический поиск, управление коллекциями и мониторинг состояния.
    Использует косинусную метрику для измерения сходства векторов.

    Attributes
    ----------
        _client: Клиент ChromaDB для взаимодействия с базой данных.
        _collection: Коллекция документов в векторной базе.

    """

    def __init__(self) -> None:
        """
        Инициализация векторного хранилища.

        Создает и настраивает клиент ChromaDB, инициализирует коллекцию
        для хранения документов с косинусной метрикой. Создает необходимые
        директории и логирует состояние инициализации.

        Raises
        ------
            VectorStoreException: При ошибке создания клиента ChromaDB
                или инициализации коллекции.

        """
        try:
            # Создание директории для ChromaDB
            os.makedirs(settings.chroma_persist_directory, exist_ok=True)

            # Инициализация ChromaDB клиента
            self._client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Получение или создание коллекции с явным указанием метрики
            self._collection = self._client.get_or_create_collection(
                name=settings.collection_name,
                metadata={"description": "База знаний компании", "hnsw:space": "cosine"},
            )

            logger.info(
                f"Инициализирована коллекция '{settings.collection_name}' "
                f"с {self._collection.count()} документами"
            )

        except Exception as e:
            logger.error(f"Ошибка инициализации ChromaDB: {e}")
            raise VectorStoreException(
                f"Не удалось инициализировать векторное хранилище: {e}"
            ) from e

    async def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """
        Добавление документов в векторное хранилище.

        Сохраняет тексты документов вместе с их векторными представлениями
        в ChromaDB. Автоматически генерирует идентификаторы и метаданные
        при их отсутствии.

        Args:
        ----
            texts: Список текстов документов для добавления.
            embeddings: Список векторных представлений документов,
                должен соответствовать по длине списку текстов.
            metadatas: Список словарей с метаданными для каждого документа.
                Если None, создаются автоматически с chunk_id.
            ids: Список уникальных идентификаторов документов.
                Если None, генерируются UUID.

        Raises:
        ------
            VectorStoreException: При пустых входных данных, несоответствии
                длин списков или ошибках добавления в ChromaDB.

        """
        if not texts or not embeddings:
            raise VectorStoreException("Пустые списки текстов или эмбеддингов")

        if len(texts) != len(embeddings):
            raise VectorStoreException(
                f"Количество текстов ({len(texts)}) не соответствует "
                f"количеству эмбеддингов ({len(embeddings)})"
            )

        try:
            # Генерация ID если не предоставлены
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]

            # Создание метаданных если не предоставлены
            if metadatas is None:
                metadatas = [{"chunk_id": i} for i in range(len(texts))]

            # Добавление документов
            self._collection.add(
                embeddings=embeddings,  # type: ignore
                documents=texts,
                metadatas=metadatas,  # type: ignore
                ids=ids,
            )

            logger.info(f"Добавлено {len(texts)} документов в векторное хранилище")

        except Exception as e:
            logger.error(f"Ошибка добавления документов: {e}")
            raise VectorStoreException(f"Не удалось добавить документы: {e}") from e

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 5,
        threshold: float = 0.7,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Поиск похожих документов.

        Выполняет семантический поиск документов в векторной базе на основе
        косинусного сходства. Фильтрует результаты по порогу релевантности
        и поддерживает дополнительные фильтры по метаданным.

        Args:
        ----
            query_embedding: Векторное представление поискового запроса.
            limit: Максимальное количество документов в результатах.
            threshold: Порог сходства от 0.0 до 1.0. Документы с меньшим
                сходством исключаются из результатов.
            where: Словарь условий для фильтрации по метаданным документов.

        Returns:
        -------
            Словарь с результатами поиска, содержащий:
            - documents: список текстов найденных документов
            - metadatas: список метаданных документов
            - distances: список оценок сходства (от 0.0 до 1.0)
            - ids: список идентификаторов документов

        Raises:
        ------
            VectorStoreException: При ошибках выполнения поиска в ChromaDB.

        """
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],  # type: ignore
                n_results=limit,
                where=where,
            )

            # Фильтрация по порогу схожести (ChromaDB использует расстояние, не схожесть)
            filtered_results: dict[str, list[Any]] = {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }

            if results["documents"] and results["documents"][0] and results["distances"]:
                for i, distance in enumerate(results["distances"][0]):
                    # Для косинусного расстояния: схожесть = 1 - расстояние
                    # ChromaDB возвращает косинусное расстояние в диапазоне [0, 2]
                    # где 0 = идентичные векторы, 2 = противоположные векторы
                    similarity = 1 - (distance / 2)  # Нормализуем к [0, 1]

                    if similarity >= threshold:
                        filtered_results["documents"].append(results["documents"][0][i])
                        if results["metadatas"] and results["metadatas"][0]:
                            filtered_results["metadatas"].append(results["metadatas"][0][i])
                        filtered_results["distances"].append(similarity)
                        filtered_results["ids"].append(results["ids"][0][i])
            else:
                logger.debug("Нет результатов поиска или пустые списки")

            logger.debug(
                f"Найдено {len(filtered_results['documents'])} документов "
                f"с порогом схожести {threshold}"
            )

            return filtered_results

        except Exception as e:
            logger.error(f"Ошибка поиска документов: {e}")
            raise VectorStoreException(f"Не удалось выполнить поиск: {e}") from e

    async def get_collection_stats(self) -> dict[str, Any]:
        """
        Получение статистики коллекции.

        Возвращает информацию о состоянии коллекции документов:
        количество документов, статус и метаданные коллекции.

        Returns
        -------
            Словарь со статистикой коллекции, включающий:
            - name: имя коллекции
            - documents_count: количество документов в коллекции
            - status: статус коллекции ('active', 'empty', 'error')
            - error: описание ошибки (при наличии)

        """
        try:
            count = self._collection.count()
            return {
                "name": settings.collection_name,
                "documents_count": count,
                "status": "active" if count > 0 else "empty",
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {
                "name": settings.collection_name,
                "documents_count": 0,
                "status": "error",
                "error": str(e),
            }

    async def clear_collection(self) -> None:
        """
        Очистка коллекции.

        Удаляет все документы из коллекции векторной базы данных.
        Операция необратима - все векторы и метаданные будут потеряны.

        Raises
        ------
            VectorStoreException: При ошибках доступа к коллекции
                или удаления документов.

        """
        try:
            # Получаем все ID и удаляем
            all_data = self._collection.get()
            if all_data["ids"]:
                self._collection.delete(ids=all_data["ids"])
                logger.info(f"Очищена коллекция '{settings.collection_name}'")
        except Exception as e:
            logger.error(f"Ошибка очистки коллекции: {e}")
            raise VectorStoreException(f"Не удалось очистить коллекцию: {e}") from e

    async def health_check(self) -> dict[str, Any]:
        """
        Проверка здоровья векторного хранилища.

        Выполняет диагностику состояния ChromaDB: проверяет соединение,
        доступность коллекции и собирает статистику для мониторинга.

        Returns
        -------
            Словарь с информацией о состоянии хранилища, включающий:
            - status: статус хранилища ('healthy', 'unhealthy')
            - collection: статистика коллекции документов
            - client_version: версия клиента ChromaDB
            - error: описание ошибки (при неудаче)

        """
        try:
            stats = await self.get_collection_stats()
            # Проверяем подключение
            self._client.heartbeat()

            return {
                "status": "healthy",
                "collection": stats,
                "client_version": chromadb.__version__,
            }
        except Exception as e:
            logger.error(f"Проблемы с векторным хранилищем: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "client_version": chromadb.__version__,
            }
