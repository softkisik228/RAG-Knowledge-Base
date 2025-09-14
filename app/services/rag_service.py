import logging
from typing import Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.exceptions import LLMException
from app.core.schemas import DocumentChunk, QAResponse, SearchResponse
from app.data.loader import DocumentLoader
from app.data.preprocessor import TextPreprocessor
from app.services.embeddings import EmbeddingsService
from app.services.retriever import RetrieverService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RAGService:
    """
    Основной сервис RAG системы.

    Координирует работу всех компонентов RAG (Retrieval-Augmented Generation):
    загрузку документов, создание эмбеддингов, поиск релевантной информации
    и генерацию ответов с помощью языковой модели.

    Attributes
    ----------
        embeddings_service: Сервис для создания векторных представлений текста.
        vector_store_service: Сервис для работы с векторной базой данных.
        retriever_service: Сервис для поиска релевантных документов.
        llm: Языковая модель для генерации ответов.
        qa_chain: Цепочка обработки вопросов и ответов.
        loader: Загрузчик документов из файловой системы.
        preprocessor: Препроцессор для обработки и разбиения текстов.

    """

    def __init__(self) -> None:
        """
        Инициализация RAG сервиса.

        Создает и настраивает все необходимые компоненты системы:
        сервисы эмбеддингов, векторного хранилища, поиска, языковую модель
        и цепочку обработки вопросов.

        """
        # Инициализация компонентов
        self.embeddings_service = EmbeddingsService()
        self.vector_store_service = VectorStoreService()
        self.retriever_service = RetrieverService(
            self.embeddings_service, self.vector_store_service
        )
        self.document_loader = DocumentLoader()
        self.text_preprocessor = TextPreprocessor()

        # Инициализация LLM
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )

        # Настройка промпта для Q&A
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Ты - помощник по базе знаний компании. Отвечай на вопросы, используя только предоставленную информацию.

Контекст из документов компании:
{context}

Вопрос: {question}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Если информации недостаточно, честно скажи об этом
- Будь точным и конкретным
- Используй профессиональный тон
- Если возможно, укажи источник информации

Ответ:""",
        )

        # Создание цепочки для Q&A
        self.qa_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)

        logger.info("Инициализирован RAG сервис")

    async def search_documents(
        self,
        query: str,
        max_docs: int = 5,
        threshold: float = 0.7,
    ) -> SearchResponse:
        """
        Семантический поиск документов.

        Args:
        ----
            query: Поисковый запрос
            max_docs: Максимальное количество документов
            threshold: Порог релевантности

        Returns:
        -------
            Результаты поиска

        """
        logger.info(f"Поиск документов по запросу: '{query[:50]}...'")

        documents = await self.retriever_service.search_similar_documents(
            query=query,
            max_docs=max_docs,
            threshold=threshold,
        )

        return SearchResponse(
            query=query,
            documents=documents,
            total_found=len(documents),
        )

    async def answer_question(
        self,
        question: str,
        max_docs: int = 5,
        threshold: float = 0.7,
    ) -> QAResponse:
        """
        Ответ на вопрос с использованием RAG.

        Args:
        ----
            question: Вопрос пользователя
            max_docs: Максимальное количество документов для контекста
            threshold: Порог релевантности документов

        Returns:
        -------
            Ответ с источниками

        Raises:
        ------
            LLMException: При ошибке генерации ответа

        """
        logger.info(f"Обработка вопроса: '{question[:50]}...'")

        try:
            # Поиск релевантных документов
            relevant_documents = await self.retriever_service.search_similar_documents(
                query=question,
                max_docs=max_docs,
                threshold=threshold,
            )

            if not relevant_documents:
                # Если не найдено релевантных документов
                return QAResponse(
                    query=question,
                    answer="К сожалению, в базе знаний не найдено информации для ответа на ваш вопрос. Попробуйте переформулировать запрос или обратиться к администратору.",
                    sources=[],
                    confidence=0.0,
                )

            # Формирование контекста из найденных документов
            context = self._build_context(relevant_documents)

            # Генерация ответа с помощью LLM
            response = await self.qa_chain.arun(
                context=context,
                question=question,
            )

            # Расчет уверенности на основе релевантности документов
            confidence = self._calculate_confidence(relevant_documents)

            logger.info(f"Сгенерирован ответ с уверенностью {confidence:.2f}")

            return QAResponse(
                query=question,
                answer=response.strip(),
                sources=relevant_documents,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            raise LLMException(f"Не удалось сгенерировать ответ: {e}") from e

    def _build_context(self, documents: list[DocumentChunk]) -> str:
        """
        Построение контекста из релевантных документов.

        Объединяет содержимое найденных документов в единый контекст
        для передачи языковой модели, добавляя информацию об источниках.

        Args:
        ----
            documents: Список релевантных документов с метаданными и содержимым.

        Returns:
        -------
            Объединенный контекст в виде форматированной строки
            с нумерацией документов и указанием источников.

        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source_info = ""
            if doc.metadata:
                filename = doc.metadata.get("filename", "Неизвестный источник")
                source_info = f" (Источник: {filename})"

            context_parts.append(f"[Документ {i}]{source_info}:\n{doc.content}")

        return "\n\n".join(context_parts)

    def _calculate_confidence(self, documents: list[DocumentChunk]) -> float:
        """
        Расчет уверенности в ответе на основе релевантности документов.

        Вычисляет уровень уверенности в сгенерированном ответе на основе
        средней релевантности найденных документов и их количества.

        Args:
        ----
            documents: Список релевантных документов с оценками релевантности.

        Returns:
        -------
            Уровень уверенности от 0.0 до 1.0, где 1.0 означает максимальную
            уверенность в точности ответа.

        """
        if not documents:
            return 0.0

        # Средняя релевантность документов
        avg_score = sum(doc.score for doc in documents) / len(documents)

        # Бонус за количество документов (больше источников = выше уверенность)
        quantity_bonus = min(0.1 * len(documents), 0.3)

        # Финальная уверенность
        confidence = min(avg_score + quantity_bonus, 1.0)

        return round(confidence, 3)

    async def reload_knowledge_base(self) -> dict[str, Any]:
        """
        Переиндексация базы знаний.

        Полная переиндексация базы знаний: очистка существующей коллекции,
        загрузка документов из директории, их предобработка, создание
        эмбеддингов и сохранение в векторной базе данных.

        Returns
        -------
            Словарь с результатами операции переиндексации, включающий:
            - status: статус операции ('success', 'warning', 'error')
            - processed_files: список обработанных файлов
            - total_chunks: количество созданных чанков
            - message: описание результата операции

        """
        logger.info("Начало переиндексации базы знаний")

        try:
            # Очистка текущей коллекции
            await self.vector_store_service.clear_collection()

            # Загрузка документов
            documents = await self.document_loader.load_documents()

            if not documents:
                logger.warning("Не найдено документов для индексации")
                return {
                    "status": "warning",
                    "processed_files": [],
                    "total_chunks": 0,
                    "message": "Не найдено документов для индексации",
                }

            # Предобработка документов
            chunks = await self.text_preprocessor.process_documents(documents)

            if not chunks:
                logger.warning("Не создано чанков после предобработки")
                return {
                    "status": "warning",
                    "processed_files": [doc["metadata"]["filename"] for doc in documents],
                    "total_chunks": 0,
                    "message": "Не удалось создать чанки из документов",
                }

            # Генерация эмбеддингов
            texts = [chunk["content"] for chunk in chunks]
            embeddings = await self.embeddings_service.embed_documents(texts)

            # Подготовка метаданных и ID
            metadatas = [chunk["metadata"] for chunk in chunks]
            ids = [f"{chunk['metadata']['chunk_id']}" for chunk in chunks]

            # Добавление в векторную базу
            await self.vector_store_service.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

            processed_files = list({doc["metadata"]["filename"] for doc in documents})

            logger.info(
                f"Переиндексация завершена: {len(processed_files)} файлов, " f"{len(chunks)} чанков"
            )

            return {
                "status": "success",
                "processed_files": processed_files,
                "total_chunks": len(chunks),
                "message": f"Успешно переиндексировано {len(chunks)} чанков из {len(processed_files)} файлов",
            }

        except Exception as e:
            logger.error(f"Ошибка переиндексации: {e}")
            return {
                "status": "error",
                "processed_files": [],
                "total_chunks": 0,
                "message": f"Ошибка переиндексации: {e}",
            }

    async def get_system_health(self) -> dict[str, Any]:
        """
        Проверка здоровья всей RAG системы.

        Выполняет комплексную проверку всех компонентов системы:
        сервиса эмбеддингов, векторной базы данных, языковой модели
        и наличия документов в коллекции.

        Returns
        -------
            Словарь с состоянием компонентов системы, включающий:
            - status: общий статус системы ('healthy', 'unhealthy', 'error')
            - embeddings: состояние сервиса эмбеддингов
            - vector_store: состояние векторной базы данных
            - llm_status: состояние языковой модели
            - collection: статистика коллекции документов
            - configuration: текущая конфигурация системы

        """
        try:
            # Проверка компонентов
            embeddings_health = await self.embeddings_service.health_check()
            vector_store_health = await self.vector_store_service.health_check()
            collection_stats = await self.vector_store_service.get_collection_stats()

            # Тест генерации ответа
            try:
                test_response = await self.llm.apredict("Привет! Это тест.")
                llm_status = "healthy" if test_response else "unhealthy"
            except Exception as e:
                logger.error(f"Ошибка теста LLM: {e}")
                llm_status = "unhealthy"

            # Общий статус системы
            system_healthy = (
                embeddings_health.get("status") == "healthy"
                and vector_store_health.get("status") == "healthy"
                and llm_status == "healthy"
                and collection_stats.get("documents_count", 0) > 0
            )

            return {
                "status": "healthy" if system_healthy else "unhealthy",
                "embeddings": embeddings_health,
                "vector_store": vector_store_health,
                "llm_status": llm_status,
                "collection": collection_stats,
                "configuration": {
                    "llm_model": settings.llm_model,
                    "embedding_model": settings.embedding_model,
                    "max_retrieved_docs": settings.max_retrieved_docs,
                    "similarity_threshold": settings.similarity_threshold,
                },
            }

        except Exception as e:
            logger.error(f"Ошибка проверки здоровья системы: {e}")
            return {
                "status": "error",
                "message": str(e),
            }

    async def get_knowledge_base_stats(self) -> dict[str, Any]:
        """
        Получение статистики базы знаний.

        Собирает подробную статистику о состоянии всех компонентов
        системы управления знаниями: векторной базы, поиска,
        директории документов и конфигурации.

        Returns
        -------
            Словарь со статистикой системы, включающий:
            - collection: статистика векторной коллекции
            - retrieval: статистика операций поиска
            - documents_directory: информация о директории документов
            - system_config: текущая конфигурация системы

        """
        try:
            # Статистика векторной базы
            collection_stats = await self.vector_store_service.get_collection_stats()

            # Статистика поиска
            retrieval_stats = await self.retriever_service.get_retrieval_stats()

            # Валидация директории с документами
            documents_info = await self.document_loader.validate_documents_directory()

            return {
                "collection": collection_stats,
                "retrieval": retrieval_stats,
                "documents_directory": documents_info,
                "system_config": settings.model_dump_safe(),
            }

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {"status": "error", "message": str(e)}
