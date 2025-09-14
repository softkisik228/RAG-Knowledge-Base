import logging
import re
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.exceptions import DocumentProcessingException

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Предобработчик текстовых данных.

    Обеспечивает очистку, нормализацию и разбивку текстовых документов
    на чанки для эффективной обработки в RAG системе. Использует
    рекурсивный алгоритм разбивки с настраиваемыми параметрами размера
    и перекрытия чанков.

    Attributes
    ----------
        text_splitter: Объект LangChain для разбивки текста на чанки.

    """

    def __init__(self) -> None:
        """
        Инициализация предобработчика.

        Настраивает сплиттер текста с параметрами из конфигурации:
        размер чанков, перекрытие и иерархию разделителей для
        оптимального разбиения документов.

        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def process_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Обработка списка документов.

        Выполняет полную предобработку документов: очистку текста,
        разбивку на чанки и создание метаданных. Обрабатывает ошибки
        отдельных документов без прерывания общего процесса.

        Args:
        ----
            documents: Список словарей с документами, каждый содержит
                'content' (текст) и 'metadata' (метаданные).

        Returns:
        -------
            Список словарей с обработанными чанками, где каждый содержит:
            - content: очищенный текст чанка
            - metadata: расширенные метаданные с информацией о чанке

        Raises:
        ------
            DocumentProcessingException: При критических ошибках обработки
                всего набора документов.

        """
        if not documents:
            return []

        try:
            all_chunks = []

            for doc_index, document in enumerate(documents):
                try:
                    chunks = await self._process_single_document(document, doc_index)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(
                        f"Ошибка обработки документа {document.get('metadata', {}).get('filename', 'unknown')}: {e}"
                    )
                    continue

            logger.info(f"Создано {len(all_chunks)} чанков из {len(documents)} документов")
            return all_chunks

        except Exception as e:
            logger.error(f"Ошибка обработки документов: {e}")
            raise DocumentProcessingException(f"Не удалось обработать документы: {e}") from e

    async def _process_single_document(
        self, document: dict[str, Any], doc_index: int
    ) -> list[dict[str, Any]]:
        """
        Обработка одного документа.

        Выполняет очистку текста, разбивку на чанки и создание метаданных
        для отдельного документа. Обрабатывает пустые документы и ошибки.

        Args:
        ----
            document: Словарь с документом, содержащий 'content' и 'metadata'.
            doc_index: Порядковый номер документа в наборе для создания
                уникальных идентификаторов чанков.

        Returns:
        -------
            Список словарей с чанками документа, каждый содержит
            очищенный текст и расширенные метаданные.

        """
        content = document.get("content", "")
        metadata = document.get("metadata", {})

        if not content.strip():
            logger.warning(f"Пустой контент в документе {metadata.get('filename', 'unknown')}")
            return []

        try:
            # Очистка текста
            cleaned_content = self._clean_text(content)

            # Разбивка на чанки
            chunks = self.text_splitter.split_text(cleaned_content)

            # Создание чанков с метаданными
            processed_chunks = []
            for chunk_index, chunk in enumerate(chunks):
                if chunk.strip():  # Пропускаем пустые чанки
                    chunk_metadata = self._create_chunk_metadata(
                        metadata, doc_index, chunk_index, chunk
                    )

                    processed_chunks.append(
                        {
                            "content": chunk.strip(),
                            "metadata": chunk_metadata,
                        }
                    )

            logger.debug(
                f"Документ {metadata.get('filename')} разбит на {len(processed_chunks)} чанков"
            )

            return processed_chunks

        except Exception as e:
            logger.error(f"Ошибка обработки документа: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """
        Очистка текста от лишних символов и нормализация.

        Выполняет нормализацию пробелов, удаление недопустимых символов
        и стандартизацию пунктуации для подготовки текста к векторизации.

        Args:
        ----
            text: Исходный текст для очистки.

        Returns:
        -------
            Очищенный и нормализованный текст без лишних пробелов
            и специальных символов.

        """
        # Удаление лишних пробелов и переносов строк
        text = re.sub(r"\s+", " ", text)

        # Удаление специальных символов (оставляем только буквы, цифры, пунктуацию)
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\"\'\/\@\#\$\%\&\*\+\=\[\]\{\}]", "", text)

        # Нормализация пунктуации
        text = re.sub(r"\.{2,}", "...", text)  # Многоточие
        text = re.sub(r"\?{2,}", "?", text)  # Множественные вопросы
        text = re.sub(r"\!{2,}", "!", text)  # Множественные восклицания

        # Удаление множественных пробелов
        text = re.sub(r" {2,}", " ", text)

        return text.strip()

    def _create_chunk_metadata(
        self, base_metadata: dict[str, Any], doc_index: int, chunk_index: int, content: str
    ) -> dict[str, Any]:
        """
        Создание метаданных для чанка.

        Расширяет базовые метаданные документа информацией о чанке:
        индексы, размеры, количество слов и превью содержимого.

        Args:
        ----
            base_metadata: Исходные метаданные документа.
            doc_index: Индекс документа в наборе.
            chunk_index: Индекс чанка в документе.
            content: Текстовое содержимое чанка.

        Returns:
        -------
            Словарь с расширенными метаданными чанка, включающий
            всю информацию о документе и специфические данные чанка.

        """
        chunk_metadata = base_metadata.copy()

        # Добавление информации о чанке
        chunk_metadata.update(
            {
                "doc_index": doc_index,
                "chunk_index": chunk_index,
                "chunk_size": len(content),
                "word_count": len(content.split()),
                "chunk_id": f"{doc_index}_{chunk_index}",
            }
        )

        # Создание превью чанка (первые 100 символов)
        preview = content[:100] + "..." if len(content) > 100 else content
        chunk_metadata["preview"] = preview.replace("\n", " ")

        return chunk_metadata

    def get_chunk_stats(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Получение статистики по чанкам.

        Анализирует набор чанков и предоставляет детальную статистику
        о размерах, количествах и конфигурации процесса чанкинга.

        Args:
        ----
            chunks: Список обработанных чанков для анализа.

        Returns:
        -------
            Словарь со статистикой, включающий общие метрики,
            средние значения и информацию о конфигурации.

        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "total_words": 0,
                "avg_chunk_size": 0,
                "documents_count": 0,
            }

        total_chars = sum(len(chunk["content"]) for chunk in chunks)
        total_words = sum(len(chunk["content"].split()) for chunk in chunks)

        # Подсчет уникальных документов
        unique_docs = set()
        for chunk in chunks:
            filename = chunk.get("metadata", {}).get("filename", "unknown")
            unique_docs.add(filename)

        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
            "documents_count": len(unique_docs),
            "chunk_size_config": settings.chunk_size,
            "chunk_overlap_config": settings.chunk_overlap,
        }

    async def reprocess_with_different_params(
        self, documents: list[dict[str, Any]], chunk_size: int, chunk_overlap: int
    ) -> list[dict[str, Any]]:
        """
        Переобработка документов с другими параметрами чанкинга.

        Временно изменяет параметры разбивки текста и повторно обрабатывает
        документы с новыми настройками. Полезно для экспериментов с
        различными размерами чанков.

        Args:
        ----
            documents: Список документов для переобработки.
            chunk_size: Новый размер чанков в символах.
            chunk_overlap: Новое перекрытие между чанками в символах.

        Returns:
        -------
            Список чанков, созданных с новыми параметрами разбивки.

        """
        # Временно создаем новый splitter с другими параметрами
        original_splitter = self.text_splitter

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        try:
            chunks = await self.process_documents(documents)
            return chunks
        finally:
            # Восстанавливаем исходный splitter
            self.text_splitter = original_splitter
