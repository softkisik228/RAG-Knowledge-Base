import logging
import os
from typing import Any

import aiofiles  # type: ignore

from app.core.config import settings
from app.core.exceptions import DocumentProcessingException

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Загрузчик документов компании.

    Обеспечивает загрузку и валидацию текстовых документов из файловой системы.
    Поддерживает различные форматы файлов, извлекает метаданные и обрабатывает
    ошибки кодировки. Предназначен для подготовки документов к обработке
    в RAG системе.

    Attributes
    ----------
        supported_extensions: Множество поддерживаемых расширений файлов.

    """

    def __init__(self) -> None:
        """
        Инициализация загрузчика.

        Настраивает поддерживаемые форматы файлов и подготавливает
        загрузчик к работе с документами.

        """
        self.supported_extensions = {".txt", ".md"}

    async def load_documents(self) -> list[dict[str, Any]]:
        """
        Загрузка всех документов из директории.

        Рекурсивно сканирует директорию документов, загружает все
        поддерживаемые файлы и извлекает их метаданные. Обрабатывает
        ошибки отдельных файлов без прерывания общего процесса.

        Returns
        -------
            Список словарей с документами, где каждый содержит:
            - content: текстовое содержимое документа
            - metadata: метаданные файла (имя, размер, даты и т.д.)

        Raises
        ------
            DocumentProcessingException: При отсутствии директории или
                критических ошибках загрузки.

        """
        if not os.path.exists(settings.documents_path):
            raise DocumentProcessingException(
                f"Директория с документами не найдена: {settings.documents_path}"
            )

        documents: list[dict[str, Any]] = []

        try:
            files = [
                f
                for f in os.listdir(settings.documents_path)
                if any(f.lower().endswith(ext) for ext in self.supported_extensions)
            ]

            if not files:
                logger.warning(f"Не найдено документов в {settings.documents_path}")
                return documents

            for filename in files:
                try:
                    doc = await self._load_single_document(filename)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Ошибка загрузки {filename}: {e}")
                    continue

            logger.info(f"Загружено {len(documents)} документов")
            return documents

        except Exception as e:
            logger.error(f"Ошибка загрузки документов: {e}")
            raise DocumentProcessingException(f"Не удалось загрузить документы: {e}") from e

    async def _load_single_document(self, filename: str) -> dict[str, Any] | None:
        """
        Загрузка одного документа.

        Читает файл с диска, извлекает его содержимое и метаданные.
        Обрабатывает ошибки кодировки и пустые файлы.

        Args:
        ----
            filename: Имя файла в директории документов.

        Returns:
        -------
            Словарь с содержимым и метаданными документа, или None
            при ошибках загрузки или пустом файле.

        """
        filepath = os.path.join(settings.documents_path, filename)

        try:
            async with aiofiles.open(filepath, "r", encoding="utf-8") as file:
                content = await file.read()

            if not content.strip():
                logger.warning(f"Пустой файл: {filename}")
                return None

            # Получение информации о файле
            stat = os.stat(filepath)

            return {
                "content": content.strip(),
                "metadata": {
                    "filename": filename,
                    "source": filepath,
                    "file_size": stat.st_size,
                    "file_type": os.path.splitext(filename)[1].lower(),
                    "created_at": stat.st_ctime,
                    "modified_at": stat.st_mtime,
                },
            }

        except UnicodeDecodeError:
            logger.error(f"Ошибка кодировки файла {filename}")
            return None
        except Exception as e:
            logger.error(f"Ошибка чтения файла {filename}: {e}")
            return None

    async def load_document_by_name(self, filename: str) -> dict[str, Any]:
        """
        Загрузка конкретного документа по имени.

        Загружает указанный файл с валидацией его типа и обработкой ошибок.
        Предназначен для загрузки отдельных документов по запросу.

        Args:
        ----
            filename: Имя файла для загрузки. Должен иметь поддерживаемое
                расширение.

        Returns:
        -------
            Словарь с содержимым и метаданными загруженного документа.

        Raises:
        ------
            DocumentProcessingException: При неподдерживаемом типе файла
                или ошибках загрузки.

        """
        if not any(filename.lower().endswith(ext) for ext in self.supported_extensions):
            raise DocumentProcessingException(
                f"Неподдерживаемый тип файла: {filename}. "
                f"Поддерживаемые: {', '.join(self.supported_extensions)}"
            )

        document = await self._load_single_document(filename)
        if not document:
            raise DocumentProcessingException(f"Не удалось загрузить документ: {filename}")

        return document

    def get_supported_extensions(self) -> set[str]:
        """
        Получение поддерживаемых расширений файлов.

        Возвращает копию множества поддерживаемых расширений файлов
        для внешнего использования без возможности модификации.

        Returns
        -------
            Множество строк с поддерживаемыми расширениями файлов
            (например, {'.txt', '.md'}).

        """
        return self.supported_extensions.copy()

    async def validate_documents_directory(self) -> dict[str, Any]:
        """
        Валидация директории с документами.

        Проверяет существование и доступность директории документов,
        анализирует содержащиеся файлы и разделяет их на поддерживаемые
        и неподдерживаемые типы.

        Returns
        -------
            Словарь с информацией о состоянии директории, включающий:
            - exists: существует ли директория
            - readable: доступна ли для чтения
            - files_count: количество поддерживаемых файлов
            - supported_files: список поддерживаемых файлов
            - unsupported_files: список неподдерживаемых файлов
            - error: описание ошибки (при наличии)

        """
        result: dict[str, Any] = {
            "exists": os.path.exists(settings.documents_path),
            "readable": False,
            "files_count": 0,
            "supported_files": [],
            "unsupported_files": [],
        }

        if not result["exists"]:
            return result

        try:
            result["readable"] = os.access(settings.documents_path, os.R_OK)

            if result["readable"]:
                all_files = os.listdir(settings.documents_path)

                for filename in all_files:
                    if any(filename.lower().endswith(ext) for ext in self.supported_extensions):
                        result["supported_files"].append(filename)
                    else:
                        result["unsupported_files"].append(filename)

                result["files_count"] = len(result["supported_files"])

        except Exception as e:
            logger.error(f"Ошибка валидации директории: {e}")
            result["error"] = str(e)

        return result
