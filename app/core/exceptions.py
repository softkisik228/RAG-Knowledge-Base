"""Кастомные исключения."""
from typing import Any


class BaseAppException(Exception):
    """
    Базовое исключение приложения.

    Родительский класс для всех кастомных исключений в приложении.
    Предоставляет стандартный интерфейс для обработки ошибок.

    Attributes
    ----------
        message: Описание ошибки для пользователя.
        details: Дополнительная техническая информация об ошибке.

    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """
        Инициализация исключения.

        Args:
        ----
            message (str): Сообщение об ошибке.
            details (dict[str, Any] | None): Дополнительные детали ошибки.

        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class VectorStoreException(BaseAppException):
    """
    Исключение при работе с векторной базой данных.

    Возникает при ошибках взаимодействия с ChromaDB или другими
    векторными хранилищами данных.
    """


class EmbeddingException(BaseAppException):
    """
    Исключение при генерации эмбеддингов.

    Возникает при ошибках создания векторных представлений текста
    с помощью модели эмбеддингов.
    """


class DocumentProcessingException(BaseAppException):
    """
    Исключение при обработке документов.

    Возникает при ошибках загрузки, парсинга или обработки
    документов из файловой системы.
    """


class RetrievalException(BaseAppException):
    """
    Исключение при поиске документов.

    Возникает при ошибках семантического поиска или получения
    релевантных документов из векторной базы данных.
    """


class LLMException(BaseAppException):
    """
    Исключение при работе с языковой моделью.

    Возникает при ошибках взаимодействия с OpenAI API или
    генерации ответов языковой моделью.
    """

    pass
    """Исключение при работе с языковой моделью."""


class ConfigurationException(BaseAppException):
    """Исключение конфигурации."""
