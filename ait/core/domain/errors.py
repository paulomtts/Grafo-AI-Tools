import logging

from httpx import HTTPError


class BaseError(HTTPError):
    """Base error class for all custom exceptions."""

    def __init__(
        self,
        status_code: int,
        message: str,
    ):
        self.status_code = status_code
        super().__init__(message=message)
        logger = logging.getLogger("ROOT")
        logger.error(
            f"\033[91m{self.__class__.__name__} ({self.status_code}): {str(message)}\033[0m"
        )


class LLMAdapterError(BaseError):
    """
    Exception raised when an error occurs in the LLM adapter.
    """
