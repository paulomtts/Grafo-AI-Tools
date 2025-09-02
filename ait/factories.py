import os

from ait import LLMClient, PromptFormatter, ResponseModelService


def create_llm_client(
    model: str,
    embedding_model: str,
    api_key: str,
) -> LLMClient:
    """
    Factory function to create an LLMClient instance with default configuration.

    Args:
        model (Optional[str]): The model to use for completions. Defaults to LLM_MODEL env var.
        embedding_model (Optional[str]): The model to use for embeddings. Defaults to EMBEDDING_MODEL env var.

    Returns:
        LLMClient: Configured LLM client instance

    Raises:
        ValueError: If required configuration is missing
    """
    model = model or os.getenv("LLM_MODEL", "gpt-5-mini")
    embedding_model = embedding_model or os.getenv(
        "EMBEDDING_MODEL", "text-embedding-ada-002"
    )
    return LLMClient(
        model=model,
        embedding_model=embedding_model,
        api_key=api_key,
    )


def create_prompt_formatter() -> PromptFormatter:
    """
    Factory function to create a PromptFormatter instance.

    Returns:
        PromptFormatter: Configured prompt formatter instance
    """
    return PromptFormatter()


def create_model_handler() -> ResponseModelService:
    """
    Factory function to create a ResponseModelService instance.

    Returns:
        ResponseModelService: Configured model handler instance
    """
    return ResponseModelService()
