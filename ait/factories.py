import os

from ait.adapters import InstructorAdapter, Jinja2Adapter, PydanticAdapter
from ait.core.ports import FormatterPort, LLMPort, ModellerPort


def create_llm_client(
    model: str,
    embedding_model: str,
    api_key: str,
) -> LLMPort:
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
    model = model
    embedding_model = embedding_model or os.getenv(
        "EMBEDDING_MODEL", "text-embedding-ada-002"
    )
    return InstructorAdapter(
        model=model,
        embedding_model=embedding_model,
        api_key=api_key,
    )


def create_prompt_formatter() -> FormatterPort:
    """
    Factory function to create a PromptFormatter instance.

    Returns:
        PromptFormatter: Configured prompt formatter instance
    """
    return Jinja2Adapter()


def create_model_handler() -> ModellerPort:
    """
    Factory function to create a ResponseModelService instance.

    Returns:
        ResponseModelService: Configured model handler instance
    """
    return PydanticAdapter()
