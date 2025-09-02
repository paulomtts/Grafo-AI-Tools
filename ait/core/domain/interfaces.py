from typing import Generic, Optional, TypeVar

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class CompletionResponse(BaseModel, Generic[T]):
    """
    Data model for completion response.
    """

    completion: ChatCompletion | ChatCompletionChunk
    model: str
    content: str | T

    @property
    def response_model(self) -> T:
        """
        Returns the instance of the response model of the completion response.
        """
        if isinstance(self.content, str) or isinstance(self.content, list):
            raise ValueError("Content is not structured.")
        return self.content


class BaseEvaluation(BaseModel):
    is_valid: bool = Field(description="Whether the response is valid or not.")
    reasoning: str = Field(description="Reasoning about the validity of the response.")
    humanized_failure_reason: Optional[str] = Field(
        default=None, description="A humanized failure reason for the response."
    )
