from http import HTTPStatus
from typing import AsyncGenerator, Type

import instructor
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ait.core.domain.errors import LLMAdapterError
from ait.core.domain.interfaces import CompletionResponse, T
from ait.core.ports import LLMPort


class InstructorAdapter(LLMPort):
    """
    Instructor implementation of the LLM port.
    """

    def __init__(
        self,
        model: str,
        embedding_model: str,
        api_key: str,
    ):
        self._model = model
        self._embedding_model = embedding_model

        # cache = AutoCache(maxsize=1000)
        openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url="http://localhost:11434/v1",
        )
        self.client = instructor.from_openai(
            client=openai_client,
            mode=instructor.Mode.JSON,
        )

        # self.client = instructor.from_openai(
        #     client=client,
        #     model=model,
        #     # async_client=True,
        #     base_url="http://localhost:11434",
        # )
        # self.client = instructor.from_provider(
        #     model=model,
        #     api_key=api_key,
        #     cache=cache,
        #     async_client=True,
        #     base_url="http://localhost:11434",
        # )

    async def chat(self, messages: list[dict[str, str]]) -> CompletionResponse:
        """
        Sends a message to the LLM and returns a structured response.

        Args:
            messages (list[dict[str, str]]): The messages to generate a response to

        Returns:
            str: The response from the LLM
        """
        output: ChatCompletion = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            stream=False,
        )
        response = output.choices[0].message.content
        if not response:
            raise LLMAdapterError(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                message="No response from the model",
            )
        return CompletionResponse(
            completion=output,
            content=response,
        )

    async def stream(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[CompletionResponse, None]:
        """
        Streams text outputs from the model.

        Args:
            messages (list[dict[str, str]]): The messages to generate a response to

        Returns:
            AsyncGenerator[str, None]: The response from the LLM
        """
        output: AsyncGenerator[
            ChatCompletionChunk, None
        ] = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            stream=True,
        )
        async for chunk in output:
            response = chunk.choices[0].delta.content
            if not response:
                continue
            yield CompletionResponse(
                completion=chunk,
                content=response,
            )

    async def asend(
        self,
        messages: list[dict[str, str]],
        response_model: Type[T],
    ) -> CompletionResponse[T]:
        """
        Sends a message to the LLM asynchronously and returns a structured response.

        Args:
            messages (list[dict[str, str]]): The messages to generate a response to
            response_model (Type[T]): The model to return the response as

        Returns:
            CompletionResponse[T]: The response from the LLM
        """

        (
            instance,
            completion,
        ) = await self.client.chat.completions.create_with_completion(
            response_model=response_model,
            model=self._model,
            messages=messages,  # type: ignore
        )
        if not instance:
            raise LLMAdapterError(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                message="No response content from the model",
            )
        return CompletionResponse(
            completion=completion,  # type: ignore
            content=instance,
        )
