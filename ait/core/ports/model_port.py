from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ModelPort(ABC):
    """
    Abstract base class for model operations.
    """

    # TODO: work on decoupling
    @abstractmethod
    def build_model(self, schema: str) -> Type[BaseModel]:
        """
        Builds a Pydantic model from the given schema.

        Args:
            schema (str): The schema to build the model from

        Returns:
            Type[BaseModel]: The built model
        """
        pass

    @abstractmethod
    def inject_type(
        self,
        model: Type[T],
        fields: list[tuple[str, Any]],
    ) -> Type[T]:
        """
        Injects field types into a model.
        """
        pass

    @abstractmethod
    def reduce_model_schema(
        self, model: Type[T], include_description: bool = True
    ) -> str:
        """
        Reduces the model schema into version with less tokens. Helpful for reducing prompt noise.
        """
        pass
