from abc import ABC, abstractmethod
from typing import Type, TypedDict, Union

from pydantic import BaseModel


class RawMessage(TypedDict):
    content: str
    role: str


class LanguageModel(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
    
    @abstractmethod
    def _generate_structured(
        self,
        context: list[RawMessage],
        target_model: Type[BaseModel],
        iterable: bool,
        *args,
        **kwargs,
    ): ...

    @abstractmethod
    def _generate_raw(self, context: list[RawMessage], *args, **kwargs): ...

    def generate(
        self,
        context: list[RawMessage],
        target_model: Union[Type[BaseModel], None] = None,
        tools: list[dict] = None,
        *args,
        iterable: bool = False,
        **kwargs,
    ):
        kwargs = {**self.kwargs, **kwargs}
        if target_model is None:
            return self._generate_raw(context, *args, tools=tools, **kwargs)

        if issubclass(target_model, BaseModel):
            return self._generate_structured(
                context, target_model, iterable, *args, **kwargs
            )

        return "Invalid target model"
