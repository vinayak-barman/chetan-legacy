import datetime
from typing import Any, NotRequired, Optional, Required, Self, TypedDict
from uuid import UUID, uuid4
from llama_cpp import List, Literal, Union
from pydantic import BaseModel

from chetan.core.types import FunctionCall, ToolCall


Tag = Literal["Retrieval", "Memory", "Recommendation", "Other"]
Source = Literal["agent", "system", "user"]


class Feedback(BaseModel):
    reward: float
    feedback: str
    positive: bool
    source: Literal["model", "user"] = (
        "user"  # Accept feedback from the user or a reward model
    )


class Message(TypedDict):
    """Base class for messages"""

    name: NotRequired[str]
    role: str
    content: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_id: NotRequired[str]


class IterationItem(BaseModel):
    id: UUID = uuid4()
    timestamp: datetime.datetime = datetime.datetime.now()
    source: Source
    feedback: Optional[str] = None

    def export(self) -> Message:
        raise NotImplementedError("Export method must be implemented")


class UserMessage(IterationItem):
    content: str
    source: Source = "user"

    def export(self) -> Message:
        return Message(
            role="user",
            content=self.content,
        )


class PreludeItem(IterationItem):
    content: Union[str, dict, BaseModel]
    tag: Tag
    source: Source = "system"

    def export(self) -> Message:
        return Message(
            role="system",
            content=self.content,
        )


class Processing(IterationItem):
    content: str
    source: Source = "agent"

    def export(self) -> Message:
        return Message(
            role="assistant",
            content=self.content,
        )


class ToolCalling(IterationItem):
    calls: List[ToolCall]
    source: Source = "agent"

    def export(self) -> Message:
        return Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=str(call["id"]),
                    function=FunctionCall(
                        name=call["function"]["name"],
                        arguments=call["function"]["arguments"],
                    ),
                    type="function",
                )
                for call in self.calls
            ],
        )


class Results(IterationItem):
    results: Any
    id: str
    source: Source = "system"

    def export(self) -> Message:
        return Message(
            role="tool",
            content=self.results,
            tool_call_id=self.id,
        )


class StateUpdate(IterationItem):
    state: dict
    source: Source = "agent"

    def export(self) -> Message:
        return Message(
            role="system",
            content=str(self.state),
        )


# Class-level mappings as frozen sets and dictionaries
_TYPE_MAP = {
    "prelude": PreludeItem,
    "processing": Processing,
    "tool_calling": ToolCalling,
    "results": Results,
    "state_update": StateUpdate,
    "user_message": UserMessage,
}
_NAME_MAP = {v: k for k, v in _TYPE_MAP.items()}
_VALID_ITEM_ORDER = {
    "prelude": {"prelude", "processing"},  # Using sets for O(1) lookup
    "processing": {"tool_calling"},
    "tool_calling": {"results"},
    "results": {"results", "state_update"},
}


class Iteration(BaseModel):
    id: UUID = uuid4()
    counter: int = 0
    items: List[IterationItem] = []
    ended: bool = False

    @classmethod
    def _name_to_type(cls, name: str) -> type:
        try:
            return _TYPE_MAP[name]
        except KeyError:
            raise ValueError(f"Unknown item name: {name}")

    @classmethod
    def _type_to_name(cls, type_: type) -> str:
        try:
            return _NAME_MAP[type_]
        except KeyError:
            raise ValueError(f"Unknown item type: {type_}")

    def add(self, item: IterationItem):
        if self.ended:
            raise ValueError("Iteration has ended")

        if not self.items:  # Handle empty items list
            if not (isinstance(item, PreludeItem) or isinstance(item, Processing)):
                raise ValueError("First item must be a PreludeItem or Processing")

            self.items.append(item)
            return

        last_type = type(self.items[-1])
        last_name = self._type_to_name(last_type)
        next_name = self._type_to_name(type(item))

        if (
            next_name not in _VALID_ITEM_ORDER[last_name]
            and next_name != "user_message"
        ):  # User messages can be added anywhere in the context
            raise ValueError(f"Invalid item order: {last_name} -> {next_name}")

        if type(item) == PreludeItem:
            self.ended = True

        self.items.append(item)


class PersistentContext(BaseModel):
    items: List[Union[Iteration, UserMessage]] = []
