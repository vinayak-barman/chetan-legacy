from typing import TypedDict


class FunctionCall(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    function: FunctionCall
    type: str = "function"
