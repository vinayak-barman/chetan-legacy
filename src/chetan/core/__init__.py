from typing import List, Required, TypedDict
from pydantic import BaseModel, Field

from chetan.actions import Action


class Function(TypedDict):
    name: str
    description: str
    parameters: dict
    strict: bool = False


class Tool(TypedDict):
    type: str = "function"
    function: Function

    def create(action: Action):
        return Tool(
            function=Function(
                name=action.name,
                description=action.description,
                parameters=action.args.model_json_schema(),
            )
        )


# class Reasoning(BaseModel):
#     interpretation: str = Field(
#         title="Interpretation",
#         description="What do you think is happening?",
#     )
#     traces: List[str] = Field(
#         title="Traces",
#         description="Give points to support your interpretation",
#     )


# class StateItem(BaseModel):
#     name: str
#     value: str


# class State(BaseModel):
#     items: List[StateItem] = Field(
#         title="State Items",
#         description="List of state items",
#     )
