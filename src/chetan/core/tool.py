from typing import Any, Dict, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from chetan.actions import Action


class Tool(BaseModel):
    type: str = "function"
    function: Dict[str, Any]

    @classmethod
    def create(cls, action: "Action", qualified_name: str):
        
        params = action.args.model_json_schema()
        params.pop("title", None)
        
        return cls(
            type="function",
            function={
                "name": qualified_name.replace(
                    ".", "-"
                ),  # We replace the dots with hyphens, since OpenAI doesn't allow dots in the function name
                "description": action.description,
                "parameters": params,
            },
        ).model_dump()
