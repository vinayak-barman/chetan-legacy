from typing import List, Optional, Type
from uuid import uuid4
from pydantic import BaseModel
from chetan.core.context.iteration import Message
from chetan.core.types import FunctionCall, ToolCall
from chetan.lm import LanguageModel
from openai import OpenAI
import instructor

import ollama


class OllamaLM(LanguageModel):
    ins_client: instructor.Instructor
    model: str

    def __init__(self, model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.ins_client = instructor.from_openai(
            OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
            *args,
            **kwargs,
        )

    def _generate_structured(
        self,
        context: list[Message],
        target_model: Type[BaseModel],
        iterable: bool,
        *args,
        **kwargs,
    ):
        if iterable:
            return [
                x
                for x in self.ins_client.chat.completions.create_iterable(
                    response_model=target_model,
                    messages=context,
                    model=self.model,
                    *args,
                    **kwargs,
                )
            ]

        res = self.ins_client.chat.completions.create(
            response_model=target_model,
            messages=context,
            model=self.model,
            *args,
            **kwargs,
        )
        return res

    def _generate_raw(self, context: list[Message], tools: Optional[List[dict]], **kwargs):
        res = ollama.chat(
            model=self.model,
            messages=context,
            tools=tools,
            **kwargs,
        )
        
        print(res)

        if tools:
            return [
                ToolCall(
                    id=uuid4(),
                    function=FunctionCall(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                    type="function",
                )
                for tool_call in res["message"]["tool_calls"]
            ]

        return res["message"]["content"]
