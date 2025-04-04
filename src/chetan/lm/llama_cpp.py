from typing import List, Optional, Type
from pydantic import BaseModel
from chetan.core.types import FunctionCall, ToolCall
from chetan.lm import LanguageModel, RawMessage
from openai.types.chat.chat_completion import ChatCompletion
import instructor
from llama_cpp import Llama

from chetan.utils import primitive_base_model


class LlamaCppLM(LanguageModel):
    client: Llama
    ins_client: instructor.Instructor

    def __init__(self, model: Llama, *args, **kwargs):
        self.model = model
        self.ins_client = instructor.patch(
            *args,
            create=self.model.create_chat_completion_openai_v1,
            mode=instructor.Mode.JSON_SCHEMA,
            **kwargs,
        )

    def _generate_structured(
        self,
        context: list[RawMessage],
        target_model: Type[BaseModel],
        iterable: bool,
        *args,
        **kwargs,
    ):

        t = target_model

        if iterable:
            t = primitive_base_model(List[target_model])

        print(t)

        res = self.ins_client(
            response_model=t,
            messages=context,
            *args,
            **kwargs,
        )
        return res.value

    def _generate_raw(
        self, context: list[RawMessage], *args, tools: Optional[List[dict]], **kwargs
    ):
        res: ChatCompletion = self.model.create_chat_completion_openai_v1(
            messages=context,
            tools=tools,
        )

        if tools:
            return [
                ToolCall(
                    id=tool_call.id,
                    function=FunctionCall(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                    type="function",
                )
                for tool_call in res.choices[0].message.tool_calls
            ]

        return res.choices[0].message.content
