from typing import List, Optional, Type
from pydantic import BaseModel
from chetan.core.context.iteration import Message
from chetan.core.types import FunctionCall, ToolCall
from chetan.lm import LanguageModel
from openai import NotGiven, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
import instructor

class OpenAILM(LanguageModel):
    client: OpenAI
    ins_client: instructor.Instructor
    model: str

    def __init__(self, client: OpenAI, model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self.model = model
        self.ins_client = instructor.from_openai(client)

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

    def _generate_raw(
        self, context: list[Message], *args, tools: Optional[List[dict]], **kwargs
    ):
        res: ChatCompletion = self.client.chat.completions.create(
            messages=context,
            model=self.model,
            *args,
            tools=tools,
            # tool_choice=("required" if tools else NotGiven()),
            **kwargs,
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
