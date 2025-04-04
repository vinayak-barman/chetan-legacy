from typing import Dict, List, Literal, Optional, Self, Type, Union
from pydantic import BaseModel, create_model
from chetan.actions import Action, ActionInvocationWithArgs, ActionSystem
from chetan.core.tool import Tool
from chetan.core.context import ContextManager
from chetan.core.context.iteration import (
    PreludeItem,
    Processing,
    Results,
    ToolCall,
    ToolCalling,
)
from chetan.lm import LanguageModel
from functools import wraps
from termcolor import colored
import json
from chetan.utils import rand_code_name_pairs, stringify


def iteration_stage(stage_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            print(colored(f"[STAGE] : {stage_name}", "green"))
            result = func(self, *args, **kwargs)
            return result

        return wrapper

    return decorator


class AgentLoopConfig(BaseModel):
    max_iterations: int = 20


class AgentLoop:
    id: str
    context: ContextManager
    action_system: ActionSystem
    lm: LanguageModel

    config: AgentLoopConfig

    def __init__(
        self,
        lm: LanguageModel,
        *,
        ctx_manager: ContextManager = ContextManager(),
        action_system: ActionSystem = ActionSystem(),
        config: AgentLoopConfig = AgentLoopConfig(),
    ):
        self.id = rand_code_name_pairs()

        self.context = ctx_manager
        self.action_system = action_system
        self.lm = lm
        self.config = config

    def generate(
        self,
        *args,
        target_model: Union[Type[BaseModel], None] = None,
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> Union[BaseModel, str]:
        return self.lm.generate(
            target_model=target_model,
            tools=tools,
            context=self.context.exported,
            *args,
            **kwargs,
        )

    iteration_wide_storage: dict = {}

    async def __call__(self, max_iter=None):
        current_iteration = 0
        self.iteration_wide_storage.clear()

        for iteration in range((max_iter or self.config.max_iterations)):
            self.context.iteration(iteration)
            await self._prelude()
            await self._processing()
            exited = await self._tool_calling()

            if exited:
                break

            await self._action_execution()
            await self._state_synthesis()

            # TODO: Implement feedback
            # self.feedback_system.active_feedback()
            current_iteration += 1
            self.iteration_wide_storage.clear()

    def current_iteration(self) -> int:
        """Get the current iteration number

        Returns:
            int: The current iteration number
        """
        return self.context.current_iteration

    @iteration_stage("Prelude")
    async def _prelude(self):
        # TODO: Implement module systems to run certain functions before the main loop starts
        recommendation, actions = self.action_system.recommend(self.context.exported)
        self.context.add(PreludeItem(content=recommendation, tag="Recommendation"))
        self.iteration_wide_storage["rec_actions"] = actions

    @iteration_stage("Processing")
    async def _processing(self):
        content: str = (
            self.generate()
        )  # We do open ended generation for maximum flexibility
        print(colored(f"[PROCESS]: {content}", "yellow"))
        self.context.add(Processing(content=content))
        pass

    @iteration_stage("Tool Calling")
    async def _tool_calling(self) -> bool:
        tool_calls: List[ToolCall] = self.generate(
            tools=self.iteration_wide_storage["rec_actions"],
        )

        for call in tool_calls:
            print(colored(f"[TOOL CALL] : {call}", "cyan"))

        for call in tool_calls:
            if call["function"]["name"] == "exit":
                return True

        self.context.add(ToolCalling(calls=tool_calls))
        self.iteration_wide_storage["tool_calls"] = tool_calls

    @iteration_stage("Action Execution")
    async def _action_execution(self):
        calls: List[ToolCall] = self.iteration_wide_storage["tool_calls"]

        invocations: List[ActionInvocationWithArgs] = [
            ActionInvocationWithArgs(
                action=action_name,
                args=self.action_system.actions.traverse(action_name).args(
                    **json.loads(call["function"]["arguments"])
                ),
                id=call["id"],
            )
            for call in calls
            if (action_name := call["function"]["name"].replace("-", "."))
        ]

        res = await self.action_system.invoke_actions(invocations, method="linear")

        for key in res.keys():
            self.context.add(Results(results=stringify(res[key]), id=key))
            print(colored(f"[RESULT] : {str(res[key])[:200]}", "magenta"))

        # TODO: Implement action execution
        pass

    @iteration_stage("State Synthesis")
    async def _state_synthesis(self):
        pass

    def use(self, *modules) -> Self:
        """
        Use modules for higher level functionalities like planning, memory, multi-agent systems etc.
        """
        return self
