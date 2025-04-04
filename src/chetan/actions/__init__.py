from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Coroutine, Literal, Optional, Type, Union, Any
from pydantic import BaseModel, SerializeAsAny, ConfigDict, ValidationError
from chetan.core.context import Message
from chetan.core.tool import Tool
import json
import yaml


class Action(BaseModel):
    name: str
    description: str
    args: SerializeAsAny[Type[BaseModel]]
    output: SerializeAsAny[Optional[Type[BaseModel]]]
    metadata: dict = {}
    fn: Callable

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_dump_json(self, **kwargs) -> str:
        return json.dumps(
            {
                "name": self.name,
                "description": self.description,
                "args": self.args.__name__,
                "output": self.output.__name__,
                "metadata": self.metadata,
            }
        )


class ActionGroup(BaseModel):
    name: str
    actions: list[Union[Action, "ActionGroup"]]

    def model_dump_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict:
        """Convert ActionGroup to a properly nested dictionary structure"""
        return {
            "name": self.name,
            "actions": [
                (
                    action.to_dict()
                    if isinstance(action, ActionGroup)
                    else json.loads(action.model_dump_json())
                )
                for action in self.actions
            ],
        }

    def traverse(self, path: str):
        path = path.split(".")

        # We handle the root case
        if self.name == "root" and path[0] == "root":
            path = path[1:]

        if len(path) == 0:
            return self

        # Return matching action or group at this level
        for action in self.actions:
            if action.name == path[0]:
                if isinstance(action, ActionGroup):
                    next_result = action.traverse(".".join(path[1:]))
                    return action if next_result is None else next_result
                return action

        return None

    def flatten(self):
        flattened = []
        for action in self.actions:
            if isinstance(action, Action):
                flattened.append(action)
            elif isinstance(action, ActionGroup):
                flattened.extend(action.flatten())
        return flattened

    def get_flattened_actions(self, prefix: str = "") -> list[tuple[str, Action]]:
        """Get a flat list of (qualified_name, action) tuples"""
        flattened = []
        for action in self.actions:
            name = f"{prefix}{action.name}" if prefix else action.name
            if isinstance(action, Action):
                flattened.append((name, action))
            elif isinstance(action, ActionGroup):
                flattened.extend(action.get_flattened_actions(f"{name}."))
        return flattened


class ActionSelection(BaseModel):
    actions: list[str]
    reasoning: str


class ActionInvocation(BaseModel):
    reasoning: str
    action: str


class ActionInvocationWithArgs[T](BaseModel):
    id: str
    action: str
    args: T


class ActionSystem:
    actions: ActionGroup

    def __init__(self):
        self.actions = ActionGroup(name="root", actions=[])

    async def execute(self, action: str, args: BaseModel):

        action_obj = self.actions.traverse(action)

        if not action_obj:
            return "A nonexistent action was called."

        if isinstance(action_obj, ActionGroup):
            return "A group was called, not an action."

        # Check if the invocation args are valid
        try:
            action_obj.args.model_validate(args)

            result = await action_obj.fn(args)

            return result
        except Exception as e:
            return repr(e)

    def validate(self, actions: list[str]):
        issues = []
        for action in actions:
            action_obj = self.actions.traverse(action)
            if not action_obj:
                issues.append(f"Action {action} does not exist.")
        return issues, len(issues) == 0

    def recommend(self, context: list[Message]):
        flattened_actions = self.actions.get_flattened_actions()
        actions_list = [
            f"- '{qualified_name.replace(".", "-")}': {action.description}"
            for qualified_name, action in flattened_actions
        ]
        toolified_actions = [
            (Tool.create(action, qualified_name))
            for qualified_name, action in flattened_actions
        ]
        return (
            "Recommended available actions:\n" + "\n".join(actions_list),
            toolified_actions,
        )

    async def invoke_actions(
        self,
        invocations: list[ActionInvocationWithArgs],
        method: Literal["linear", "parallel"],
        **kwargs,
    ) -> dict[Any]:

        actions = [invocation.action for invocation in invocations]
        if not self.validate(actions):
            return "Invalid actions specified"

        # print all the executed actions names in blue color
        print("\033[94m" + "Executed actions: " + str(actions) + "\033[0m")

        if method == "parallel":
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self.execute,
                        invocation.action,
                        invocation.args,
                        **kwargs,
                    )
                    for invocation in invocations
                ]
                results = dict((i.id, f.result()) for f, i in zip(futures, invocations))
            return results
        elif method == "linear":
            results = {}
            for invocation in invocations:
                results[invocation.id] = await self.execute(
                    invocation.action, invocation.args, **kwargs
                )
            return results
        else:
            raise ValueError("Invalid method specified")
