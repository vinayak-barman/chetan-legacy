from chetan.core.context.iteration import (
    Iteration,
    IterationItem,
    Message,
    PersistentContext,
    UserMessage,
)


class ContextManager:
    """Context manager bound to an agent loop instance"""

    # We will have a persistent context and a working context

    # The persistent context will be used to store an observable, debuggable context that can be used to analyze the agent's behavior. We generally want to keep this context in a database, to reduce memory usage

    # The working context will be used to facilitate Chat formatted messages that is directly used in the context

    # TODO: Implement smart context cleaning to remove unnecessary, irrelevant context items to reduce memory usage and improve performance

    current_iteration: int = -1

    exported: list[Message]
    context: PersistentContext  # TODO: Move persistent context to a database, to reduce memory usage

    def __init__(self):
        with open(
            "/Users/arjo/Work/snayu/oss/chetan_v0/chetan/src/chetan/core/context/system.prompt.txt"
        ) as f:
            system_message = Message(role="system", content=f.read())

            self.exported = [system_message]
        self.context = PersistentContext(iterations=[])

    def iteration(self, iteration: int):
        self.current_iteration = iteration
        self.context.items.append(Iteration())

    def add(
        self,
        item: IterationItem,
    ):
        """Add an item to the context

        Args:
            item (IterationItem): The item to add to the context
        """

        if type(item) is UserMessage:
            self.context.items.append(item)
        else:
            if type(self.context.items[-1]) is not Iteration:
                raise RuntimeError("Non-user message item must go within an iteration")

            self.context.items[-1].add(item)

        self.exported.append(item.export())
