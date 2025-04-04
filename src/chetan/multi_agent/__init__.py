from chetan.core.loop import AgentLoop


class HomologousAgents:
    def __init__(self, agent_arch: AgentLoop, config: list[dict]):
        self.agent_arch = agent_arch
        self.config = config
