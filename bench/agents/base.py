"""Agent wrapper interface. Register implementations in configs/agents.yaml.

An agent receives one StepInput (teacher-forced: history is ground truth) and returns
one Prediction in AndroidControl's action space. Steps are independent, so predict()
must be thread-safe.
"""

from typing import Any, Dict, List, Set

from ..core.types import Prediction, StepInput
from ..providers.base import Provider


class AgentWrapper:
    # slot -> provider kind, e.g. {"decision": "systemone", "text": "chat"}
    required_providers: Dict[str, str] = {}
    # AndroidControl action types this agent can ever emit (used for mismatch tags)
    action_space: Set[str] = set()

    def __init__(self, name: str, providers: Dict[str, Provider], params: Dict[str, Any]):
        self.name = name
        self.providers = providers
        self.params = params

    def setup(self, dataset_meta: Dict[str, Any]) -> None:
        """Called once per run before any predict()."""

    def predict(self, step: StepInput) -> Prediction:
        raise NotImplementedError

    def current_calls(self) -> List[Dict[str, Any]]:
        """Model calls made so far by predict() on this thread (kept when predict() raises)."""
        return []

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class": "%s:%s" % (type(self).__module__, type(self).__name__),
            "providers": {slot: p.describe() for slot, p in self.providers.items()},
            "params": self.params,
            "action_space": sorted(self.action_space),
        }
