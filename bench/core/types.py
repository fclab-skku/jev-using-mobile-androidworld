"""Shared types. The action space is AndroidControl's, plus agent-only terminal/extra types."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# AndroidControl ground-truth action types.
AC_ACTIONS = ("click", "long_press", "scroll", "input_text", "open_app", "navigate_back", "navigate_home", "wait")
# Predictions an agent can make that have no AndroidControl counterpart.
AGENT_ONLY_ACTIONS = ("done", "blocked", "enter", "needs_input", "invalid", "answer")


@dataclass
class HistoryItem:
    """A previous ground-truth step (teacher forcing)."""

    action: Dict[str, Any]  # raw AndroidControl action JSON
    instruction: str
    target_label: str = ""  # label of the element a GT click/long_press hit, if any
    target_bounds: Optional[List[int]] = None
    package_before: str = ""
    package_after: str = ""
    # Selected windows before/after this action, so agents can derive their own labels and
    # page-changed signals. Shared references; do not mutate.
    windows: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    next_windows: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    screen: List[int] = field(default_factory=list)


@dataclass
class StepInput:
    episode_id: int
    step_index: int
    num_steps: int
    goal: str  # high-level goal from the dataset
    step_instruction: str  # low-level instruction from the dataset
    instruction_mode: str  # "high" | "high+low"
    agent_goal: str  # goal string actually given to the agent
    windows: List[Dict[str, Any]]  # selected windows, topmost first, nested `roots`
    screen: List[int]  # [width, height]
    active_package: str
    keyboard_visible: bool  # an input-method window exists (before window filtering)
    screenshot: Optional[str]  # absolute path, for future VLM agents
    ime_windows: List[Dict[str, Any]] = field(default_factory=list, repr=False)  # keyboard windows, whatever the mode
    all_windows: List[Dict[str, Any]] = field(default_factory=list, repr=False)  # every window, topmost first (grading only)
    history: List[HistoryItem] = field(default_factory=list)
    app_vocab: List[str] = field(default_factory=list)  # all open_app names in the subset


@dataclass
class PredAction:
    type: str  # one of AC_ACTIONS or AGENT_ONLY_ACTIONS
    x: Optional[int] = None
    y: Optional[int] = None
    bounds: Optional[List[int]] = None  # target element [l, t, r, b]
    direction: Optional[str] = None
    text: Optional[str] = None
    app_name: Optional[str] = None
    raw_operation: str = ""  # the agent's own operation name
    detail: str = ""  # human-readable target description
    clear: Optional[bool] = None  # input_text only: False = type without clearing (target field unknown); None = adapter rule


@dataclass
class Prediction:
    action: PredAction
    tags: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)  # per-provider-slot usage
    raw: Dict[str, Any] = field(default_factory=dict)  # requests/responses, agent internals
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RunAborted(BaseException):
    """Raised by an agent when continuing the run is pointless (e.g. its world-model server is gone).
    A BaseException on purpose: AndroidWorld's suite runner catches Exception per task and moves on to the next one,
    which would spend the remaining tasks without the service. The aw-run runner records the reason and stops;
    `aw-run --run-id <id>` resumes (finished tasks are skipped by AndroidWorld's checkpoint)."""

