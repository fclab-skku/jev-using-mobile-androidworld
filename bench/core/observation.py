"""Observation helpers: window selection, tree walking, element lookup, instruction modes."""

from typing import Any, Dict, Iterator, List, Optional, Tuple

WINDOW_MODES = ("no_statusbar", "no_ime_statusbar", "app_only", "all")
DEFAULT_WINDOW_MODE = "no_statusbar"  # AndroidControl's view: app windows + keyboard when shown
# Keyboard keys that submit/confirm the focused field (AndroidControl records these as clicks).
CONFIRM_KEY_LABELS = ("search", "go", "done", "next", "enter", "send", "return", "ok")
INSTRUCTION_MODES = ("high", "high+low")
DEFAULT_GOAL_TEMPLATES = {
    "high": "{goal}",
    "high+low": "{goal}\nCurrent step: {step_instruction}",
}


def is_status_bar(window: Dict[str, Any], screen: List[int]) -> bool:
    top, bottom = window["bounds"][1], window["bounds"][3]
    return window["type"] == "SYSTEM" and top <= 0 and bottom <= 0.1 * screen[1]


def select_windows(windows: List[Dict[str, Any]], mode: str, screen: List[int]) -> List[Dict[str, Any]]:
    """Filter windows by mode and order them topmost (highest layer) first."""
    if mode == "all":
        kept = list(windows)
    elif mode == "app_only":
        kept = [w for w in windows if w["type"] == "APPLICATION"]
    elif mode == "no_statusbar":
        kept = [w for w in windows if not is_status_bar(w, screen)]
    elif mode == "no_ime_statusbar":
        kept = [w for w in windows if w["type"] != "INPUT_METHOD" and not is_status_bar(w, screen)]
    else:
        raise ValueError("Unknown window mode %r; choose from %s" % (mode, WINDOW_MODES))
    return sorted(kept, key=lambda w: -w["layer"])


def keyboard_windows(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [w for w in windows if w["type"] == "INPUT_METHOD"]


def confirm_key(ime_windows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The keyboard's action key (Search/Go/Done/Next/Enter/...), if a keyboard is shown."""
    keys = [n for n in walk(roots(ime_windows)) if n.get("is_clickable") and area(n["bounds"]) > 0]
    for wanted in CONFIRM_KEY_LABELS:
        for node in keys:
            label = ((node.get("content_description") or "") or (node.get("text") or "")).strip().lower()
            if label == wanted:
                return node
    for node in keys:  # Gboard-style resource ids when the key is unlabeled
        rid = (node.get("view_id_resource_name") or "").lower()
        if "ime_action" in rid or rid.endswith("/key_enter") or "enter" in rid.rsplit("/", 1)[-1]:
            return node
    return None


def roots(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [root for window in windows for root in window["roots"]]


def walk(nodes: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for node in nodes:
        yield node
        yield from walk(node["children"])


def contains(bounds: List[int], x: float, y: float) -> bool:
    return bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]


def area(bounds: List[int]) -> int:
    return max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])


def node_label(node: Dict[str, Any], descend: bool = True) -> str:
    for key in ("text", "content_description", "hint_text", "tooltip_text"):
        value = (node.get(key) or "").strip()
        if value:
            return value[:80]
    if descend:
        for child in walk(node["children"]):
            label = node_label(child, descend=False)
            if label:
                return label
    rid = node.get("view_id_resource_name") or ""
    if "id/" in rid:
        return rid.rsplit("id/", 1)[-1]
    return (node.get("class_name") or "View").rsplit(".", 1)[-1]


def element_at(windows: List[Dict[str, Any]], x: float, y: float) -> Optional[Dict[str, Any]]:
    """The element a tap at (x, y) most plausibly hit: smallest clickable/editable node,
    else the smallest node, containing the point, searching topmost windows first."""
    for window in windows:
        hits = [n for n in walk(window["roots"]) if n.get("is_visible_to_user", True) and contains(n["bounds"], x, y) and area(n["bounds"]) > 0]
        if not hits:
            continue
        actionable = [n for n in hits if n.get("is_clickable") or n.get("is_editable") or n.get("is_long_clickable")]
        return min(actionable or hits, key=lambda n: area(n["bounds"]))
    return None


def compose_goal(goal: str, step_instruction: str, mode: str, templates: Optional[Dict[str, str]] = None) -> str:
    if mode not in INSTRUCTION_MODES:
        raise ValueError("Unknown instruction mode %r; choose from %s" % (mode, INSTRUCTION_MODES))
    template = (templates or DEFAULT_GOAL_TEMPLATES)[mode]
    return template.format(goal=goal.strip(), step_instruction=step_instruction.strip())


def describe_gt(action: Dict[str, Any], label: str = "") -> Tuple[str, str]:
    """(operation, readable label) for a ground-truth action, used in synthesized history."""
    kind = action["action_type"]
    if kind in ("click", "long_press"):
        return kind, label or "(%s, %s)" % (action.get("x"), action.get("y"))
    if kind == "scroll":
        return kind, action.get("direction", "")
    if kind == "input_text":
        return kind, action.get("text", "")
    if kind == "open_app":
        return kind, action.get("app_name", "")
    return kind, ""
