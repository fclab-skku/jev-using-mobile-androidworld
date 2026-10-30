"""droidrun/mobile-jev, offline, via the Python port in mobile_jev_policy/.

Per step: AndroidControl windows -> Mobilerun ui-state -> summarize_state() -> TypeSafePolicy.decide()
(request sent to the registered decision provider) -> AndroidControl action.
"""

import re
import threading
from typing import Any, Dict, List, Optional

from ...core.types import HistoryItem, PredAction, Prediction, StepInput
from ..base import AgentWrapper
from .generation import sum_usage
from .convert import to_mobilerun_state
from .mobile_jev_policy.actions import candidates_for, describe_action
from .generation import with_invalid_response_retry
from .mobile_jev_policy.policy import TypeSafeChoiceError, TypeSafePolicy
from .mobile_jev_policy.state import summarize_state


def synthetic_package(app_name: str) -> str:
    return "androidcontrol.app." + (re.sub(r"[^0-9a-z]+", "_", app_name.lower()).strip("_") or "unknown")


def _tap_target(observation: Dict[str, Any], x: float, y: float) -> Optional[Dict[str, Any]]:
    hits = [
        e
        for e in observation["elements"]
        if e["enabled"] and (e["clickable"] or e["editable"])
        and e["bounds"]["left"] <= x <= e["bounds"]["right"]
        and e["bounds"]["top"] <= y <= e["bounds"]["bottom"]
    ]
    area = lambda e: (e["bounds"]["right"] - e["bounds"]["left"]) * (e["bounds"]["bottom"] - e["bounds"]["top"])
    return min(hits, key=area) if hits else None


def _path_flags(tree: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    """summarizeState element id (path) -> flags summarizeState does not keep."""
    out: Dict[str, Dict[str, bool]] = {}

    def visit(node, path):
        if not isinstance(node, dict):
            return
        cls = str(node.get("className") or "")
        out[path] = {
            "long": node.get("isLongClickable") is True,
            "visible": node.get("benchVisible", True) is not False,
            "cls": cls.rsplit(".", 1)[-1] if cls else "",
        }
        for i, child in enumerate(node.get("children") or []):
            visit(child, "%s.%d" % (path, i))

    visit(tree, "ui")
    return out


def _bounds_list(b: Dict[str, int]) -> List[int]:
    return [b["left"], b["top"], b["right"], b["bottom"]]


class MobileJevAgent(AgentWrapper):
    required_providers = {"decision": "systemone"}
    action_space = {"click", "input_text", "scroll", "open_app", "navigate_back", "navigate_home", "wait"}

    def __init__(self, name, providers, params):
        super().__init__(name, providers, params)
        self.apps: List[Dict[str, str]] = []
        self._cache: Dict[Any, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._local = threading.local()

    def setup(self, dataset_meta: Dict[str, Any]) -> None:
        source = self.params.get("apps_source", "dataset_vocab")
        if source == "dataset_vocab":
            self.apps = [{"packageName": synthetic_package(n), "label": n} for n in dataset_meta.get("app_vocab", [])]
        elif source == "none":
            self.apps = []
        else:
            raise ValueError("apps_source must be dataset_vocab or none")

    def current_calls(self) -> List[Dict[str, Any]]:
        return list(getattr(self._local, "calls", None) or [])

    def observe(self, windows, screen, package: str, keyboard: bool) -> Dict[str, Any]:
        key = (id(windows), keyboard, package)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and hit["_windows"] is windows:
                return hit["observation"]
        raw = to_mobilerun_state(windows, keyboard, screen, package, self.params.get("invisible_nodes", "drop"), bool(self.params.get("row_checked_from_child", False)))
        observation = summarize_state(raw)
        observation["_paths"] = _path_flags(raw["a11y_tree"])
        with self._lock:
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[key] = {"_windows": windows, "observation": observation}
        return observation

    def _history(self, items: List[HistoryItem]) -> List[Dict[str, Any]]:
        """Ground-truth steps in the shape runAgent() pushes (fields decide() reads)."""
        out = []
        for item in items:
            action = item.action
            kind = action["action_type"]
            before = self.observe(item.windows, item.screen, item.package_before, False)
            after = self.observe(item.next_windows, item.screen, item.package_after, False)
            entry: Dict[str, Any] = {"screenChanged": before["fingerprint"] != after["fingerprint"]}
            if kind in ("click", "long_press"):
                target = _tap_target(before, action["x"], action["y"])
                label = self._describe_tap(target, before) if target else "Tap (%d, %d)." % (action["x"], action["y"])
                if kind == "long_press":
                    label = "Long press " + label[4:] if label.startswith("Tap ") else "Long press: " + label
                entry.update(operation="TAP" if kind == "click" else "LONG_PRESS", label=label)
            elif kind == "input_text" and "x" in action:  # AndroidWorld: typed into the field at (x, y)
                target = _tap_target(before, action["x"], action["y"])
                name = (target.get("hint") or target.get("label") or target.get("text") or target.get("resourceId", "").split("/")[-1]) if target else ""
                entry.update(operation="TYPE_TEXT", label="Type into %s." % (name or "the input field"), text=action.get("text", ""))
            elif kind == "input_text":
                typed = {"type": "type", "text": action.get("text", ""), "clear": True}
                entry.update(operation="TYPE_TEXT", label=describe_action(typed, before), text=typed["text"])
            elif kind == "scroll":
                direction = action.get("direction", "down")
                swipe = next((a for cid, a in candidates_for(before).items() if cid.startswith("scroll_%s_" % direction)), None)
                label = describe_action(swipe, before) if swipe else "Scroll %s." % direction
                entry.update(operation="SCROLL_%s" % direction.upper(), label=label)
            elif kind == "keyboard_enter":  # AndroidWorld keyboard_enter = upstream ENTER control
                entry.update(operation="ENTER", label=describe_action({"type": "key", "key": "enter"}, before))
            elif kind in ("navigate_back", "navigate_home"):
                name = "back" if kind == "navigate_back" else "home"
                entry.update(operation=name.upper(), label=describe_action({"type": "global", "name": name}, before))
            elif kind == "wait":
                entry.update(operation="WAIT", label="Wait for screen update")
            elif kind == "open_app":
                entry.update(operation="OPEN_APP", label="Open %s" % action.get("app_name", ""))
            else:
                entry.update(operation=kind.upper(), label=kind)
            out.append(entry)
        return out

    def _describe_tap(self, target: Dict[str, Any], observation: Dict[str, Any]) -> str:
        """History label for a tapped element (upstream describeAction)."""
        return describe_action({"type": "tap-element", "elementId": target["id"]}, observation)

    def _request(self, body: Dict[str, Any]) -> Dict[str, Any]:
        response, meta = self.providers["decision"].post(body)
        self._local.calls.append({"slot": "decision", "request": body, "response": response, "meta": meta})
        return response

    def predict(self, step: StepInput) -> Prediction:
        self._local.calls = []
        observation = self.observe(step.windows, step.screen, step.active_package, step.keyboard_visible)
        history = self._history(step.history)
        policy = TypeSafePolicy(request=self._request, model=self.providers["decision"].model, threshold=float(self.params.get("threshold", 0)))
        decision, retries = with_invalid_response_retry(
            lambda: policy.decide(step.agent_goal, observation, history, [], self.apps),
            lambda e: isinstance(e, TypeSafeChoiceError),
            int(self.params.get("invalid_response_retries", 2)),
        )
        tags = ["invalid_response_retry"] if retries else []
        if decision.pop("_text_overflow"):
            tags.append("text_candidates_overflow")
        if self.apps:
            tags.append("app_list")  # an installed-app list was offered (offline name: synthetic_apps)
        decision.pop("_request")
        decision.pop("_trimmed")
        pred = self._map(decision, observation)
        calls = self._local.calls
        offered = [_bounds_list(e["bounds"]) for e in observation["elements"] if e["enabled"] and (e["clickable"] or e["editable"])]
        return Prediction(
            action=pred,
            tags=tags,
            latency_ms=decision["latencyMs"],
            usage=sum_usage(calls),
            raw={"decision": decision, "calls": calls, "n_elements": len(observation["elements"]), "offered_bounds": offered, "invalid_response_retries": retries},
        )

    def _map(self, decision: Dict[str, Any], observation: Dict[str, Any]) -> PredAction:
        status, op = decision["status"], decision["operation"]
        if status in ("done", "blocked", "needs_input"):
            return PredAction(type=status, raw_operation=op, detail=decision.get("reason", ""))
        if status != "action":
            return PredAction(type="invalid", raw_operation=op, detail="status=%s" % status)
        action = decision["action"]
        kind = action["type"]
        if kind == "tap-element":
            element = next(e for e in observation["elements"] if e["id"] == action["elementId"])
            b = element["bounds"]
            return PredAction(
                type="click",
                x=(b["left"] + b["right"]) // 2,
                y=(b["top"] + b["bottom"]) // 2,
                bounds=_bounds_list(b),
                raw_operation=op,
                detail=decision.get("label", ""),
            )
        if kind == "type":
            return PredAction(type="input_text", text=action["text"], raw_operation=op)
        if kind == "swipe":
            direction = op.replace("SCROLL_", "").lower()
            # scroll_region=element: the swipe runs inside the region Jev chose (upstream swipes inside it too);
            # screen: only the direction is passed and the adapter swipes from the screen centre (AW-11).
            region = next((e for e in observation["elements"] if e["id"] == action.get("regionId")), None)
            bounds = _bounds_list(region["bounds"]) if region and self.params.get("scroll_region", "screen") == "element" else None
            return PredAction(type="scroll", direction=direction, bounds=bounds, raw_operation=op, detail=decision.get("label", ""))
        if kind == "global":
            return PredAction(type="navigate_back" if action["name"] == "back" else "navigate_home", raw_operation=op)
        if kind == "key":
            return PredAction(type="enter", raw_operation=op)
        if kind == "open-app":
            return PredAction(type="open_app", app_name=action.get("appLabel"), raw_operation=op)
        if kind == "wait":
            return PredAction(type="wait", raw_operation=op)
        return PredAction(type="invalid", raw_operation=op, detail="unmapped action %s" % kind)
