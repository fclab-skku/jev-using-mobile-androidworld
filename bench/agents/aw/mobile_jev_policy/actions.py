"""Adapted from mobile-jev scripts/mobile-agent/actions.mjs @395fc22."""

import math
from typing import Any, Dict, List

from .state import js_json


def _area(b: Dict[str, int]) -> int:
    return (b["right"] - b["left"]) * (b["bottom"] - b["top"])


def candidates_for(observation: Dict[str, Any], texts: List[str] = ()) -> Dict[str, Dict[str, Any]]:
    actions: Dict[str, Dict[str, Any]] = {}
    for node in observation["elements"]:
        if node["enabled"] and (node["clickable"] or node["editable"]):
            actions["tap_%s" % node["id"]] = {"type": "tap-element", "elementId": node["id"]}
    actions["back"] = {"type": "global", "name": "back"}
    actions["home"] = {"type": "global", "name": "home"}
    # Derive gestures from each scrollable region instead of assuming a full-screen list.
    scroll_bounds = set()
    scrollable = [e for e in observation["elements"] if e["enabled"] and e["scrollable"]]
    for node in scrollable:
        # Prefer the list when it occupies most of the container, retaining small nested panes.
        if any(child["id"].startswith(node["id"] + ".") and _area(child["bounds"]) >= _area(node["bounds"]) * 0.7 for child in scrollable):
            continue
        bounds_key = js_json(node["bounds"])
        if bounds_key in scroll_bounds:
            continue
        scroll_bounds.add(bounds_key)
        b = node["bounds"]
        left, top, right, bottom = b["left"], b["top"], b["right"], b["bottom"]
        x, y = math.floor((left + right) / 2), math.floor((top + bottom) / 2)
        x1, x2 = math.floor(left + (right - left) * 0.2), math.floor(left + (right - left) * 0.8)
        y1, y2 = math.floor(top + (bottom - top) * 0.2), math.floor(top + (bottom - top) * 0.8)
        for name, coords in (("down", (x, y2, x, y1)), ("up", (x, y1, x, y2)), ("right", (x2, y, x1, y)), ("left", (x1, y, x2, y))):
            actions["scroll_%s_%s" % (name, node["id"])] = {
                "type": "swipe",
                "startX": coords[0],
                "startY": coords[1],
                "endX": coords[2],
                "endY": coords[3],
                "duration": 300,
                "regionId": node["id"],
            }
    if observation["phone"]["isEditable"]:
        actions["enter"] = {"type": "key", "key": "enter"}
        input_id = observation["phone"].get("inputElementId")
        if not any(node["password"] and (node["focused"] or node["id"] == input_id) for node in observation["elements"]):
            for index, text in enumerate(texts):
                actions["text_%d" % index] = {"type": "type", "text": text, "clear": True}
    return actions


def describe_action(action: Dict[str, Any], observation: Dict[str, Any]) -> str:
    if action["type"] == "open-app":
        return "Open %s" % (action.get("appLabel") or action.get("packageName"))
    if action["type"] == "tap-element":
        target = next((e for e in observation["elements"] if e["id"] == action["elementId"]), None)
        if target and target["editable"]:
            return "Focus text input: %s." % (target["hint"] or target["label"] or target["text"] or "empty input field")
        labels: List[str] = []
        for e in observation["elements"]:
            if e["id"] == action["elementId"] or e["id"].startswith(action["elementId"] + "."):
                labels.extend(v for v in (e["text"], e["label"]) if v)
        unique = list(dict.fromkeys(labels))
        return "Tap %s." % (" / ".join(unique) or action["elementId"])
    if action["type"] == "swipe":
        if action["startY"] > action["endY"]:
            direction = "down"
        elif action["startY"] < action["endY"]:
            direction = "up"
        elif action["startX"] > action["endX"]:
            direction = "right"
        else:
            direction = "left"
        return "Scroll %s to reveal content further %s in this scrollable region. Gesture: %s" % (direction, direction, js_json(action))
    return js_json(action)
