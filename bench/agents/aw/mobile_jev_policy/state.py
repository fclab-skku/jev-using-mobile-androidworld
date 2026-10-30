"""Adapted from summarizeState() from mobile-jev scripts/mobile-agent/device.mjs @395fc22."""

import hashlib
import json
import math
import time
from typing import Any, Dict, Optional

EDITABLE_CLASSES = (
    "android.widget.EditText",
    "android.widget.AutoCompleteTextView",
    "android.widget.MultiAutoCompleteTextView",
)


def _is_safe_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and abs(value) <= 2**53 - 1


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def js_json(value: Any) -> str:
    """JSON.stringify equivalent for plain data (compact, non-ASCII kept)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def summarize_state(raw: Dict[str, Any], device_id: Optional[str] = None) -> Dict[str, Any]:
    screen = ((raw or {}).get("device_context") or {}).get("screen_bounds")
    if (
        not screen
        or not _is_safe_int(screen.get("width"))
        or not _is_safe_int(screen.get("height"))
        or screen["width"] < 1
        or screen["height"] < 1
        or not raw.get("phone_state")
        or not raw.get("a11y_tree")
    ):
        raise ValueError("UI state is missing its tree, phone state, or valid screen bounds.")
    elements = []

    def visit(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        b = node.get("boundsInScreen")
        bounds = None
        if isinstance(b, dict) and all(_finite(b.get(k)) for k in ("left", "top", "right", "bottom")):
            bounds = {
                "left": max(0, b["left"]),
                "top": max(0, b["top"]),
                "right": min(screen["width"], b["right"]),
                "bottom": min(screen["height"], b["bottom"]),
            }
        if node.get("isVisibleToUser") is not False and bounds and bounds["right"] > bounds["left"] and bounds["bottom"] > bounds["top"]:
            password = node.get("isPassword") is True
            text = "[password]" if password else (node.get("text") or "")
            label = "" if password else (node.get("contentDescription") or "")
            editable = node.get("isEditable") is True or node.get("className") in EDITABLE_CLASSES
            if text or label or node.get("isClickable") or editable or node.get("isScrollable"):
                elements.append(
                    {
                        "id": path,
                        "text": text,
                        "label": label,
                        "resourceId": node.get("resourceId") or "",
                        "hint": node.get("hint") or "",
                        "bounds": bounds,
                        "clickable": node.get("isClickable") is True,
                        "editable": editable,
                        "scrollable": node.get("isScrollable") is True,
                        "enabled": node.get("isEnabled") is not False,
                        "focused": node.get("isFocused") is True,
                        "password": password,
                        "checkable": node.get("isCheckable") is True,
                        "checked": node.get("isChecked") is True,
                        "selected": node.get("isSelected") is True,
                    }
                )
        children = node.get("children")
        if isinstance(children, list):
            for i, child in enumerate(children):
                visit(child, "%s.%d" % (path, i))

    visit(raw["a11y_tree"], "ui")
    phone_state = raw["phone_state"]
    inputs = [e for e in elements if e["editable"] and e["enabled"]]
    focused_inputs = [e for e in inputs if e["focused"]]
    if len(focused_inputs) == 1:
        input_el = focused_inputs[0]
    elif phone_state.get("keyboardVisible") and len(inputs) == 1:
        input_el = inputs[0]
    else:
        input_el = None
    focused_element = phone_state.get("focusedElement") or {}
    phone = {
        "packageName": phone_state.get("packageName") or "",
        "currentApp": phone_state.get("currentApp") or "",
        "isEditable": phone_state.get("isEditable") is True or bool(input_el),
        "inputElementId": input_el["id"] if input_el else None,
        "focusEvidence": "reported"
        if phone_state.get("isEditable")
        else "focused-node"
        if input_el and input_el["focused"]
        else "single-input-with-keyboard"
        if input_el
        else "none",
        "keyboardVisible": phone_state.get("keyboardVisible") is True,
        "focusedElement": {
            "resourceId": focused_element.get("resourceId") or "",
            "className": focused_element.get("className") or "",
        },
    }
    # JSON.stringify drops undefined: inputElementId is omitted when there is no input.
    phone_json = {k: v for k, v in phone.items() if not (k == "inputElementId" and v is None)}
    content = {"deviceId": device_id, "phone": phone_json, "screen": screen, "elements": elements}
    if device_id is None:
        content_json = {k: v for k, v in content.items() if k != "deviceId"}
    else:
        content_json = content
    return {
        **content,
        "fingerprint": hashlib.sha256(js_json(content_json).encode("utf-8")).hexdigest(),
        "observedAt": int(time.time() * 1000),
    }
