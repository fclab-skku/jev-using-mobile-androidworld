"""Decode android_env AndroidAccessibilityForest protos into nested JSON trees.

Field numbers follow google-deepmind/android_env android_env/proto/a11y/*.proto.
Nodes arrive as a flat list linked by child_ids; we rebuild the hierarchy per window.
"""

from typing import Any, Dict, List

from .tfrecord import iter_fields, packed_varints

WINDOW_TYPES = {
    0: "UNKNOWN",
    1: "APPLICATION",
    2: "INPUT_METHOD",
    3: "SYSTEM",
    4: "ACCESSIBILITY_OVERLAY",
    5: "SPLIT_SCREEN_DIVIDER",
    6: "MAGNIFICATION_OVERLAY",
    7: "WINDOW_CONTROL",
}

NODE_STRINGS = {
    3: "class_name",
    4: "content_description",
    5: "hint_text",
    6: "package_name",
    7: "text",
    10: "view_id_resource_name",
    31: "tooltip_text",
}
NODE_BOOLS = {
    12: "is_checkable",
    13: "is_checked",
    14: "is_clickable",
    15: "is_editable",
    16: "is_enabled",
    17: "is_focusable",
    18: "is_focused",
    19: "is_long_clickable",
    20: "is_password",
    21: "is_scrollable",
    22: "is_selected",
    23: "is_visible_to_user",
}


def _int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def _rect(buf: bytes) -> List[int]:
    rect = {1: 0, 2: 0, 3: 0, 4: 0}  # left, top, right, bottom
    for field, _, value in iter_fields(buf):
        if field in rect:
            rect[field] = _int32(value)
    return [rect[1], rect[2], rect[3], rect[4]]


def _node(buf: bytes) -> Dict[str, Any]:
    node: Dict[str, Any] = {"unique_id": 0, "bounds": [0, 0, 0, 0], "child_ids": []}
    node.update({name: "" for name in NODE_STRINGS.values()})
    node.update({name: False for name in NODE_BOOLS.values()})
    for field, wire, value in iter_fields(buf):
        if field == 1:
            node["unique_id"] = _int32(value)
        elif field == 2:
            node["bounds"] = _rect(value)
        elif field in NODE_STRINGS:
            node[NODE_STRINGS[field]] = value.decode("utf-8", "replace")
        elif field in NODE_BOOLS:
            node[NODE_BOOLS[field]] = bool(value)
        elif field == 25:
            node["child_ids"].extend(packed_varints(value) if wire == 2 else [value])
        elif field == 27:
            node["depth"] = _int32(value)
    return node


def _nest(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {n["unique_id"]: n for n in nodes}
    referenced = set()
    for n in nodes:
        n["children"] = [by_id[c] for c in n["child_ids"] if c in by_id]
        referenced.update(c for c in n["child_ids"] if c in by_id)
    for n in nodes:
        del n["child_ids"]
    return [n for n in nodes if n["unique_id"] not in referenced]


def decode_forest(buf: bytes) -> List[Dict[str, Any]]:
    """AndroidAccessibilityForest -> list of windows, each with nested `roots`."""
    windows = []
    for field, _, wbuf in iter_fields(buf):
        if field != 1:
            continue
        window: Dict[str, Any] = {
            "id": 0,
            "type": "UNKNOWN",
            "title": "",
            "layer": 0,
            "bounds": [0, 0, 0, 0],
            "is_active": False,
            "is_focused": False,
            "roots": [],
        }
        nodes: List[Dict[str, Any]] = []
        for f, _, v in iter_fields(wbuf):
            if f == 1:
                window["bounds"] = _rect(v)
            elif f == 3:
                window["id"] = _int32(v)
            elif f == 4:
                window["layer"] = _int32(v)
            elif f == 5:
                window["title"] = v.decode("utf-8", "replace")
            elif f == 6:
                window["type"] = WINDOW_TYPES.get(v, "UNKNOWN")
            elif f == 8:
                window["is_active"] = bool(v)
            elif f == 9:
                window["is_focused"] = bool(v)
            elif f == 11:
                nodes = [_node(nb) for tf, _, nb in iter_fields(v) if tf == 1]
        window["roots"] = _nest(nodes)
        windows.append(window)
    return windows
