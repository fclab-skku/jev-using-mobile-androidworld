"""AndroidControl windows -> the live-device tree formats each agent consumes.

invisible="drop" (original agents): nodes with is_visible_to_user=false are dropped and their
children lifted, matching live dumps (uiautomator/Portal report only visible nodes).
invisible="keep": such nodes stay (off-screen ones are still removed by each agent's bounds checks).
prune_empty=True: structural nodes that carry nothing an agent reads (no text/description, not
clickable/long-clickable/editable/scrollable/focused, not an edit field) are removed and their
children lifted. Relative order of the remaining nodes is unchanged.
"""

from typing import Any, Dict, List

from ...core.observation import walk


INVISIBLE_MODES = ("drop", "keep")
EDIT_CLASS_HINTS = ("EditText", "AutoCompleteTextView")


def is_empty_structure(node: Dict[str, Any]) -> bool:
    return not (
        (node.get("text") or "").strip()
        or (node.get("content_description") or "").strip()
        or node.get("is_clickable")
        or node.get("is_long_clickable")
        or node.get("is_editable")
        or node.get("is_scrollable")
        or node.get("is_focused")
        or any(h in (node.get("class_name") or "") for h in EDIT_CLASS_HINTS)
    )


def _kept_children(nodes: List[Dict[str, Any]], convert, invisible: str = "drop", prune_empty: bool = False) -> List[Dict[str, Any]]:
    if invisible not in INVISIBLE_MODES:
        raise ValueError("invisible must be one of %s" % (INVISIBLE_MODES,))
    out: List[Dict[str, Any]] = []
    for node in nodes:
        dropped = (invisible == "drop" and not node.get("is_visible_to_user", True)) or (prune_empty and is_empty_structure(node))
        if dropped:
            out.extend(_kept_children(node["children"], convert, invisible, prune_empty))
        else:
            out.append(convert(node))
    return out


def child_text(node: Dict[str, Any]) -> str:
    """Visible descendants' text and content descriptions, de-duplicated in tree order, joined by ' / '."""
    out: List[str] = []
    for child in walk(node["children"]):
        if not child.get("is_visible_to_user", True):
            continue
        for value in (child.get("text"), child.get("content_description")):
            value = (value or "").strip()
            if value and value not in out:
                out.append(value)
    return " / ".join(out)


def to_portal(windows: List[Dict[str, Any]], invisible: str = "drop", prune_empty: bool = False, with_child_text: bool = False) -> List[Dict[str, Any]]:
    """Droidrun/Mobilerun Portal-style nested dicts (jev-mobile's a11y.snapshot_state input).
    with_child_text adds `childText` (not read by upstream; used by jev_mobile_ac labels)."""

    def convert(node: Dict[str, Any]) -> Dict[str, Any]:
        extra = {"childText": child_text(node)} if with_child_text else {}
        return {
            **extra,
            "className": node["class_name"],
            "text": node["text"],
            "contentDescription": node["content_description"],
            "resourceId": node["view_id_resource_name"],
            "hint": node["hint_text"],
            "bounds": list(node["bounds"]),
            "clickable": node["is_clickable"],
            "longClickable": node["is_long_clickable"],
            "editable": node["is_editable"],
            "enabled": node["is_enabled"],
            "scrollable": node["is_scrollable"],
            "focused": node["is_focused"],
            "checkable": node["is_checkable"],
            "checked": node["is_checked"],
            "selected": node["is_selected"],
            "password": node["is_password"],
            "packageName": node["package_name"],
            "visibleToUser": node["is_visible_to_user"],  # not read by upstream; used by the choice guard
            "children": _kept_children(node["children"], convert, invisible, prune_empty),
        }

    return _kept_children([r for w in windows for r in w["roots"]], convert, invisible, prune_empty)


def child_checked_states(nodes: List[Dict[str, Any]]) -> Dict[int, bool]:
    """id(row) -> checked, for a clickable, non-checkable row that owns exactly one checkable widget.

    Android settings rows are a clickable container (text "Wi-Fi") holding a non-clickable Switch. mobile-jev keeps only
    elements with text / click / edit / scroll, so the Switch is dropped and the row carries no on/off state
    (docs/EXTENSIONS.md AW-9). A widget belongs to its nearest clickable ancestor; a row with zero or several
    such widgets is left alone, and a checkable widget that is clickable itself stays its own element."""
    owned: Dict[int, List[bool]] = {}

    def visit(node: Dict[str, Any], owner: Any) -> None:
        if node.get("is_checkable") and not node.get("is_clickable") and owner is not None and not owner.get("is_checkable"):
            owned.setdefault(id(owner), []).append(bool(node.get("is_checked")))
        here = node if node.get("is_clickable") else owner
        for child in node.get("children") or []:
            visit(child, here)

    for root in nodes:
        visit(root, None)
    return {key: states[0] for key, states in owned.items() if len(states) == 1}


def to_mobilerun_state(windows: List[Dict[str, Any]], keyboard_visible: bool, screen: List[int], package: str, invisible: str = "drop", lift_checked: bool = False) -> Dict[str, Any]:
    """Mobilerun `/ui-state` shape consumed by mobile-jev's summarizeState.
    lift_checked: a row that owns one checkable widget reports that widget's state (see child_checked_states)."""
    lifted = child_checked_states([r for w in windows for r in w["roots"]]) if lift_checked else {}

    def convert(node: Dict[str, Any]) -> Dict[str, Any]:
        left, top, right, bottom = node["bounds"]
        return {
            "className": node["class_name"],
            "text": node["text"],
            "contentDescription": node["content_description"],
            "resourceId": node["view_id_resource_name"],
            "hint": node["hint_text"],
            "boundsInScreen": {"left": left, "top": top, "right": right, "bottom": bottom},
            "isVisibleToUser": True,
            "isClickable": node["is_clickable"],
            "isEditable": node["is_editable"],
            "isEnabled": node["is_enabled"],
            "isScrollable": node["is_scrollable"],
            "isFocused": node["is_focused"],
            "isPassword": node["is_password"],
            "isCheckable": node["is_checkable"] or id(node) in lifted,
            "isChecked": lifted.get(id(node), node["is_checked"]),
            "isSelected": node["is_selected"],
            # extra keys, not read by summarizeState; used by the mobile_jev_ac extensions
            "isLongClickable": node["is_long_clickable"],
            "benchVisible": node["is_visible_to_user"],
            "children": _kept_children(node["children"], convert, invisible),
        }

    nodes = [r for w in windows for r in w["roots"]]
    focused = next((n for n in walk(nodes) if n.get("is_focused") and n.get("is_editable")), None)
    return {
        "a11y_tree": {"className": "bench.WindowStack", "children": _kept_children(nodes, convert, invisible)},
        "phone_state": {
            "packageName": package,
            "currentApp": "",
            "isEditable": focused is not None,
            "keyboardVisible": keyboard_visible,
            "focusedElement": {
                "resourceId": focused["view_id_resource_name"] if focused else "",
                "className": focused["class_name"] if focused else "",
            },
        },
        "device_context": {"screen_bounds": {"width": int(screen[0]), "height": int(screen[1])}},
    }
