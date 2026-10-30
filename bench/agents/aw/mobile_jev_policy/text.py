"""Adapted from mobile-jev scripts/mobile-agent/text.mjs @395fc22."""

import re
from typing import Any, Dict, List

_LEAD = re.compile(r"^[\"'“‘(\[{]+")
_TRAIL = re.compile(r"[\"'”’)\]},.!?;:]+\Z")
# JS \s (ECMAScript WhiteSpace + LineTerminator), so \S matches exactly what the original does.
JS_SPACE = "\t\n\u000b\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_JS_NON_SPACE = re.compile("[^%s]+" % JS_SPACE)


def js_trim(value: str) -> str:
    return value.strip(JS_SPACE)


def text_candidates(goal: str, supplied: List[str] = ()) -> Dict[str, Any]:
    """Jev selects source text; it never invents a field value. Explicit values take precedence."""
    if supplied:
        return {"values": list(dict.fromkeys(supplied)), "source": "supplied", "overflow": False}
    words = list(_JS_NON_SPACE.finditer(goal))
    values: Dict[str, None] = {}
    for length in range(1, min(8, len(words)) + 1):
        for start in range(0, len(words) - length + 1):
            end = words[start + length - 1]
            value = goal[words[start].start() : end.end()]
            value = js_trim(_TRAIL.sub("", _LEAD.sub("", value)))
            if value:
                values[value] = None
            if len(values) > 254:
                return {"values": [], "source": "goal", "overflow": True}
    return {"values": list(values), "source": "goal", "overflow": False}


def text_candidates_capped(goal: str, priority_text: str = "") -> Dict[str, Any]:
    """bench extension (not upstream): spans of `priority_text` (e.g. the low-level step
    instruction) first, then the goal's, capped at 254 instead of dropping everything."""
    values: Dict[str, None] = {}
    overflow = False
    for source in (priority_text, goal):
        words = list(_JS_NON_SPACE.finditer(source))
        for length in range(1, min(8, len(words)) + 1):
            for start in range(0, len(words) - length + 1):
                end = words[start + length - 1]
                value = js_trim(_TRAIL.sub("", _LEAD.sub("", source[words[start].start() : end.end()])))
                if value and value not in values:
                    if len(values) >= 254:
                        overflow = True
                        continue
                    values[value] = None
    return {"values": list(values), "source": "goal", "overflow": overflow}
