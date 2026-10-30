"""Offline mock providers for pipeline tests. Deterministic pseudo-random valid answers; no network."""

import hashlib
import json
import random
from typing import Any, Dict, Tuple

from .base import Provider


class MockSystemOneProvider(Provider):
    kind = "systemone"
    path = "/systemone"

    @property
    def api_key(self) -> str:
        return "mock"

    def post(self, body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        seed = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        rng = random.Random(seed)
        answers = {}
        for qid, question in body["questions"].items():
            if question["type"] == "choice":
                keys = list(question["criteria"])
                weights = [rng.random() + 1e-3 for _ in keys]
                total = sum(weights)
                probs = {k: w / total for k, w in zip(keys, weights)}
                # renormalize exactly and make the choice the argmax
                choice = max(probs, key=probs.get)
                answers[qid] = {"type": "choice", "choice": choice, "probabilities": probs, "confidence": round(rng.random(), 4)}
            elif question["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": round(rng.random() * 0.6, 4), "confidence": round(rng.random(), 4)}
            else:
                levels = question.get("criteria") or [0, 1]
                answers[qid] = {"type": "score", "score": rng.random(), "probabilities": [1 / len(levels)] * len(levels), "confidence": 0.5}
        usage = {"prompt_tokens": len(json.dumps(body)) // 4, "completion_tokens": 0}
        return {"model": "mock-jev", "answers": answers, "usage": usage}, {"provider": self.name, "status": 200, "attempts": 1, "latency_ms": 0.0}


class MockChatProvider(Provider):
    kind = "chat"
    path = "/chat/completions"

    @property
    def api_key(self) -> str:
        return "mock"

    def post(self, body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        system = next((m["content"] for m in body.get("messages", []) if m["role"] == "system"), "")
        content = json.dumps({"app_name": "Mock App"} if "app_name" in system else {"text": "mock text"})
        response = {"model": "mock-chat", "choices": [{"message": {"role": "assistant", "content": content}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return response, {"provider": self.name, "status": 200, "attempts": 1, "latency_ms": 0.0}


class MockWorldModelProvider(Provider):
    """World-model answer for WM-lookahead tests: a small HTML page naming the action it saw, after a gWorld-style
    reasoning line, or HTML only when the request has a system message (Code2World)."""

    kind = "chat"
    path = "/chat/completions"

    @property
    def api_key(self) -> str:
        return "mock"

    def post(self, body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        text = next((p["text"] for m in body.get("messages", []) for p in (m["content"] if isinstance(m["content"], list) else []) if p.get("type") == "text"), "")
        action = text.split("Action:\n", 1)[-1].split("\n", 1)[0] if "Action:\n" in text else "?"
        page = (
            "<!DOCTYPE html><html><body><header><button aria-label=\"Back\">&lt;</button><h1>Next page</h1></header>"
            "<main><p>Result of %s</p><button>Save</button></main></body></html>" % action
        )
        if any(m["role"] == "system" for m in body.get("messages", [])):  # Code2World: HTML only
            content = page
        else:
            content = "# Next State Reasoning: After %s the screen shows the next page.\n# HTML: %s" % (action, page)
        response = {"model": "mock-wm", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return response, {"provider": self.name, "status": 200, "attempts": 1, "latency_ms": 0.0}


class MockRerunProvider(Provider):
    """Valid answers for mobile_jev_llm_aw tests: the first offered operation/target (free), candidate 2 (rerank),
    or a mobile_use call pressing Back (gui)."""

    kind = "chat"
    path = "/chat/completions"

    @property
    def api_key(self) -> str:
        return "mock"

    def post(self, body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        user_content = next(m["content"] for m in body["messages"] if m["role"] == "user")
        if isinstance(user_content, list):  # rerun_mode=gui (Qwen mobile_use): go back
            content = 'Thought: Mock.\nAction: Go back.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "system_button", "button": "Back"}}\n</tool_call>'
            response = {"model": "mock-rerun", "choices": [{"message": {"role": "assistant", "content": content}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            return response, {"provider": self.name, "status": 200, "attempts": 1, "latency_ms": 0.0}
        user = json.loads(user_content)
        if "candidates" in user:
            out = {"choice": min(2, len(user["candidates"])), "reason": "mock"}
        else:
            heads = {"TAP": "tap_target", "LONG_PRESS": "long_press_target", "TYPE_TEXT": "type_target", "OPEN_APP": "app_target"}
            op = next((o for o in user["operations"] if o not in ("DONE", "BLOCKED", "ANSWER")), next(iter(user["operations"])))
            head = heads.get(op) or ("scroll_target" if op.startswith("SCROLL_") else None)
            targets = user["targets"].get(head) if head else None
            out = {"operation": op, "target": next(iter(targets)) if targets else None, "reason": "mock"}
        response = {"model": "mock-rerun", "choices": [{"message": {"role": "assistant", "content": json.dumps(out)}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return response, {"provider": self.name, "status": 200, "attempts": 1, "latency_ms": 0.0}
