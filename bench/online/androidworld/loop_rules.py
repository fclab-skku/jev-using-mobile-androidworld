"""Upstream multi-step stop rules, reproduced outside the (step-level) agents.

The agents answer one step at a time; the rules their upstream loops apply across steps live here.
Each rule set sees the executed-step records the adapter keeps and may stop the episode (reported to
AndroidWorld as `status infeasible`) or override one prediction.

  jev_mobile  (xinwang-nwpu/jev-mobile, jev_mobile/agent.py)
    - DONE veto: the goal gate may veto DONE (the agent then returns WAIT, tag done_vetoed). After
      MAX_DONE_VETOES consecutive vetoes the repeated DONE wins. The count resets on any other operation.
    - stuck: the last 3 executed actions left the semantic fingerprint unchanged and none was a WAIT.
    - MAX_STEPS = 60 executed actions.
  mobile_jev  (droidrun/mobile-jev, scripts/mobile-agent/agent.mjs runAgent)
    - stuck: a non-wait action repeats a `fingerprint:action` signature already executed in this run.
      Upstream stops at the first repeat (repeat_limit 1). An identical confirm dialog for the second item of a
      "delete several" task has the same signature, so legitimate repeats were stopped (docs/EXTENSIONS.md
      AW-8). Mode `default` therefore lets the same signature run `repeat_limit` times (3) and stops on the next one.
    - loading_timeout: consecutive WAITs for 15 s (waitTimeoutMs, wall clock since the first of them).
    (maxSteps is a CLI option upstream; AndroidWorld's step budget applies instead.)
  repeat_guard (not upstream; for mobilegptsys/MobileGPT, whose server has no repeat or step-limit rule)
    - stuck: the mobile_jev signature rule with the same repeat_limit. Without it a wrong choice was repeated until
      AndroidWorld's step budget ran out (76 of 90 failures in aw_full_mgpt_v2).
  none        (no rule; only AndroidWorld's budget)

Modes (aw-run --loop-rules): default = jev_mobile as upstream, mobile_jev with repeat_limit, MobileGPT with repeat_guard;
upstream = every agent exactly as its original (mobile_jev stops at the first repeat, MobileGPT has no rule); off = none.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from ...core.types import PredAction

WAIT_TYPES = ("wait", "needs_input", "invalid")  # executed as an AndroidWorld wait
TERMINAL_TYPES = ("done", "blocked", "answer")


class NoRules:
    name = "none"

    def reset(self) -> None:
        pass

    def before_predict(self, records: List[Dict[str, Any]]) -> Optional[str]:
        """Stop reason checked on a fresh observation, before any model call."""
        return None

    def after_predict(self, pred: PredAction, tags: List[str], fingerprint: str) -> Tuple[PredAction, Optional[str], List[str]]:
        """(prediction to execute, stop reason, rule events)."""
        return pred, None, []

    def state(self) -> Dict[str, Any]:
        return {}


class JevMobileRules(NoRules):
    name = "jev_mobile"
    MAX_DONE_VETOES = 2
    STUCK_WINDOW = 3
    MAX_STEPS = 60

    def reset(self) -> None:
        self.done_vetoes = 0

    def before_predict(self, records):
        if len(records) >= self.MAX_STEPS:
            return "action_budget"
        last = records[-self.STUCK_WINDOW:]
        if len(last) == self.STUCK_WINDOW and all(r["page_changed"] is False and r["kind"] != "wait" for r in last):
            return "stuck"
        return None

    def after_predict(self, pred, tags, fingerprint):
        if "done_vetoed" in tags:
            if self.done_vetoes >= self.MAX_DONE_VETOES:
                self.done_vetoes = 0
                accepted = PredAction(type="done", raw_operation="DONE", detail="DONE accepted after %d consecutive vetoes" % self.MAX_DONE_VETOES)
                return accepted, None, ["veto_limit_accept"]
            self.done_vetoes += 1
            return pred, None, ["done_veto_%d" % self.done_vetoes]
        if pred.raw_operation != "DONE":
            self.done_vetoes = 0
        return pred, None, []

    def state(self):
        return {"done_vetoes": self.done_vetoes}


class MobileJevRules(NoRules):
    name = "mobile_jev"
    WAIT_TIMEOUT_S = 15.0
    WAIT_RULE = True

    def __init__(self, repeat_limit: int = 1):
        if repeat_limit < 1:
            raise ValueError("repeat_limit must be >= 1")
        self.repeat_limit = repeat_limit  # how many times one fingerprint:action signature may be executed

    def reset(self) -> None:
        self.seen: Dict[str, int] = {}
        self.waiting_since: Optional[float] = None

    @staticmethod
    def signature(pred: PredAction, fingerprint: str) -> str:
        action = {k: v for k, v in (("type", pred.type), ("x", pred.x), ("y", pred.y), ("text", pred.text), ("direction", pred.direction), ("app_name", pred.app_name), ("bounds", pred.bounds)) if v is not None}
        return "%s:%s" % (fingerprint, json.dumps(action, sort_keys=True, ensure_ascii=False))

    def after_predict(self, pred, tags, fingerprint):
        if pred.type in TERMINAL_TYPES:
            return pred, None, []
        if pred.type in WAIT_TYPES:
            if not self.WAIT_RULE:
                return pred, None, []
            now = time.monotonic()
            if self.waiting_since is None:
                self.waiting_since = now
            if now - self.waiting_since >= self.WAIT_TIMEOUT_S:
                return pred, "loading_timeout", []
            return pred, None, []
        self.waiting_since = None
        sig = self.signature(pred, fingerprint)
        count = self.seen.get(sig, 0)
        if count >= self.repeat_limit:
            return pred, "stuck", []
        self.seen[sig] = count + 1
        return pred, None, (["repeat_%d" % (count + 1)] if count else [])

    def state(self):
        return {"signatures": len(self.seen), "max_repeats": max(self.seen.values(), default=0), "repeat_limit": self.repeat_limit,
                "waiting_s": round(time.monotonic() - self.waiting_since, 1) if self.waiting_since else 0.0}


class RepeatGuardRules(MobileJevRules):
    """MobileGPT has no loop rule upstream. Same signature rule as mobile_jev, without the WAIT timeout."""

    name = "repeat_guard"
    WAIT_RULE = False


DEFAULT_REPEAT_LIMIT = 3
MODES = ("default", "upstream", "off")


def rules_for(agent: Any, mode: str = "default", repeat_limit: int = DEFAULT_REPEAT_LIMIT) -> NoRules:
    """default: upstream rules with the relaxed repeat limit, plus a repeat guard for agents without a rule;
    upstream: each agent's original rules exactly; off: none (AndroidWorld's step budget only)."""
    if mode not in MODES:
        raise ValueError("loop_rules must be one of %s" % (MODES,))
    kind = "none" if mode == "off" else getattr(agent, "LOOP_RULES", "none")
    if kind == "jev_mobile":
        rules: NoRules = JevMobileRules()
    elif kind == "mobile_jev":
        rules = MobileJevRules(repeat_limit if mode == "default" else 1)
    elif kind == "none" and mode == "default" and getattr(agent, "REPEAT_GUARD", False):
        rules = RepeatGuardRules(repeat_limit)
    else:
        rules = NoRules()
    rules.reset()
    return rules
