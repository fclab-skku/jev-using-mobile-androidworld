"""AndroidWorld adapter: runs one of our step-level agents (bench/agents/aw/) as an AndroidWorld agent.

Per AndroidWorld step:  State -> StepInput (same window decoding/selection as the offline benchmark)
  -> loop-rule check -> agent.predict -> loop-rule override -> execute -> log one row in steps.jsonl.

Execution map (docs/RUN_GUIDE.md):
  click / long_press (x, y)   JSONAction click / long_press at the element centre
  scroll direction [+bounds]  swipe inside the target element's bounds (whole screen without one), same
                              geometry as AndroidWorld's scroll (direction = content direction, like AndroidControl)
  input_text                  by input_mode:
                                tap_type_enter  AndroidWorld input_text (tap, clear if clear_before_type, type, Enter)
                                tap_type        adb tap -> (clear) -> type, no Enter
                                type_only       (clear) -> type into the focused field, no Enter
  open_app / navigate_back / navigate_home / wait / enter(keyboard_enter)
  answer                      JSONAction answer, then status complete
  done                        status complete (answer first when the prediction carries text)
  blocked, loop-rule stop     status infeasible
  needs_input / invalid       wait (the agent already regenerated once)
"""

import dataclasses
import datetime
import gzip
import hashlib
import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...core.observation import area, contains, element_at, keyboard_windows, node_label, select_windows, walk
from ...core.types import HistoryItem, PredAction, StepInput
from ...data.a11y_proto import decode_forest
from .loop_rules import rules_for

try:  # android_world is only needed on a live run; unit tests pass fakes
    from android_world.agents import base_agent as _base_agent

    _AgentBase = _base_agent.EnvironmentInteractingAgent
except ImportError:  # pragma: no cover - exercised without android_world installed
    _base_agent = None

    class _AgentBase:  # minimal stand-in with the same surface
        def __init__(self, env, name="", transition_pause=1.0):
            self._env, self._name, self._transition_pause, self._max_steps = env, name, transition_pause, None

        @property
        def env(self):
            return self._env

        def set_max_steps(self, max_steps):
            self._max_steps = max_steps

        def reset(self, go_home=False):
            self.env.reset(go_home=go_home)

        def get_post_transition_state(self):  # same rule as android_world base_agent
            if self._transition_pause is None:
                return self.env.get_state(wait_to_stabilize=True)
            time.sleep(self._transition_pause)
            return self.env.get_state(wait_to_stabilize=False)


CLEAR_FIELD = ["shell", "input", "keycombination", "113", "29", "&&", "input", "keyevent", "67"]  # = AndroidWorld's clear_text
MAX_CONSECUTIVE_ERRORS = 3


def _result(done: bool, data: Dict[str, Any]):
    if _base_agent is not None:
        return _base_agent.AgentInteractionResult(done, data)
    return type("AgentInteractionResult", (), {"done": done, "data": data})()


def active_package(windows: List[Dict[str, Any]]) -> str:
    """Same rule as the AndroidControl preparation (bench/data/prepare.py)."""
    apps = sorted((w for w in windows if w["type"] == "APPLICATION"), key=lambda w: (not w["is_active"], -w["layer"]))
    for window in apps:
        for root in window["roots"]:
            if root.get("package_name"):
                return root["package_name"]
    return ""


def field_at(windows: List[Dict[str, Any]], x: float, y: float) -> Optional[Dict[str, Any]]:
    """Smallest editable node containing the point (the field a typing action targets)."""
    hits = [n for w in windows for n in walk(w["roots"]) if n.get("is_editable") and contains(n["bounds"], x, y)]
    return min(hits, key=lambda n: area(n["bounds"])) if hits else None


def field_has_text(node: Optional[Dict[str, Any]]) -> Tuple[Optional[bool], str]:
    """(has text, why). An empty field shows its hint as text in AndroidWorld's forest; None = cannot tell."""
    if node is None:
        return None, "no editable node at the tap point"
    text, hint = node.get("text") or "", node.get("hint_text") or ""
    if not text:
        return False, "field text is empty"
    if hint and text == hint:
        return False, "field shows only its hint %r" % hint[:40]
    return True, "field holds %r" % text[:40]


def drop_keyboard_covered(windows: List[Dict[str, Any]], screen: List[int]) -> int:
    """Remove app-window nodes that sit under the on-screen keyboard (their centre is inside an INPUT_METHOD window).
    The accessibility tree still lists them as visible, but a tap on them hits a keyboard key (AW-14). Containers
    that still have uncovered children are kept. Only windows drawn below the keyboard (lower layer) are pruned, so a
    popup shown on top of the keyboard keeps its nodes. Mutates `windows`; returns the number of removed nodes."""
    boxes = []
    for w in windows:
        if w["type"] == "INPUT_METHOD" and w["roots"]:
            l, t, r, b = w["bounds"]
            l, t, r, b = max(l, 0), max(t, 0), min(r, screen[0]), min(b, screen[1])
            if r > l and b > t and (b - t) < screen[1] * 0.75:  # a full-screen IME window is not a keyboard area
                boxes.append((l, t, r, b, w.get("layer", 0)))
    if not boxes:
        return 0
    removed = 0

    def covered(node, layer) -> bool:
        l, t, r, b = node["bounds"]
        cx, cy = (l + r) / 2.0, (t + b) / 2.0
        return any(layer < kl and bl <= cx <= br and bt <= cy <= bb for bl, bt, br, bb, kl in boxes)

    def prune(nodes, layer):
        nonlocal removed
        kept = []
        for node in nodes:
            node["children"] = prune(node.get("children") or [], layer)
            if covered(node, layer) and not node["children"]:
                removed += 1
            else:
                kept.append(node)
        return kept

    for w in windows:
        if w["type"] != "INPUT_METHOD":
            w["roots"] = prune(w["roots"], w.get("layer", 0))
    return removed


class Actuator:
    """Thin wrapper over the AndroidWorld env so tests can substitute a fake."""

    def __init__(self, env):
        from android_world.env import adb_utils, json_action

        self.env, self.adb, self.json_action = env, adb_utils, json_action

    def aw(self, action: Dict[str, Any]) -> None:
        self.env.execute_action(self.json_action.JSONAction(**action))

    def tap(self, x: int, y: int) -> None:
        self.adb.tap_screen(int(x), int(y), self.env.controller)

    def clear(self) -> None:
        self.adb.issue_generic_request(CLEAR_FIELD, self.env.controller)

    def type_text(self, text: str) -> None:
        self.adb.type_text(text, self.env.controller, timeout_sec=10)

    def swipe(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.adb.issue_generic_request(self.adb.generate_swipe_command(int(x1), int(y1), int(x2), int(y2)), self.env.controller)


def scroll_path(direction: str, bounds: Optional[List[int]], screen: List[int]) -> Tuple[int, int, int, int]:
    """AndroidWorld actuation's scroll geometry: centre -> edge, finger opposite to the content direction."""
    w, h = screen
    l, t, r, b = (max(bounds[0], 0), max(bounds[1], 0), min(bounds[2], w), min(bounds[3], h)) if bounds else (0, 0, w, h)
    cx, cy = (l + r) // 2, (t + b) // 2
    end = {"down": (cx, t), "up": (cx, b), "right": (l, cy), "left": (r, cy)}[direction]
    return cx, cy, end[0], end[1]


class AwAdapter(_AgentBase):
    def __init__(
        self,
        env,
        agent,
        run_dir: Path,
        window_mode: str = "no_statusbar",
        loop_rules: str = "default",
        screenshots: bool = True,
        save_forest: bool = True,
        hide_keyboard_covered: bool = True,
        repeat_limit: int = 3,
        save_raw: bool = True,
        actuator: Optional[Actuator] = None,
        task_lookup: Optional[Callable[[str], Dict[str, Any]]] = None,
        app_vocab: Optional[List[str]] = None,
        transition_pause: Optional[float] = 1.0,
        settle_s: float = 1.0,
    ):
        super().__init__(env, name=agent.name, transition_pause=transition_pause)
        self.agent, self.run_dir, self.window_mode = agent, Path(run_dir), window_mode
        self.loop_rules_mode, self.screenshots, self.save_raw = loop_rules, screenshots, save_raw
        self.save_forest = save_forest
        self.hide_keyboard_covered, self.repeat_limit = hide_keyboard_covered, repeat_limit
        self.actuator = actuator or Actuator(env)
        self.task_lookup = task_lookup or (lambda goal: {})
        self.settle_s = settle_s
        self.app_vocab = app_vocab or []
        self.input_mode = agent.input_mode() if hasattr(agent, "input_mode") else "tap_type_enter"
        self.clear_before_type = bool(agent.params.get("clear_before_type", True))
        self.answer_on_finish = agent.params.get("answer_on_finish") == "on"
        self.episode_index = -1
        self.current_task: Optional[Dict[str, Any]] = None  # set by the runner when AndroidWorld initializes a task
        self.episodes: List[Dict[str, Any]] = []  # per-episode summaries for tasks.jsonl
        self._lock = threading.Lock()
        (self.run_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        if save_forest:
            (self.run_dir / "forest").mkdir(parents=True, exist_ok=True)
        self._new_episode()

    # ---- episode bookkeeping ----
    def _new_episode(self) -> None:
        self.history: List[HistoryItem] = []
        self.records: List[Dict[str, Any]] = []  # executed actions (loop rules)
        self.rules = rules_for(self.agent, self.loop_rules_mode, self.repeat_limit)
        self.consecutive_errors = 0
        self.steps_taken = 0  # AndroidWorld steps in this episode (budget = set_max_steps)
        self.summary: Dict[str, Any] = {}

    def set_task(self, name: str, instance: int) -> None:
        """Called just before AndroidWorld initializes a task. Goals cannot identify a task: several templates share
        one goal text (SystemWifiTurnOn / ...Verify) and some goals are only final after initialization."""
        self.current_task = {"name": name, "instance": instance}

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.episode_index += 1
        self._new_episode()

    # ---- observation ----
    def observe(self, state, goal: str) -> StepInput:
        raw_forest = state.forest.SerializeToString()
        windows = decode_forest(raw_forest)
        self.forest_path = None
        if self.save_forest:  # the unmodified AndroidAccessibilityForest proto; decode with bench.data.a11y_proto.decode_forest
            self.forest_path = str(self.run_dir / "forest" / ("e%03d_s%03d.pb.gz" % (self.episode_index, self.steps_taken)))
            with gzip.open(self.forest_path, "wb") as f:
                f.write(raw_forest)
        screen = [int(v) for v in self.env.logical_screen_size]
        self.covered_removed = drop_keyboard_covered(windows, screen) if self.hide_keyboard_covered else 0
        package = active_package(windows)
        if not package:
            package = (self.env.foreground_activity_name or "").split("/")[0]
        selected = select_windows(windows, self.window_mode, screen)
        shot = None
        if self.screenshots and getattr(state, "pixels", None) is not None:
            shot = str(self.run_dir / "screenshots" / ("e%03d_s%03d.png" % (self.episode_index, self.steps_taken)))
            try:
                from PIL import Image

                Image.fromarray(state.pixels).save(shot)
            except Exception:  # a missing screenshot must not stop the episode
                shot = None
        return StepInput(
            episode_id=self.episode_index,
            step_index=len(self.records),
            num_steps=self._max_steps or 0,
            goal=goal,
            step_instruction="",
            instruction_mode="high",
            agent_goal=goal,
            windows=selected,
            screen=screen,
            active_package=package,
            keyboard_visible=any(w["type"] == "INPUT_METHOD" for w in windows),
            screenshot=shot,
            ime_windows=keyboard_windows(windows),
            all_windows=select_windows(windows, "all", screen),
            history=list(self.history),
            app_vocab=list(self.app_vocab),
        )

    def _fingerprint(self, step: StepInput) -> str:
        if hasattr(self.agent, "fingerprint"):
            return self.agent.fingerprint(step)
        return hashlib.sha256(json.dumps([step.active_package, step.windows], sort_keys=True, default=str).encode()).hexdigest()

    # ---- AndroidWorld step ----
    def step(self, goal: str):
        started = time.time()
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        task = self.current_task or self.task_lookup(goal)
        state = self.get_post_transition_state()  # fixed pause + one read, or wait_to_stabilize (transition_pause None)
        read_seconds = round(time.time() - started, 2)
        step = self.observe(state, goal)
        fingerprint = self._fingerprint(step)
        if self.history:  # the previous action's outcome is visible now
            last = self.history[-1]
            last.next_windows, last.package_after = step.windows, step.active_package
            self.records[-1]["page_changed"] = self.records[-1]["fingerprint"] != fingerprint

        row: Dict[str, Any] = {
            "ts": stamp, "task": task.get("name"), "instance": task.get("instance"), "episode": self.episode_index,
            "step": len(self.records), "goal": goal, "package": step.active_package, "screenshot": step.screenshot,
            "forest": self.forest_path, "fingerprint": fingerprint[:16], "keyboard_visible": step.keyboard_visible,
            "keyboard_covered_removed": self.covered_removed, "read_seconds": read_seconds,
        }
        stop = self.rules.before_predict(self.records)
        pred, tags, raw, usage, rule_events, error = None, [], None, {}, [], None
        if stop is None:
            try:
                prediction = self.agent.predict(step)
                pred, tags, raw, usage = prediction.action, list(prediction.tags), prediction.raw, prediction.usage or {}
                self.consecutive_errors = 0
                pred, stop, rule_events = self.rules.after_predict(pred, tags, fingerprint)
                if "veto_limit_accept" in rule_events and self.answer_on_finish:
                    answer = self.agent.generate_answer(step)
                    pred.text = answer.get("text")
                    (raw or {}).setdefault("calls", []).extend(answer.get("calls", []))
            except Exception as e:  # provider/network/agent failure: wait and try again, give up after 3 in a row
                error = "%s: %s" % (type(e).__name__, e)
                row["traceback"] = traceback.format_exc()
                self.consecutive_errors += 1
                pred = PredAction(type="wait", detail="agent error -> wait")
                if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    stop = "agent_errors"

        if stop is not None:
            executed, done, hist_action = self._finish("infeasible", None)
            row["stop_reason"] = stop
        else:
            executed, done, hist_action = self.execute(pred, step)
        if hist_action is not None:
            self._remember(hist_action, pred, step, fingerprint)
        self.steps_taken += 1
        if done:
            row["end_reason"] = stop or pred.type  # loop-rule / error stop, else the agent's own done / answer / blocked
        elif self._max_steps and self.steps_taken >= self._max_steps:
            row["end_reason"] = "budget_exhausted"  # AndroidWorld ends the episode after this step (counted as a failure)
        if done:
            self.summary = {"stop_reason": stop or ("answer" if any(a["action_type"] == "answer" for a in executed) else pred.type), "answer": next((a.get("text") for a in executed if a["action_type"] == "answer"), None)}

        row.update(
            pred=dataclasses.asdict(pred) if pred is not None else None,
            tags=tags, rule_events=rule_events, rules_state=self.rules.state(), executed=executed, done=done,
            error=error, usage=usage, seconds=round(time.time() - started, 2),
        )
        if self.save_raw and raw is not None:
            row["raw"] = raw
        with self._lock, open(self.run_dir / "steps.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        data = {"pred_type": pred.type if pred is not None else None, "executed": executed, "tags": tags, "stop_reason": row.get("stop_reason"), "error": error}
        return _result(done, data)

    # ---- execution ----
    def _finish(self, goal_status: str, answer: Optional[str]) -> Tuple[List[Dict[str, Any]], bool, None]:
        executed = []
        if answer:
            executed.append({"action_type": "answer", "text": answer})
            self.actuator.aw(executed[-1])
        executed.append({"action_type": "status", "goal_status": goal_status})
        self.actuator.aw(executed[-1])
        return executed, True, None

    def execute(self, pred: PredAction, step: StepInput) -> Tuple[List[Dict[str, Any]], bool, Optional[Dict[str, Any]]]:
        """-> (AndroidWorld/adb actions executed, episode done, history action in AndroidControl shape)."""
        t = pred.type
        if t == "answer":
            return self._finish("complete", pred.text or "")
        if t == "done":
            return self._finish("complete", pred.text)
        if t == "blocked":
            return self._finish("infeasible", None)
        if t in ("click", "long_press"):
            a = {"action_type": t, "x": int(pred.x), "y": int(pred.y)}
            self.actuator.aw(a)
            return [a], False, dict(a)
        if t == "scroll":
            x1, y1, x2, y2 = scroll_path(pred.direction, pred.bounds, step.screen)
            self.actuator.swipe(x1, y1, x2, y2)
            return [{"action_type": "swipe_adb", "path": [x1, y1, x2, y2], "direction": pred.direction}], False, {"action_type": "scroll", "direction": pred.direction}
        if t == "input_text":
            return self._input(pred, step)
        if t == "open_app":
            a = {"action_type": "open_app", "app_name": pred.app_name}
            self.actuator.aw(a)
            return [a], False, dict(a)
        if t in ("navigate_back", "navigate_home"):
            a = {"action_type": t}
            self.actuator.aw(a)
            return [a], False, dict(a)
        if t == "enter":
            a = {"action_type": "keyboard_enter"}
            self.actuator.aw(a)
            return [a], False, dict(a)
        # wait, needs_input, invalid
        a = {"action_type": "wait"}
        self.actuator.aw(a)
        return [a], False, {"action_type": "wait"}

    def _input(self, pred: PredAction, step: Optional[StepInput] = None) -> Tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
        text = pred.text or ""
        has_xy = pred.x is not None and pred.y is not None
        hist: Dict[str, Any] = {"action_type": "input_text", "text": text}
        # Clear only a field that holds text. The clear is "select all + delete" key presses; in an empty field the
        # select-all goes to the app instead (Files selected every file, AW-13). When the target field is unknown
        # (no coordinates) the old behaviour stays: clear.
        has_text, why = field_has_text(field_at(step.all_windows, pred.x, pred.y) if (step is not None and has_xy) else None)
        do_clear = self.clear_before_type and has_text is not False
        clear_note = {"clear": do_clear, "clear_reason": ("clear_before_type is off" if not self.clear_before_type else why)}
        if pred.clear is False:  # the agent does not know the target field: never select-all + delete blindly (AW-13)
            do_clear, clear_note = False, {"clear": False, "clear_reason": "agent: target field unknown"}
        if self.input_mode == "tap_type_enter":
            a: Dict[str, Any] = {"action_type": "input_text", "text": text, "clear_text": do_clear}
            if has_xy:
                a.update(x=int(pred.x), y=int(pred.y))
                hist.update(x=a["x"], y=a["y"])
            self.actuator.aw(a)
            return [{**a, **clear_note}], False, hist
        executed: List[Dict[str, Any]] = []
        if self.input_mode == "tap_type" and has_xy:
            self.actuator.tap(pred.x, pred.y)
            executed.append({"action_type": "tap_adb", "x": int(pred.x), "y": int(pred.y)})
            hist.update(x=int(pred.x), y=int(pred.y))
            time.sleep(self.settle_s)
        if do_clear:
            self.actuator.clear()
            executed.append({"action_type": "clear_adb", "reason": clear_note["clear_reason"]})
            time.sleep(self.settle_s)
        else:
            executed.append({"action_type": "clear_skipped", "reason": clear_note["clear_reason"]})
        if text:
            self.actuator.type_text(text)
            executed.append({"action_type": "type_adb", "text": text})
        return executed, False, hist

    def _remember(self, action: Dict[str, Any], pred: PredAction, step: StepInput, fingerprint: str) -> None:
        node = element_at(step.windows, action["x"], action["y"]) if "x" in action else None
        self.history.append(
            HistoryItem(
                action=action,
                instruction="",
                target_label=node_label(node) if node else "",
                target_bounds=list(node["bounds"]) if node else None,
                package_before=step.active_package,
                package_after=step.active_package,  # updated on the next observation
                windows=step.windows,
                next_windows=step.windows,
                screen=step.screen,
            )
        )
        self.records.append({"kind": "wait" if action["action_type"] == "wait" else action["action_type"], "fingerprint": fingerprint, "page_changed": None})
