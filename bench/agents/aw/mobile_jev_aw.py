"""mobile_jev_aw: AndroidWorld copy of mobile_jev_ac (copied 2026-09-30 from bench/agents/mobile_jev_ac.py, v6).
Additions as jev_mobile_aw: answer / answer_on_finish / input_mode (native = type_only) / clear_before_type /
app_source=list:device_apps / enter=keyboard_enter / needs_input_retries.
type_requires_focus=false (AndroidWorld default): TYPE_TEXT is offered on every editable field and Jev picks the field
(question type_target), in every input_mode. AndroidWorld's accessibility forest has no element focus, so upstream's
"type into the focused field" never offered TYPE_TEXT on a form with several fields (docs/EXTENSIONS.md AW-1).
input_mode native then means tap_type (tap the chosen field, clear, type; no Enter). With input_mode=type_only the
chosen field is not tapped: the text goes to whichever field the device has focused.
type_requires_focus=true restores upstream (focused field only, native = type_only).
"""
"""mobile_jev_ac: droidrun/mobile-jev (Python port) extended to AndroidControl's action space.

Uses the port's extension hooks (`ext`); with ext=None the port equals the Node original.
Additions (params in configs/agents.yaml):
  long_press   LONG_PRESS over long-clickable elements                       (on)
  text_source  llm: TYPE_TEXT value from the text model (shared prompt)      (llm)
               goal_spans: upstream; goal_spans_capped: low-level spans first, cap 254
  app_source   list:subset_vocab: upstream app list built from the subset's open_app answers (default)
               list:ac_vocab: all prepared shards' open_app names; llm: OPEN_APP without a list, name
               from the text model; none. List packages are synthetic.
  enter_offer  keyboard: ENTER only with a keyboard shown (default) | never | upstream (focused input)
  enter        map_to_keyboard | keep_raw                                    (map_to_keyboard)
  terminal     keep | suppress                                               (keep)
  invisible_nodes keep | drop                                                (keep)
Choice budget guard: tap / long-press candidates above 255 are trimmed (visible first).
"""

from typing import Any, Dict, List, Optional

from ...core.observation import confirm_key
from ...core.types import PredAction, Prediction, StepInput
from . import generation
from .mobile_jev_base import MobileJevAgent, _bounds_list, synthetic_package
from .mobile_jev_policy.policy import TypeSafeChoiceError, TypeSafePolicy


class MobileJevAwAgent(MobileJevAgent):
    EXTENSION_VERSION = "aw1"  # AndroidWorld copy of mobile_jev_ac v6
    LOOP_RULES = "mobile_jev"  # bench/online/androidworld/loop_rules.py
    NATIVE_INPUT = "tap_type"  # with type_requires_focus=false; "type_only" (upstream device.mjs 'type') when it is true
    required_providers = {"decision": "systemone", "text": "chat"}
    action_space = {"click", "long_press", "input_text", "scroll", "open_app", "navigate_back", "navigate_home", "wait", "answer"}

    DEFAULTS = {
        "long_press": True,
        "text_source": "llm",
        "app_source": "list:device_apps",
        "enter_offer": "keyboard",
        "scroll_description": "finger_hint",
        "scroll_region": "element",  # swipe inside the scrollable region Jev chose (AW-11) | screen: from the screen centre
        "label_fallback": "resource_class",
        "blocked_offer": "keep",
        "enter": "keyboard_enter",
        "answer": "on",
        "answer_on_finish": "off",
        "input_mode": "native",
        "type_requires_focus": False,
        "app_match": "relaxed",  # which installed apps count as "named in the goal": relaxed (AW-12) | exact (upstream)
        "row_checked_from_child": True,  # a row that owns one Switch/CheckBox reports its on/off state (AW-9); False = upstream
        "clear_before_type": True,
        "needs_input_retries": 1,
        "terminal": "keep",
        "invisible_nodes": "keep",
        "threshold": 0,
        "text_reasoning": "none",
        "invalid_response_retries": 2,
        **generation.CONTEXT_DEFAULTS,
    }

    def __init__(self, name, providers, params):
        super().__init__(name, providers, {**self.DEFAULTS, **params})
        self.action_space = set(self.action_space) - ({"answer"} if self.params["answer"] != "on" else set())
        if self.params["input_mode"] not in ("native", "tap_type", "type_only", "tap_type_enter"):
            raise ValueError("input_mode must be native, tap_type, type_only or tap_type_enter")
        if self.params["scroll_region"] not in ("element", "screen"):
            raise ValueError("scroll_region must be element or screen")
        if self.params["app_match"] not in ("relaxed", "exact"):
            raise ValueError("app_match must be relaxed or exact")
        if not isinstance(self.params["type_requires_focus"], bool):
            raise ValueError("type_requires_focus must be true or false")
        if self.params["text_source"] not in ("llm", "goal_spans", "goal_spans_capped"):
            raise ValueError("text_source must be llm, goal_spans or goal_spans_capped")
        if self.params["blocked_offer"] not in ("keep", "drop"):
            raise ValueError("blocked_offer must be keep or drop")
        if self.params["label_fallback"] not in ("resource_class", "upstream"):
            raise ValueError("label_fallback must be resource_class or upstream")
        if self.params["scroll_description"] not in ("finger_hint", "upstream"):
            raise ValueError("scroll_description must be finger_hint or upstream")
        if self.params["enter_offer"] not in ("keyboard", "never", "upstream"):
            raise ValueError("enter_offer must be keyboard, never or upstream")

    def setup(self, dataset_meta: Dict[str, Any]) -> None:
        source = self.params["app_source"]
        if source.startswith("list:"):
            vocab = dataset_meta.get(source.split(":", 1)[1]) or []
            if not vocab:
                raise ValueError("app_source %s: dataset_meta has no %s" % (source, source.split(":", 1)[1]))
            # AndroidWorld: real package names (runner: dataset_meta["device_app_packages"]), so upstream's "leave out the
            # foreground app" comparison works. Names without a known package keep the offline placeholder (AW-6).
            packages = dataset_meta.get("device_app_packages") or {}
            self.apps = [{"packageName": packages.get(n) or synthetic_package(n), "label": n} for n in vocab]
        elif source in ("llm", "none"):
            self.apps = []
        else:
            raise ValueError("app_source must be llm, none or list:<vocab>")

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info["extension_version"] = self.EXTENSION_VERSION
        added = ["LONG_PRESS"] if self.params["long_press"] else []
        removed = (["DONE", "BLOCKED"] if self.params["terminal"] == "suppress" else (["BLOCKED"] if self.params["blocked_offer"] == "drop" else [])) + (["ENTER"] if self.params["enter_offer"] == "never" else [])
        info.update(added_operations=added, removed_operations=removed, enter_handling=self.params["enter"], enter_offer=self.params["enter_offer"])
        return info

    def predict(self, step: StepInput) -> Prediction:
        self._local.calls = []
        observation = self.observe(step.windows, step.screen, step.active_package, step.keyboard_visible)
        history = self._history(step.history)
        paths = observation["_paths"]
        ext = {
            "long_press_ids": {p for p, f in paths.items() if f["long"]} if self.params["long_press"] else set(),
            "text_mode": self.params["text_source"],
            "app_mode": "list" if self.apps else self.params["app_source"],
            "terminal": self.params["terminal"],
            "priority": {p: (0 if f["visible"] else 1) for p, f in paths.items()},
            "priority_text": step.step_instruction if step.instruction_mode == "high+low" else "",
            "enter_offer": self.params["enter_offer"],
            "scroll_description": self.params["scroll_description"],
            "label_fallback": self.params["label_fallback"],
            "blocked_offer": self.params["blocked_offer"],
            "class_by_id": {p: f["cls"] for p, f in paths.items()},
            "keyboard_visible": step.keyboard_visible,
            "answer": self.params["answer"] == "on",
            "type_targets": not self.params["type_requires_focus"],
            "app_match": self.params["app_match"],
        }
        observation, context_rounds = self.fit_context(observation, step, history, ext)
        policy = TypeSafePolicy(request=self._request, model=self.providers["decision"].model, threshold=float(self.params["threshold"]))
        decision, retries = generation.with_invalid_response_retry(
            lambda: policy.decide(step.agent_goal, observation, history, [], self.apps, dict(ext)),
            lambda e: isinstance(e, TypeSafeChoiceError),
            int(self.params["invalid_response_retries"]),
        )
        tags = ["invalid_response_retry"] if retries else []
        if context_rounds:
            tags.append("context_trimmed")
        if decision.pop("_text_overflow"):
            tags.append("text_candidates_overflow")
        if decision.pop("_trimmed"):
            tags.append("choice_trimmed")
        if self.apps:
            tags.append("app_list")  # an installed-app list was offered (offline name: synthetic_apps)
        decision.pop("_request")
        pred = self._map_ext(decision, observation, step, history, tags)
        if pred.type == "done" and self.params["answer_on_finish"] == "on":
            pred.text = self._answer_text(step, observation, history)
            pred.detail = "done with answer" if pred.text else "done (no answer found)"
        if pred.type == "enter" and self.params["enter"] == "map_to_keyboard":
            key = confirm_key(step.ime_windows)
            if key is not None:
                b = key["bounds"]
                label = key.get("content_description") or key.get("text") or "keyboard action key"
                pred = PredAction(type="click", x=(b[0] + b[2]) // 2, y=(b[1] + b[3]) // 2, bounds=list(b), raw_operation="ENTER", detail="ENTER -> keyboard key %s" % label)
                tags.append("enter_mapped_to_keyboard")
        calls = self._local.calls
        offered = [_bounds_list(e["bounds"]) for e in observation["elements"] if e["enabled"] and (e["clickable"] or e["editable"] or e["id"] in ext["long_press_ids"])]
        return Prediction(
            action=pred,
            tags=tags,
            latency_ms=float(sum(c["meta"]["latency_ms"] for c in calls)),
            usage=generation.sum_usage(calls),
            raw={"decision": decision, "calls": calls, "n_elements": len(observation["elements"]), "offered_bounds": offered, "invalid_response_retries": retries, "context_trim_rounds": context_rounds},
        )

    def _dry_body(self, observation, step: StepInput, history, ext):
        """Build the Jev request without sending it (None if the upstream size check rejects it)."""

        class _Captured(Exception):
            pass

        box = {}

        def capture(body):
            box["body"] = body
            raise _Captured()

        policy = TypeSafePolicy(request=capture, model=self.providers["decision"].model, threshold=float(self.params["threshold"]))
        try:
            policy.decide(step.agent_goal, observation, history, [], self.apps, dict(ext))
        except _Captured:
            return box["body"]
        except ValueError as error:  # upstream: "Screen is too large for this policy ..."
            if "too large" in str(error):
                return None
            raise
        return box.get("body")

    def fit_context(self, observation, step: StepInput, history, ext):
        """Context guard (not upstream). Shrinks a copy of the observation until the request fits:
        drop invisible elements -> drop text-only elements from the end -> drop the last 20% of
        actionable elements. The focused input element is always kept. Returns (observation, rounds)."""
        rounds = 0
        paths = observation["_paths"]
        keep_id = observation["phone"].get("inputElementId")
        while True:
            body = self._dry_body(observation, step, history, ext)
            if body is not None and not generation.over_context_budget(body, self.params):
                return observation, rounds
            rounds += 1
            elements = observation["elements"]
            actionable = lambda e: e["clickable"] or e["editable"] or e["scrollable"] or e["id"] in ext["long_press_ids"]
            hidden = [e for e in elements if e["id"] != keep_id and not paths.get(e["id"], {}).get("visible", True)]
            text_only = [e for e in elements if e["id"] != keep_id and not actionable(e)]
            if hidden:
                drop = {e["id"] for e in hidden}
            elif text_only:
                drop = {e["id"] for e in text_only[len(text_only) // 2 :]}
            else:
                acts = [e for e in elements if e["id"] != keep_id]
                if len(acts) <= 1:
                    return observation, rounds
                drop = {e["id"] for e in acts[-max(1, len(acts) // 5) :]}
            observation = {**observation, "elements": [e for e in elements if e["id"] not in drop]}

    def _describe_tap(self, target, observation):
        """History label with the same fallback as the element list (resource id, else class name)."""
        from .mobile_jev_policy.policy import fallback_label

        text = super()._describe_tap(target, observation)
        if self.params["label_fallback"] != "resource_class":
            return text
        if text.startswith("Tap ") and text.endswith("."):
            return "Tap %s." % fallback_label(target, text[4:-1], {p: f["cls"] for p, f in observation["_paths"].items()})
        if text.startswith("Focus text input: ") and text.endswith("."):
            return fallback_label(target, text[:-1], {}) + "."
        return text

    def _answer_text(self, step: StepInput, observation, history) -> Optional[str]:
        page = self._page_context(observation, step.active_package)
        gen_history = [{"action": h.get("label"), "text": h.get("text")} for h in history]
        try:
            text, gen_calls, _ = generation.with_regeneration(
                lambda: generation.answer_text(self.providers["text"], step.agent_goal, page, gen_history, self.params.get("text_reasoning")),
                int(self.params["needs_input_retries"]),
            )
            self._local.calls.extend(gen_calls)
            return text
        except generation.GenerationError as error:
            self._local.calls.extend(getattr(error, "calls", []))
            return None

    def fingerprint(self, step: StepInput) -> str:
        """Upstream observation fingerprint (for the adapter's repeated-action rule)."""
        return self.observe(step.windows, step.screen, step.active_package, step.keyboard_visible)["fingerprint"]

    def generate_answer(self, step: StepInput) -> Dict[str, Any]:
        self._local.calls = []
        observation = self.observe(step.windows, step.screen, step.active_package, step.keyboard_visible)
        text = self._answer_text(step, observation, self._history(step.history))
        return {"text": text, "calls": list(self._local.calls)}

    def input_mode(self) -> str:
        mode = self.params["input_mode"]
        if mode != "native":
            return mode
        return "type_only" if self.params["type_requires_focus"] else self.NATIVE_INPUT

    def _page_context(self, observation: Dict[str, Any], package: str) -> Dict[str, Any]:
        texts = [v for e in observation["elements"] for v in (e["text"], e["label"]) if v]
        return {"app": package, "activity": "", "text": "\n".join(texts)}

    def _map_ext(self, decision: Dict[str, Any], observation: Dict[str, Any], step: StepInput, history: List[Dict[str, Any]], tags: List[str]) -> PredAction:
        if decision["status"] != "action":
            return self._map(decision, observation)
        action, op = decision["action"], decision["operation"]
        text_provider, reasoning = self.providers["text"], self.params.get("text_reasoning")
        gen_history = [{"action": h.get("label"), "text": h.get("text")} for h in history]
        page = self._page_context(observation, step.active_package)
        if action["type"] == "long-press-element":
            element = next(e for e in observation["elements"] if e["id"] == action["elementId"])
            b = element["bounds"]
            return PredAction(type="long_press", x=(b["left"] + b["right"]) // 2, y=(b["top"] + b["bottom"]) // 2, bounds=_bounds_list(b), raw_operation=op, detail=decision.get("label", ""))
        if action["type"] == "answer":
            text = self._answer_text(step, observation, history)
            if text is None:
                return PredAction(type="needs_input", raw_operation=op, detail="text model found no answer")
            return PredAction(type="answer", text=text, raw_operation=op)
        retries = int(self.params["needs_input_retries"])
        if action["type"] == "type-llm":
            input_id = action.get("elementId") or observation["phone"].get("inputElementId")  # chosen field, else the focused one
            field = next((e for e in observation["elements"] if e["id"] == input_id), None) or {}
            field_info = {"label": field.get("hint") or field.get("label") or field.get("text") or "focused field", "role": "EditText", "value": field.get("text", "")}
            try:
                value, gen_calls, _ = generation.with_regeneration(lambda: generation.field_value(text_provider, step.agent_goal, field_info, page, gen_history, reasoning), retries)
                self._local.calls.extend(gen_calls)
            except generation.GenerationError as error:
                self._local.calls.extend(getattr(error, "calls", []))
                return PredAction(type="needs_input", raw_operation=op, detail=str(error))
            if value is None:
                return PredAction(type="needs_input", raw_operation=op, detail="text model returned null")
            # The focused field's area travels along so tapping input modes can tap it first.
            b = field.get("bounds")
            if b:
                return PredAction(type="input_text", text=value, x=(b["left"] + b["right"]) // 2, y=(b["top"] + b["bottom"]) // 2, bounds=_bounds_list(b), raw_operation=op, detail=field_info["label"])
            return PredAction(type="input_text", text=value, raw_operation=op, detail=field_info["label"])
        if action["type"] == "open-app-llm":
            try:
                name, gen_calls, _ = generation.with_regeneration(lambda: generation.app_name(text_provider, step.agent_goal, page, gen_history, reasoning), retries)
                self._local.calls.extend(gen_calls)
            except generation.GenerationError as error:
                self._local.calls.extend(getattr(error, "calls", []))
                return PredAction(type="needs_input", raw_operation=op, detail=str(error))
            if not name:
                return PredAction(type="needs_input", raw_operation=op, detail="text model named no app")
            return PredAction(type="open_app", app_name=name, raw_operation=op)
        return self._map(decision, observation)
