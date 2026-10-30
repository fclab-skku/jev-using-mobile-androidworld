# AndroidWorld pipeline: behaviour and options

The vision agent (`mobile_jev_hybrid_aw`) and the text-only reference (`mobile_jev_aw`) share one pipeline. The rules
and question set start from droidrun/mobile-jev, which drives a Mobilerun device; AndroidWorld gives an accessibility
forest and its own action set, so the pipeline adds a conversion layer and the behaviour below. Each change below is an option in
`configs/agents.yaml` (or an `aw-run` flag) and can be switched back to upstream behaviour.
Item numbers (AW-N) are the ones referenced in the code comments.

## Action space
| mobile-jev | this pipeline |
|---|---|
| TAP, TYPE_TEXT (focused field), SCROLL up/down, OPEN_APP, BACK, HOME, ENTER, WAIT, DONE, BLOCKED | + LONG_PRESS on long-clickable elements, SCROLL left/right, ANSWER (information tasks; text from the text model), TYPE_TEXT into a chosen field |

- **ANSWER** (`answer: "on"`): for questions such as "how many …", Jev picks ANSWER and the text model writes the
  answer from the page context; it is sent as AndroidWorld's `answer` action.
- **ENTER** (`enter: keyboard_enter`): AndroidWorld's `keyboard_enter`.
- **Typing** (`input_mode: native`): tap the chosen field, clear it if it holds text, type. No Enter.

## Changes and why
| ID | Change | Why | Param (upstream value) |
|---|---|---|---|
| AW-1 | TYPE_TEXT is offered on every editable field and Jev picks the field (`type_target`) | AndroidWorld's forest has no input focus, so upstream never offered TYPE_TEXT on a form; the agent tapped keyboard keys one by one | `type_requires_focus: false` (`true`) |
| AW-2 | A number returned by the text model (`{"answer": 3}`) is accepted as text | it was rejected as "not a string" and the step lost its answer | — |
| AW-6 | The app list carries real package names | invented package names kept the current app in the OPEN_APP list | `app_source: list:device_apps` |
| AW-8 | The same action on the same screen may run 3 times before the episode stops as `stuck` | upstream stops at the first repeat; AndroidWorld tasks legitimately repeat (delete several items, add several songs) | `aw-run --loop-rules upstream` or `--repeat-limit 1` |
| AW-9 | A row that owns exactly one Switch/CheckBox passes its checked state | Jev saw only "Wi-Fi", not whether it was on | `row_checked_from_child: true` (`false`) |
| AW-11 | SCROLL swipes inside the scrollable region Jev chose | upstream swiped from the screen centre; most scrolls changed nothing | `scroll_region: element` (`screen`) |
| AW-12 | An app counts as "named in the goal" when 2+ consecutive words of its name appear | "Simple Gallery" in the goal did not match the app "simple gallery pro" | `app_match: relaxed` (`exact`) |
| AW-13 | Clear-before-type only when the target field holds text | select-all on an empty field went to the app (Files selected every file) | `clear_before_type: true` |
| AW-14 | Elements lying under the on-screen keyboard are dropped from the candidates | the tree still lists them, but a tap hits a keyboard key | `aw-run --keep-keyboard-covered` |
| AW-22 | `aw-run` refuses to start while the emulator's clipboard sharing is on | the host clipboard appeared in the screen text, was typed into tasks and sent to the model APIs | `aw-run --allow-clipboard-sharing` |

## What is unchanged
- The question set, the answer options and the confidence (P(operation) × P(target | operation)). The text-only
  reference keeps mobile-jev's rule wording; the vision agent rewrites the operation rules and target instructions
  for the screenshot (below).
- android_world: tasks, step budget (10 × complexity), scoring and suite runner are used as they are.

## Vision request (`mobile_jev_hybrid_aw`)
| Param | Values | Effect |
|---|---|---|
| `history` | `labels` (default) · `coords` | recent actions as element labels + screenChanged, or as operation + touch point (0–1000) only. `coords` lowered success to 34/116 (DONE rarely chosen) |
| `rules` | `vision` (default) · `upstream` | screenshot-aware operation rules and target instructions, or the text-only wording |
