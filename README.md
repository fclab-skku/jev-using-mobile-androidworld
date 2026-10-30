# Jev vision agent for AndroidWorld

A mobile GUI agent for [AndroidWorld](https://github.com/google-research/android_world) that decides each step with
TypeSafe System One **Jev 1.14** from the **screenshot**. The screen is given to Jev only as an image; the on-screen
elements appear only as the answer options it chooses from. No world model and no second LLM: one Jev request per
step, plus a small chat model that writes text values (what to type, answers).

| Agent | Jev input | Success (116 tasks) | Steps | Wall time | API cost |
|---|---|---|---|---|---|
| **`mobile_jev_hybrid_aw`** (vision) | screenshot + element options | **41 (35.3%)** | 1,216 | 59 min | $0.30 |
| `mobile_jev_aw` (text-only reference) | element list + screen text | 39 (33.6%) | 1,497 | 68 min | $0.27 |

Both with `typesafe/jev-1.14`, android_world `2655e27`, seed 30, one instance per task, single runs.
Details: [docs/RESULTS.md](docs/RESULTS.md).

## How it decides
One System One request per step:

- **state** — what Jev sees
  ```json
  {"goal": "Delete the note shopping.md in Markor.",
   "app": "net.gsantner.markor",
   "keyboardVisible": false,
   "screenshot": "img-3f9c…",
   "recentActions": [{"operation": "OPEN_APP", "label": "Open markor", "screenChanged": true},
                     {"operation": "LONG_PRESS", "label": "Long press shopping.md.", "screenChanged": true}],
   "availableApps": ["audio recorder", "broccoli", "markor", "..."]}
  ```
  plus the screenshot as an image asset (PNG, scaled to fit 1568 px). No element list and no screen text.
- **questions** — what Jev answers
  - `operation`: TAP, LONG_PRESS, TYPE_TEXT, SCROLL_UP/DOWN/LEFT/RIGHT, OPEN_APP, BACK, HOME, ENTER, WAIT, DONE,
    ANSWER, BLOCKED (only those possible on this screen). The rules tell Jev the screenshot is its only view of the
    screen and that DONE needs visible evidence on it.
  - one target question per operation (`tap_target`, `long_press_target`, `type_target`, `scroll_target`,
    `app_target`) whose options are the screen's elements (`"[4] Delete"`, `"[7] shopping.md"`). Jev finds the
    target on the screenshot, uses the recent actions to avoid repeating finished steps, and picks the option that
    names it.
- **action** — the chosen operation on the chosen element: tap at the element's centre, tap-clear-type for text,
  swipe inside the chosen scroll region, AndroidWorld `answer` for ANSWER. Typed values and answers come from the
  text model (`inception/mercury-2.5`), which reads the page text.
- **confidence** — P(operation) × P(target | operation), logged per step.

The element options come from AndroidWorld's accessibility tree; the tree is used to build the options and to
execute, never shown to Jev as text. Pipeline details and the AndroidWorld-specific behaviour:
[docs/EXTENSIONS.md](docs/EXTENSIONS.md).

## Layout
```
bench/agents/aw/mobile_jev_hybrid_aw.py   the vision agent (state, rules, image asset)
bench/agents/aw/mobile_jev_aw.py          shared pipeline + text-only reference agent
bench/agents/aw/mobile_jev_base.py        Jev request / answer plumbing
bench/agents/aw/mobile_jev_policy/        operation rules, questions, state summary, action candidates
bench/agents/aw/convert.py                AndroidWorld accessibility forest -> agent state
bench/agents/aw/generation.py             text-model calls (TYPE_TEXT value, app name, answer)
bench/online/androidworld/                adapter (agent <-> env), loop rules, suite runner
bench/providers/                          System One / chat HTTP clients (+ offline mocks)
experiments/androidworld_state.py         capture / restore / verify the emulator's set-up state
configs/agents.yaml, providers.yaml       agent params and model endpoints
results/                                  summary.json, tasks.jsonl, config.json of the runs above
```

## Setup
1. **AndroidWorld + emulator** — follow android_world's README (Pixel 6 AVD, API 33, named `AndroidWorldAvd`,
   `hw.keyboard=yes`), then clone it **into this folder** (the runner looks for `./android_world`):
   ```sh
   git clone https://github.com/google-research/android_world.git
   git -C android_world checkout 2655e27
   pip install -r android_world/requirements.txt && (cd android_world && python setup.py install)
   ```
2. **Dependencies** (Python ≥ 3.11): `pip install -r requirements.txt`
3. **Key**: `cp .env.example .env` and set `OPENROUTER_API_KEY` (never commit `.env`; it is in `.gitignore`).
   Check it with `python -m bench ping openrouter_jev114_img`.
4. **Emulator state**: start the emulator, run android_world's one-time app setup (`--perform_emulator_setup`), then
   save the set-up state once: `python experiments/androidworld_state.py capture`. Every later run restores it
   automatically. See [docs/RUN_GUIDE.md](docs/RUN_GUIDE.md).

## Run
```sh
emulator -avd AndroidWorldAvd -no-snapshot -port 5554 -grpc 8554          # separate terminal
python -m bench aw-run --agent mobile_jev_hybrid_aw --tasks ContactsAddContact --run-id try1   # one task
python -m bench aw-run --agent mobile_jev_hybrid_aw --run-id vision_full                        # 116 tasks (~1 h, ~$0.30)
python -m bench aw-run --agent mobile_jev_aw --provider decision=openrouter_jev114 --run-id text_full  # text-only reference
python -m bench aw-run --agent mobile_jev_hybrid_aw --provider decision=mock_jev --provider text=mock_text --run-id dry  # no API calls
```
Results go to `runs/aw/<run-id>/` (`summary.json`, `tasks.jsonl`, `steps.jsonl`, screenshots, raw trees); `runs/` is
git-ignored because it holds the full requests and screenshots. Re-running the same `--run-id` resumes.

Keep the emulator's clipboard sharing off (macOS: `defaults write com.android.Emulator set.clipboardSharing -bool
false`, then restart the emulator): otherwise the host clipboard shows up in the device's screen text and is sent to
the model APIs. `aw-run` refuses to start while it is on.

## Credits
- [droidrun/mobile-jev](https://github.com/droidrun/mobile-jev) (MIT, `395fc22`): the operation rules, question
  set, state summary and action candidates in `bench/agents/aw/mobile_jev_policy/` are adapted from its
  `scripts/mobile-agent/`.
- [xinwang-nwpu/jev-mobile](https://github.com/xinwang-nwpu/jev-mobile) (MIT, `0a263b6`): the TYPE_TEXT value prompt
  (`TEXT_VALUE` in `bench/agents/aw/generation.py`).
- [google-research/android_world](https://github.com/google-research/android_world) (Apache-2.0): benchmark, tasks,
  environment and scoring, used unmodified (not included).
- Jev / System One by TypeSafe, accessed through [OpenRouter](https://openrouter.ai).

## License
MIT — see [LICENSE](LICENSE).
