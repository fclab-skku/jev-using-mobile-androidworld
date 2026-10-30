# Run guide

## Before the first run
1. Create the AVD as in android_world's README (Pixel 6, API 33, `AndroidWorldAvd`). With `avdmanager`, set
   `hw.keyboard=yes` in the AVD's `config.ini` (Android Studio's default; avdmanager writes `no`).
2. Start it with fixed ports: `emulator -avd AndroidWorldAvd -no-snapshot -port 5554 -grpc 8554`.
3. Run android_world's app setup once (`python android_world/run.py --perform_emulator_setup ...`).
4. Save the set-up state: `python experiments/androidworld_state.py capture`.
   This keeps a golden copy of android_world's per-app snapshots in `/data/local/tmp` (the originals are deleted at
   boot on this image) and records the expected permissions, app-ops and screens in
   `configs/androidworld_golden_state.json`. The file in this repo is from our device; `capture` replaces it.

## Every run
`aw-run` prepares the device itself: `androidworld_state.py restore` (snapshots, permissions, app-ops, SMS role),
puts the OsmAnd map back if a previous run deleted it, and checks `verify` is 20/20. If not, it stops.
Manual commands:
```sh
python experiments/androidworld_state.py restore
python experiments/androidworld_state.py verify --screens     # also opens every app and compares its first screen
```

## Two emulators
A second AVD (`AndroidWorldAvd2`, ports 5556/8556) can run another suite in parallel:
`python -m bench aw-run ... --console-port 5556 --grpc-port 8556`. Both runs share one adb server; an adb server
restart errors the tasks running at that moment (seen once).

## Known effects
| Symptom | Cause / action |
|---|---|
| `restore: 19/20 … BAD file OsmAnd map` | a full run deletes the map (android_world behaviour); `aw-run` copies it back |
| `verify --screens` shows OsmAnd's "anonymous usage data" dialog | the device clock went back to the real date after a reboot. Tasks set the clock to 2023-10-15 themselves, so runs are not affected |
| `verify --screens` BAD on one app with another app's text | the previous app's screen was still showing; run it again |
| run stops with a clipboard message | turn off clipboard sharing (macOS: `defaults write com.android.Emulator set.clipboardSharing -bool false`) and restart the emulator |
| gRPC hangs in adb children (macOS arm64) | `bench/online/androidworld/__init__.py` sets `GRPC_ENABLE_FORK_SUPPORT=0`; set it yourself for other scripts that import android_world |

Shut the emulator down with `adb -s emulator-5554 shell svc power shutdown` (not `adb emu kill`).

## Output (`runs/aw/<run-id>/`)
| File | Content |
|---|---|
| `summary.json` | success, steps, end reasons, token usage and cost, timing |
| `tasks.jsonl` | one line per task (android_world's result + episode length) |
| `steps.jsonl` | every step: observation summary, the Jev request and response, the executed action |
| `screenshots/`, `forest/` | per-step screenshot and raw accessibility forest |
| `config.json` | agent params and models, suite options, android_world commit, device |

`end_reasons`: `done` / `answer` / `blocked` (the agent finished), `stuck` / `loading_timeout` (loop rules),
`budget_exhausted` (android_world's step budget).
