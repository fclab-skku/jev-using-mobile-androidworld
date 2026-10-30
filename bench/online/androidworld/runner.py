"""Online AndroidWorld run: one agent over an AndroidWorld task suite.

Uses android_world's own suite runner unchanged (suite_utils.create_suite / suite_utils.run with an
IncrementalCheckpointer, so an interrupted run resumes at the last unfinished task template).
The emulator must already be running (-grpc 8554). Before the suite starts the runner prepares the device by itself
(disable with prepare=False / --no-prepare): `experiments/androidworld_state.py restore`, then the OsmAnd map is copied
back if a previous full run deleted it, then `verify` must report every item OK (docs/RUN_GUIDE.md).

runs/aw/<run_id>/
  config.json       agent (describe + params), suite options, android_world commit, device apps
  steps.jsonl       one row per AndroidWorld step: observation summary, prediction, tags, loop-rule events,
                    executed actions, usage, raw requests/responses
  tasks.jsonl       one row per episode (AndroidWorld's result: success, length, time, exception)
  summary.json      success rate, totals and timing
  sessions.jsonl    one row per invocation (start, end, seconds); a resumed run has several
  screenshots/      e<episode>_s<step>.png
  forest/           e<episode>_s<step>.pb.gz  raw AndroidAccessibilityForest proto (bench.data.a11y_proto.decode_forest)
  aw_checkpoint/    android_world's checkpoint (resume)
"""

import datetime
import json
import os
import re
import tempfile
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.registry import PROJECT, load_env, make_agent
from ...core.types import RunAborted
from .adapter import AwAdapter


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""

ADB_PATH = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
AW_ROOT = PROJECT / "android_world"
STATE_TOOL = PROJECT / "experiments" / "androidworld_state.py"


def _last_episode(run_dir: Path) -> int:
    """Highest episode number already recorded (steps.jsonl rows, screenshot/forest file names); -1 for a new run."""
    last = -1
    if (run_dir / "steps.jsonl").exists():
        for line in open(run_dir / "steps.jsonl"):
            try:
                last = max(last, int(json.loads(line).get("episode", -1)))
            except (ValueError, TypeError):
                continue
    for sub in ("screenshots", "forest"):
        for p in (run_dir / sub).glob("e*_s*"):
            try:
                last = max(last, int(p.name[1:p.name.index("_")]))
            except ValueError:
                continue
    return last


def _first_created(run_dir: Path) -> Optional[str]:
    """Keep the original start time when a run is resumed."""
    try:
        return json.loads((run_dir / "config.json").read_text()).get("created")
    except (OSError, ValueError):
        return None


OSMAND_MAP = "/storage/emulated/0/Android/data/net.osmand/files/Liechtenstein_europe.obf"


def _state_tool(command: str, console_port: int = 5554, grpc_port: int = 8554) -> subprocess.CompletedProcess:
    args = [sys.executable, str(STATE_TOOL), "--console-port", str(console_port), "--grpc-port", str(grpc_port), command]
    return subprocess.run(args, capture_output=True, text=True)


def verify_device_state(console_port: int = 5554, grpc_port: int = 8554) -> bool:
    """experiments/androidworld_state.py verify on that emulator (exit 0 = every item OK)."""
    proc = _state_tool("verify", console_port, grpc_port)
    sys.stdout.write(proc.stdout[-2000:])
    return proc.returncode == 0


def restore_device_state(console_port: int = 5554, grpc_port: int = 8554) -> str:
    """experiments/androidworld_state.py restore: golden app snapshots + permissions/appops/SMS role.
    Its own verify may still report the OsmAnd map as missing; the map is handled by restore_osmand_map()."""
    proc = _state_tool("restore", console_port, grpc_port)
    lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "(no output) %s" % proc.stderr.strip()[-200:]


def host_clipboard_sharing() -> Optional[bool]:
    """The emulator's "Enable clipboard sharing" setting on this Mac (`defaults` domain com.android.Emulator, key
    set.clipboardSharing; read when an emulator starts). None when it cannot be read (not macOS / never set).
    With sharing on, text copied on the Mac reaches the device clipboard and the device's reaches the Mac (AW-22)."""
    try:
        proc = subprocess.run(["defaults", "read", "com.android.Emulator", "set.clipboardSharing"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    value = proc.stdout.strip().lower()
    if proc.returncode != 0 or value not in ("0", "1", "true", "false"):
        return None
    return value in ("1", "true")


def device_identity(console_port: int) -> Dict[str, Any]:
    """Serial, adb state and AVD name of the emulator this run will drive (recorded in config.json)."""
    serial = "emulator-%d" % console_port
    state = subprocess.run([ADB_PATH, "-s", serial, "get-state"], capture_output=True, text=True).stdout.strip()
    avd = subprocess.run([ADB_PATH, "-s", serial, "emu", "avd", "name"], capture_output=True, text=True).stdout.strip().splitlines()
    return {"serial": serial, "state": state, "avd": avd[0].strip() if avd else ""}


class DeviceLock:
    """One aw-run per emulator: a second run on the same device would corrupt both. The lock file holds the owner's
    pid; a lock whose process is gone is taken over."""

    def __init__(self, runs_dir: Path, serial: str, run_id: str):
        self.path, self.run_id = runs_dir / ".locks" / ("%s.lock" % serial), run_id

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                owner = json.loads(self.path.read_text())
                os.kill(int(owner["pid"]), 0)  # raises if that process no longer exists
            except (OSError, ValueError, KeyError):
                owner = None
            if owner:
                raise SystemExit("%s is already used by run %r (pid %s); one aw-run per emulator" % (self.path.stem, owner.get("run_id"), owner.get("pid")))
        self.path.write_text(json.dumps({"pid": os.getpid(), "run_id": self.run_id, "since": datetime.datetime.now().isoformat(timespec="seconds")}))
        return self

    def __exit__(self, *exc):
        try:
            if json.loads(self.path.read_text()).get("pid") == os.getpid():
                self.path.unlink()
        except (OSError, ValueError):
            pass


def osmand_map_present(console_port: int) -> bool:
    out = subprocess.run([ADB_PATH, "-s", "emulator-%d" % console_port, "shell", "ls", OSMAND_MAP], capture_output=True, text=True).stdout
    return out.strip() == OSMAND_MAP


def restore_osmand_map(env) -> None:
    """Put the OsmAnd map back: only the copy + security-context steps of android_world's OsmAndApp.setup (the full
    setup clears the app's data). A full run deletes the map: several task initializers call clear_internal_storage,
    which removes every file under /storage/emulated/0 (android_world's own behaviour)."""
    from android_world.env import adb_utils
    from android_world.env.setup_device import apps
    from android_world.utils import file_utils

    app = apps.OsmAndApp
    app._copy_data_to_device(app.MAP_NAMES, app.DEVICE_MAPS_PATH, env)
    for name in app.MAP_NAMES:
        adb_utils.check_ok(
            adb_utils.issue_generic_request(
                ["shell", "chcon", "u:object_r:media_rw_data_file:s0", file_utils.convert_to_posix_path(app.DEVICE_MAPS_PATH, name)],
                env.controller,
            )
        )


def end_reasons(rows: List[Dict[str, Any]]) -> Dict[Any, str]:
    """(task, instance) -> how its episode ended: the agent's done/answer/blocked, a loop-rule stop
    (stuck, loading_timeout, action_budget, agent_errors), or budget_exhausted (AndroidWorld's step budget ran
    out). Rows written before `end_reason` existed are inferred from the episode's last row."""
    last: Dict[Any, Dict[str, Any]] = {}
    for row in rows:
        last[(row.get("task"), row.get("instance"))] = row
    out = {}
    for key, row in last.items():
        if row.get("end_reason"):
            out[key] = row["end_reason"]
        elif row.get("done"):
            out[key] = row.get("stop_reason") or (row.get("pred") or {}).get("type")
        else:
            out[key] = "budget_exhausted"
    return out


def timing(run_dir: Path, rows: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wall time over all invocations (sessions.jsonl), per task (AndroidWorld run_time), per step and API wait."""
    sessions = [json.loads(line) for line in open(run_dir / "sessions.jsonl")] if (run_dir / "sessions.jsonl").exists() else []
    task_times = [r["run_time"] for r in results if isinstance(r.get("run_time"), (int, float))]
    step_times = [r["seconds"] for r in rows if isinstance(r.get("seconds"), (int, float))]
    api: Counter = Counter()
    for row in rows:
        for call in (row.get("raw") or {}).get("calls", []):
            api[call.get("slot", "?")] += (call.get("meta") or {}).get("latency_ms", 0.0) / 1000.0
    return {
        "started": sessions[0]["start"] if sessions else None,
        "finished": sessions[-1].get("end") if sessions else None,
        "sessions": len(sessions),
        "wall_seconds": round(sum(s.get("seconds") or 0.0 for s in sessions), 1),  # excludes gaps between resumed sessions
        "task_seconds_total": round(sum(task_times), 1),
        "task_seconds_mean": round(sum(task_times) / len(task_times), 1) if task_times else None,
        "step_seconds_total": round(sum(step_times), 1),
        "step_seconds_mean": round(sum(step_times) / len(step_times), 2) if step_times else None,
        "api_seconds": {slot: round(v, 1) for slot, v in api.items()},
    }


def summarize(run_dir: Path, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [json.loads(line) for line in open(run_dir / "steps.jsonl")] if (run_dir / "steps.jsonl").exists() else []
    tasks = [r for r in results if r.get("exception_info") is None]
    ok = sum(float(r.get("is_successful") or 0.0) > 0.5 for r in tasks)
    usage: Dict[str, Counter] = {}
    for row in rows:
        for slot, u in (row.get("usage") or {}).items():
            usage.setdefault(slot, Counter()).update({k: v for k, v in u.items() if isinstance(v, (int, float))})
    return {
        "tasks": len(results),
        "tasks_completed": len(tasks),
        "tasks_errored": len(results) - len(tasks),
        "success": ok,
        "success_rate": round(ok / len(tasks), 4) if tasks else None,  # AndroidWorld's convention: errored tasks excluded
        "success_rate_all": round(ok / len(results), 4) if results else None,  # errored tasks counted as failures
        "errored_tasks": [r.get("task_template") for r in results if r.get("exception_info") is not None],
        "steps": len(rows),
        "end_reasons": dict(Counter(end_reasons(rows).values())),
        "final_actions": dict(Counter((r.get("pred") or {}).get("type") for r in rows if r.get("done"))),
        "agent_errors": sum(1 for r in rows if r.get("error")),
        "usage": {k: dict(v) for k, v in usage.items()},
        "timing": timing(run_dir, rows, results),
    }


def run(
    agent_name: str,
    run_id: str,
    tasks: Optional[List[str]] = None,
    n_task_combinations: int = 1,
    seed: int = 30,
    max_tasks: Optional[int] = None,
    family: str = "android_world",
    window_mode: str = "no_statusbar",
    loop_rules: str = "default",
    repeat_limit: int = 3,
    hide_keyboard_covered: bool = True,
    screen_wait: str = "pause",
    pause_seconds: float = 1.0,
    params: Optional[Dict[str, Any]] = None,
    provider_overrides: Optional[Dict[str, str]] = None,
    console_port: int = 5554,
    grpc_port: int = 8554,
    screenshots: bool = True,
    save_forest: bool = True,
    save_raw: bool = True,
    skip_state_check: bool = False,
    prepare: bool = True,
    allow_clipboard_sharing: bool = False,
    runs_dir: Path = PROJECT / "runs" / "aw",  # AndroidWorld runs kept apart from AndroidControl runs (runs/)
) -> Dict[str, Any]:
    load_env()
    if host_clipboard_sharing() and not allow_clipboard_sharing:  # AW-22
        raise SystemExit(
            "Emulator clipboard sharing is on: the Mac clipboard can reach the device (and the screen text sent to the "
            "model APIs), and the device clipboard reaches the Mac (docs/EXTENSIONS.md AW-22). Turn it off with\n"
            "  defaults write com.android.Emulator set.clipboardSharing -bool false\n"
            "and restart the emulators (the setting is read at start), or pass --allow-clipboard-sharing."
        )
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Per-run host temp dir. android_world writes task files under tempfile.gettempdir() with fixed names
    # (user_data_generation._TMP) before pushing them to the device; two concurrent runs must not share that folder.
    # Has to happen before android_world's task modules are imported.
    tmp_dir = Path(tempfile.mkdtemp(prefix="aw_%s_" % re.sub(r"[^0-9A-Za-z_.-]", "_", run_id)[:40]))
    os.environ["TMPDIR"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)

    device = device_identity(console_port)
    if device["state"] != "device":
        raise SystemExit("emulator-%d is not running (adb state: %r). Start it with -port %d -grpc %d." % (console_port, device["state"], console_port, grpc_port))
    with DeviceLock(runs_dir, device["serial"], run_id):
        return _run_locked(
            agent_name, run_id, tasks, n_task_combinations, seed, max_tasks, family, window_mode, loop_rules, params, provider_overrides,
            console_port, grpc_port, screenshots, save_forest, save_raw, skip_state_check, prepare, run_dir, tmp_dir, device,
            repeat_limit, hide_keyboard_covered, screen_wait, pause_seconds,
        )


def transition_pause_for(screen_wait: str, pause_seconds: float) -> Optional[float]:
    """AndroidWorld's base agent: a number = sleep that long and read once; None = wait until the screen is stable."""
    if screen_wait not in ("pause", "stable"):
        raise ValueError("screen_wait must be pause or stable")
    if screen_wait == "pause" and pause_seconds < 0:
        raise ValueError("pause_seconds must be >= 0")
    return float(pause_seconds) if screen_wait == "pause" else None


def _run_locked(agent_name, run_id, tasks, n_task_combinations, seed, max_tasks, family, window_mode, loop_rules, params, provider_overrides,
                console_port, grpc_port, screenshots, save_forest, save_raw, skip_state_check, prepare, run_dir, tmp_dir, device,
                repeat_limit=3, hide_keyboard_covered=True, screen_wait="pause", pause_seconds=1.0) -> Dict[str, Any]:
    """The run itself, with the device lock held and the per-run temp dir in place."""
    from android_world import checkpointer as checkpointer_lib
    from android_world import registry as aw_registry
    from android_world import suite_utils
    from android_world.env import adb_utils, env_launcher

    # Automatic device preparation (prepare=True, the default): 1) restore the golden state, 2) put the OsmAnd map
    # back if a previous full run deleted it, 3) verify. With prepare=False only the verification runs.
    prepared: Dict[str, Any] = {"prepare": prepare, "restore": None, "osmand_map_copied": False, "verified": None}
    if prepare:
        prepared["restore"] = restore_device_state(console_port, grpc_port)
        print("[prepare] restore: %s" % prepared["restore"])

    session_start = datetime.datetime.now()
    agent = make_agent(agent_name, provider_overrides, params)
    env = env_launcher.load_and_setup_env(console_port=console_port, emulator_setup=False, adb_path=ADB_PATH, grpc_port=grpc_port)
    try:
        if prepare and not osmand_map_present(console_port):
            restore_osmand_map(env)
            prepared["osmand_map_copied"] = True
            print("[prepare] OsmAnd map was missing -> copied back")
        if not skip_state_check:
            prepared["verified"] = verify_device_state(console_port, grpc_port)
            if not prepared["verified"]:
                raise SystemExit(
                    "AndroidWorld device state is not the golden state after preparation; see the BAD rows above and "
                    "docs/RUN_GUIDE.md (or pass --skip-state-check)"
                )
        apps = sorted(adb_utils.get_all_apps(env.controller))
        # app name -> package, from android_world's own launch table (the names get_all_apps returns are its first aliases)
        packages = {k.split("|")[0]: v.split("/")[0] for k, v in adb_utils._PATTERN_TO_ACTIVITY.items()}  # pylint: disable=protected-access
        agent.setup({"device_apps": apps, "app_vocab": apps, "device_app_packages": {a: packages[a] for a in apps if a in packages}})

        task_registry = aw_registry.TaskRegistry()
        suite = suite_utils.create_suite(task_registry.get_registry(family), n_task_combinations=n_task_combinations, seed=seed, tasks=tasks)
        if max_tasks is not None:
            for name in list(suite)[max_tasks:]:
                del suite[name]
        lookup = {t.goal: {"name": name, "instance": i} for name, instances in suite.items() for i, t in enumerate(instances)}

        config = {
            "run_id": run_id,
            "created": _first_created(run_dir) or session_start.isoformat(timespec="seconds"),
            "benchmark": "androidworld",
            "android_world_commit": git_commit(AW_ROOT),
            "family": family,
            "tasks": list(suite),
            "n_task_combinations": n_task_combinations,
            "seed": seed,
            "window_mode": window_mode,
            "loop_rules": loop_rules,
            "repeat_limit": repeat_limit,
            "hide_keyboard_covered": hide_keyboard_covered,
            "screen_wait": screen_wait,  # pause: fixed sleep then one read | stable: AndroidWorld wait_to_stabilize
            "pause_seconds": pause_seconds if screen_wait == "pause" else None,
            "agent": agent.describe(),
            "device_apps": apps,
            "screen": list(env.logical_screen_size),
            "device_preparation": prepared,
            "device": {**device, "console_port": console_port, "grpc_port": grpc_port},
            "host_tmp_dir": str(tmp_dir),
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str))

        adapter = AwAdapter(
            env, agent, run_dir, window_mode=window_mode, loop_rules=loop_rules, screenshots=screenshots,
            save_forest=save_forest, save_raw=save_raw, repeat_limit=repeat_limit, hide_keyboard_covered=hide_keyboard_covered,
            transition_pause=transition_pause_for(screen_wait, pause_seconds), task_lookup=lambda goal: lookup.get(goal, {}), app_vocab=apps,
        )
        adapter.episode_index = _last_episode(run_dir)  # resumed run: keep numbering so earlier screenshots/forests are not overwritten
        for name, instances in suite.items():  # tell the adapter which task is starting (instance-level wrap, AW untouched)
            for i, task in enumerate(instances):
                def initialize(env_, _orig=task.initialize_task, _name=name, _i=i):
                    adapter.set_task(_name, _i)
                    return _orig(env_)

                task.initialize_task = initialize
        try:
            results = suite_utils.run(suite, adapter, checkpointer=checkpointer_lib.IncrementalCheckpointer(str(run_dir / "aw_checkpoint")), demo_mode=False)
        except RunAborted as aborted:  # the agent stopped the run (e.g. world-model server unreachable)
            with open(run_dir / "aborted.jsonl", "a") as f:
                f.write(json.dumps({"ts": datetime.datetime.now().isoformat(timespec="seconds"), "reason": str(aborted), "task": adapter.current_task, "episode": adapter.episode_index}, ensure_ascii=False) + "\n")
            raise
    finally:
        env.close()
        end = datetime.datetime.now()
        with open(run_dir / "sessions.jsonl", "a") as f:
            f.write(json.dumps({"start": session_start.isoformat(timespec="seconds"), "end": end.isoformat(timespec="seconds"), "seconds": round((end - session_start).total_seconds(), 1)}) + "\n")

    keep = ("task_template", "instance_id", "goal", "is_successful", "episode_length", "run_time", "finish_dtime", "exception_info", "seed")
    task_rows = [{k: (r.get(k) if not isinstance(r.get(k), float) or r.get(k) == r.get(k) else None) for k in keep} for r in results]
    with open(run_dir / "tasks.jsonl", "w") as f:
        for row in task_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    summary = summarize(run_dir, task_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
