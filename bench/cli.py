"""python -m bench <command>

  aw-run    --agent mobile_jev_aw [--tasks A,B] online AndroidWorld run (emulator running; device state is restored automatically)
  list                                      registered agents / providers
  ping      openrouter_jev                  one tiny request to check a provider/key
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .core.observation import DEFAULT_WINDOW_MODE, WINDOW_MODES
from .core.registry import PROJECT, load_env, load_yaml

RUNS_DIR = PROJECT / "runs"


def parse_pairs(items: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items or []:
        key, _, value = item.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")

    p = sub.add_parser("aw-run", help="online AndroidWorld run (emulator must be running; restores the device state first)")
    p.add_argument("--agent", required=True)
    p.add_argument("--tasks", help="comma-separated AndroidWorld task names (default: all)")
    p.add_argument("--n-task-combinations", type=int, default=1)
    p.add_argument("--seed", type=int, default=30)
    p.add_argument("--max-tasks", type=int, help="keep only the first N task templates of the suite")
    p.add_argument("--windows", choices=WINDOW_MODES, default=DEFAULT_WINDOW_MODE)
    p.add_argument("--loop-rules", choices=("default", "upstream", "off"), default="default",
                   help="default: upstream rules with --repeat-limit; upstream: the agent's original rules exactly; off: none")
    p.add_argument("--repeat-limit", type=int, default=3, help="how many times the same action on the same screen may run before the episode is stopped (default mode)")
    p.add_argument("--keep-keyboard-covered", action="store_true", help="do not remove elements that lie under the on-screen keyboard")
    p.add_argument("--screen-wait", choices=("pause", "stable"), default="pause",
                   help="how the screen is read after an action. pause: sleep --pause-seconds, read once (AndroidWorld default); "
                        "stable: AndroidWorld's wait_to_stabilize (re-read every 0.5 s until 3 equal reads, at most 6 s)")
    p.add_argument("--pause-seconds", type=float, default=1.0, help="the fixed pause for --screen-wait pause")
    p.add_argument("--provider", action="append", metavar="SLOT=NAME", help="override a provider slot")
    p.add_argument("--param", action="append", metavar="KEY=VALUE", help="override an agent param (JSON value)")
    p.add_argument("--console-port", type=int, default=5554, help="emulator console port: the device is emulator-<port> (second emulator: 5556)")
    p.add_argument("--grpc-port", type=int, default=8554, help="that emulator's -grpc port (second emulator: 8556)")
    p.add_argument("--run-id", help="resume/extend an existing run id")
    p.add_argument("--no-screenshots", action="store_true")
    p.add_argument("--no-forest", action="store_true", help="do not store the raw accessibility forest per step")
    p.add_argument("--no-raw", action="store_true", help="do not store raw requests/responses")
    p.add_argument("--skip-state-check", action="store_true", help="do not require `androidworld_state.py verify` to pass")
    p.add_argument("--no-prepare", action="store_true", help="do not restore the device before the run (default: restore golden state + OsmAnd map, then verify)")
    p.add_argument("--allow-clipboard-sharing", action="store_true", help="run even with the emulator's clipboard sharing on (default: refuse, the host clipboard would leak into the screen text)")

    p = sub.add_parser("ping", help="send one tiny request to a provider")
    p.add_argument("provider")

    args = parser.parse_args(argv)
    load_env()

    if args.command == "aw-run":
        from datetime import datetime

        from .core.types import RunAborted
        from .online.androidworld.runner import run as aw_run

        run_id = args.run_id or "%s__%s__aw" % (datetime.now().strftime("%Y%m%d-%H%M%S"), args.agent)
        try:
            summary = aw_run(
                args.agent,
                run_id,
                tasks=[t for t in (args.tasks or "").split(",") if t] or None,
                n_task_combinations=args.n_task_combinations,
                seed=args.seed,
                max_tasks=args.max_tasks,
                window_mode=args.windows,
                loop_rules=args.loop_rules,
                repeat_limit=args.repeat_limit,
                hide_keyboard_covered=not args.keep_keyboard_covered,
                screen_wait=args.screen_wait,
                pause_seconds=args.pause_seconds,
                params=parse_pairs(args.param),
                provider_overrides=parse_pairs(args.provider),
                console_port=args.console_port,
                grpc_port=args.grpc_port,
                screenshots=not args.no_screenshots,
                save_forest=not args.no_forest,
                save_raw=not args.no_raw,
                skip_state_check=args.skip_state_check,
                prepare=not args.no_prepare,
                allow_clipboard_sharing=args.allow_clipboard_sharing,
            )
        except RunAborted as aborted:
            print("\nrun aborted: %s\nresume with: python -m bench aw-run --agent %s --run-id %s" % (aborted, args.agent, run_id))
            return 2
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("\nsaved: %s" % (RUNS_DIR / "aw" / run_id))
    elif args.command == "ping":
        from .core.registry import make_provider

        provider = make_provider(args.provider)
        if provider.kind == "systemone":
            body = {"state": "The payment page shows an error.", "questions": {"q": {"type": "noul", "instructions": "Is something broken?"}}}
        else:
            body = {"messages": [{"role": "user", "content": "Reply with the word ok."}], "max_tokens": 5}
        response, meta = provider.post(body)
        print(json.dumps({"meta": meta, "response": response}, indent=2, ensure_ascii=False)[:2000])
    elif args.command == "list":
        for name, title in (("agents.yaml", "agents"), ("providers.yaml", "providers")):
            print("%s:" % title)
            for key, spec in load_yaml(name).items():
                detail = spec.get("class") or spec.get("model")
                slots = " ".join("%s=%s" % kv for kv in spec.get("providers", {}).items())
                print("  %-26s %s %s" % (key, detail, slots))
    return 0


if __name__ == "__main__":
    sys.exit(main())
