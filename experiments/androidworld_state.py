"""Capture / restore / verify the "fully set up" AndroidWorld emulator state.

Why: after `--perform_emulator_setup`, two things do not survive an emulator restart
on this image (see docs/RUN_GUIDE.md):
  1. /data/data/android_world/snapshots (per-app snapshots every task restores) is
     deleted at boot by PackageManager ("Package android_world is unknown").
  2. appops (MANAGE_EXTERNAL_STORAGE, SYSTEM_ALERT_WINDOW) are written lazily and are
     lost if the emulator is killed before the write.
android_world code is left untouched; this script keeps a golden copy of the snapshots
in /data/local/tmp (survives boot) and re-applies the expected permission state.

usage:
  python experiments/androidworld_state.py capture   # once, right after a clean full setup (+ Contacts manual step)
  python experiments/androidworld_state.py restore   # after every emulator boot (then runs verify)
  python experiments/androidworld_state.py verify [--screens]
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

# Same gRPC guard as bench/online/androidworld: --screens loads android_world (docs/RUN_GUIDE.md).
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

ADB = os.path.expanduser('~/Library/Android/sdk/platform-tools/adb')
SNAPSHOTS = '/data/data/android_world/snapshots'
GOLDEN = '/data/local/tmp/android_world_golden/snapshots'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_JSON = os.path.join(ROOT, 'configs', 'androidworld_golden_state.json')

# Expected state, transcribed from android_world/env/setup_device/apps.py (commit 2655e27).
# grants: runtime permissions the setup grants (pm grant or UI "Allow").
GRANTS = {
    'com.android.camera2': ['ACCESS_COARSE_LOCATION'],
    'com.simplemobiletools.calendar.pro': ['READ_CALENDAR', 'WRITE_CALENDAR', 'POST_NOTIFICATIONS'],
    'com.simplemobiletools.gallery.pro': ['WRITE_EXTERNAL_STORAGE', 'ACCESS_MEDIA_LOCATION',
                                          'READ_MEDIA_IMAGES', 'READ_MEDIA_VIDEO', 'POST_NOTIFICATIONS'],
    'com.dimowner.audiorecorder': ['RECORD_AUDIO', 'POST_NOTIFICATIONS'],
    'net.osmand': ['POST_NOTIFICATIONS'],
    'de.dennisguse.opentracks': ['ACCESS_COARSE_LOCATION', 'ACCESS_FINE_LOCATION', 'POST_NOTIFICATIONS',
                                 'BLUETOOTH_CONNECT', 'BLUETOOTH_SCAN'],  # bluetooth = the UI "Allow"
    'org.videolan.vlc': ['POST_NOTIFICATIONS'],
    'net.cozic.joplin': ['ACCESS_COARSE_LOCATION', 'ACCESS_FINE_LOCATION'],
    'code.name.monkey.retromusic': ['READ_MEDIA_AUDIO', 'POST_NOTIFICATIONS'],
}
# Contacts: setup taps "Don't allow"; a single denial re-prompts on every launch, so the
# settled state is denied + USER_FIXED (docs/RUN_GUIDE.md §5.6).
DENIED_FIXED = {'com.google.android.contacts': ['POST_NOTIFICATIONS']}
APPOPS = {  # package: [(op, set --uid?)]
    'net.gsantner.markor': [('MANAGE_EXTERNAL_STORAGE', True)],
    'org.videolan.vlc': [('MANAGE_EXTERNAL_STORAGE', True)],
    'com.simplemobiletools.gallery.pro': [('MANAGE_EXTERNAL_STORAGE', True)],
    'com.example.androidworld': [('SYSTEM_ALERT_WINDOW', False)],
}
SMS_HOLDER = 'com.simplemobiletools.smsmessenger'
FILES = [  # (path, what)
    ('/storage/emulated/0/Android/data/net.osmand/files/Liechtenstein_europe.obf', 'OsmAnd map'),
    ('/storage/emulated/0/VLCVideos', 'VLC videos dir'),
]
A11Y = 'com.google.androidenv.accessibilityforwarder/com.google.androidenv.accessibilityforwarder.AccessibilityForwarder'
# 24 apps in setup._APPS -> one snapshot dir each.
APP_PACKAGES = [
    'com.example.androidworld', 'com.dimowner.audiorecorder', 'com.android.camera2', 'com.android.chrome',
    'ca.zgrs.clipper', 'com.google.android.deskclock', 'com.google.android.contacts', 'com.google.android.dialer',
    'com.arduia.expense', 'com.google.android.documentsui', 'net.cozic.joplin', 'net.gsantner.markor',
    'com.google.androidenv.miniwob', 'de.dennisguse.opentracks', 'net.osmand', 'com.flauschcode.broccoli',
    'code.name.monkey.retromusic', 'com.android.settings', 'com.simplemobiletools.calendar.pro',
    'com.simplemobiletools.draw.pro', 'com.simplemobiletools.gallery.pro', 'com.simplemobiletools.smsmessenger',
    'org.tasks', 'org.videolan.vlc',
]
SCREEN_FLAG = re.compile(r'allow|permission|welcome|get started|next|skip|continue|accept|sign in|'
                         r'keeps stopping|not responding|isn.t responding|update|got it|agree', re.I)


# Target device (set from --console-port / --grpc-port; defaults = the first emulator, as before).
CONSOLE_PORT = 5554
GRPC_PORT = 8554


def _adb(*args):
  """adb pinned to the target emulator, so a second running emulator is never touched."""
  return [ADB, '-s', f'emulator-{CONSOLE_PORT}', *args]


def sh(cmd):
  return subprocess.run(_adb('shell', cmd), capture_output=True, text=True).stdout.replace('\r', '')


def ensure_root():
  if sh('whoami').strip() != 'root':
    subprocess.run(_adb('root'), capture_output=True)
    time.sleep(2)
    subprocess.run(_adb('wait-for-device'))


def perm_state(pkg):
  out, st = sh(f'dumpsys package {pkg}'), {}
  for m in re.finditer(r'android\.permission\.([A-Z_]+): granted=(true|false)(?:, flags=\[([^\]]*)\])?', out):
    st.setdefault(m.group(1), (m.group(2) == 'true', m.group(3) or ''))
  return st


def appop_allowed(pkg, op):
  out = sh(f'appops get {pkg} {op}')
  return bool(re.search(rf'{op}: allow', out))


def dir_digest(path):
  out = sh(f'cd {path} 2>/dev/null && find . -type f -exec md5sum {{}} + | sort | md5sum').split()
  return out[0] if out else None


def check_state(golden=None):
  """Returns list of (item, ok, detail)."""
  rows = []
  for pkg, perms in GRANTS.items():
    st = perm_state(pkg)
    bad = [p for p in perms if not st.get(p, (False,))[0]]
    rows.append((f'grant {pkg}', not bad, 'missing: ' + ','.join(bad) if bad else 'ok'))
  for pkg, perms in DENIED_FIXED.items():
    st = perm_state(pkg)
    bad = [p for p in perms if st.get(p, (True, ''))[0] or 'USER_FIXED' not in st.get(p, (True, ''))[1]]
    rows.append((f'denied+fixed {pkg}', not bad, 'not fixed: ' + ','.join(bad) if bad else 'ok'))
  for pkg, ops in APPOPS.items():
    for op, _ in ops:
      ok = appop_allowed(pkg, op)
      rows.append((f'appop {pkg} {op}', ok, 'allow' if ok else sh(f'appops get {pkg} {op}').strip()[:60]))
  holder = re.search(r'name=android\.app\.role\.SMS\s+holders=(\S*)', sh('dumpsys role'))
  rows.append(('role SMS', bool(holder and holder.group(1) == SMS_HOLDER), holder.group(1) if holder else '?'))
  for path, what in FILES:
    ok = sh(f'ls -d {path} 2>/dev/null').strip() == path
    rows.append((f'file {what}', ok, path))
  rows.append(('a11y forwarder enabled', A11Y in sh('settings get secure enabled_accessibility_services'), ''))
  present = set(sh(f'ls {SNAPSHOTS} 2>/dev/null').split())
  missing = [p for p in APP_PACKAGES if p not in present]
  rows.append(('snapshots present (24)', not missing, 'missing: ' + ','.join(missing) if missing else 'ok'))
  if golden and not missing:
    diff = [p for p in APP_PACKAGES if dir_digest(f'{SNAPSHOTS}/{p}') != golden['snapshot_digest'].get(p)]
    rows.append(('snapshots == golden', not diff, 'differ: ' + ','.join(diff) if diff else 'ok'))
  return rows


def screens():
  """Restore each app's snapshot (as a task does) and read its first screen."""
  from android_world.env import adb_utils, env_launcher
  from android_world.env.setup_device import setup
  from android_world.utils import app_snapshot
  env = env_launcher.load_and_setup_env(console_port=CONSOLE_PORT, emulator_setup=False, adb_path=ADB, grpc_port=GRPC_PORT)
  c, res = env.controller, {}
  def restore(name):
    # An empty app data dir (e.g. Settings, whose snapshot is empty) makes the
    # `rm -r dir/*` inside restore_snapshot fail; the data is already "restored".
    pkg = adb_utils.extract_package_name(adb_utils.get_adb_activity(name))
    if not sh(f'ls -A /data/data/{pkg}').strip():
      sh(f'cp -a {SNAPSHOTS}/{pkg}/. /data/data/{pkg}/')
      return
    app_snapshot.restore_snapshot(name, c)

  for app in setup._APPS:
    try:
      restore(app.app_name)
    except Exception as e:  # pylint: disable=broad-except
      res[app.app_name] = {'error': f'{type(e).__name__}: {str(e).splitlines()[0]}'}
      continue
    pkg = adb_utils.extract_package_name(adb_utils.get_adb_activity(app.app_name))
    adb_utils.launch_app(app.app_name, c)
    time.sleep(4)
    texts = [e.text or e.content_description for e in c.get_ui_elements()
             if (e.text or e.content_description) and e.package_name != 'com.android.systemui']
    fg = re.search(r'mCurrentFocus=Window\{\S+ \S+ ([^/}\s]+)', sh('dumpsys window'))
    res[app.app_name] = {'package': pkg, 'foreground': fg.group(1) if fg else '?',
                         'flagged': sorted({t for t in texts if SCREEN_FLAG.search(t)}), 'texts': texts[:8]}
    adb_utils.close_app(app.app_name, c)
    restore(app.app_name)  # leave app data exactly at the snapshot
  adb_utils.press_home_button(c)
  env.close()
  return res


def compare_screens(now, golden):
  rows = []
  for name, g in golden.items():
    n = now.get(name, {})
    if 'error' in n:
      rows.append((f'screen {name}', False, n['error']))
      continue
    new_flags = sorted(set(n['flagged']) - set(g['flagged']))
    ok = n['foreground'] == g['foreground'] and not new_flags
    rows.append((f'screen {name}', ok, 'ok' if ok else f"fg={n['foreground']} new={new_flags}"))
  return rows


def report(rows):
  for item, ok, detail in rows:
    print(f"{'OK ' if ok else 'BAD'}  {item:62} {detail}")
  bad = sum(not ok for _, ok, _ in rows)
  print(f'\n{len(rows) - bad}/{len(rows)} OK' + ('' if not bad else f' — {bad} BAD'))
  return bad == 0


def cmd_capture(_):
  ensure_root()
  sh('appops write-settings')  # persist setup's appops now instead of up to 30 min later
  rows = check_state()
  if not report(rows):
    sys.exit('state is not complete; fix the BAD rows before capturing')
  sh(f'rm -rf {GOLDEN} && mkdir -p {GOLDEN} && cp -a {SNAPSHOTS}/. {GOLDEN}/ && sync')
  digest = {p: dir_digest(f'{SNAPSHOTS}/{p}') for p in APP_PACKAGES}
  gdig = {p: dir_digest(f'{GOLDEN}/{p}') for p in APP_PACKAGES}
  assert digest == gdig, 'golden copy mismatch'
  golden = {'captured_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'build': sh('getprop ro.build.fingerprint').strip(), 'snapshot_digest': digest,
            'screens': screens()}
  json.dump(golden, open(GOLDEN_JSON, 'w'), indent=1, ensure_ascii=False)
  print(f'golden snapshots -> {GOLDEN}\nspec -> {GOLDEN_JSON}')


def cmd_restore(args):
  ensure_root()
  golden = json.load(open(GOLDEN_JSON))
  if sh(f'ls {GOLDEN} 2>/dev/null').split() == []:
    sys.exit(f'{GOLDEN} missing on device — full setup + capture required (see guide)')
  # 1) snapshots from golden
  sh(f'rm -rf {SNAPSHOTS} && mkdir -p {SNAPSHOTS} && cp -a {GOLDEN}/. {SNAPSHOTS}/ && sync')
  # 2) permissions / appops / role
  for pkg, perms in GRANTS.items():
    for p in perms:
      sh(f'pm grant {pkg} android.permission.{p}')
  for pkg, perms in DENIED_FIXED.items():
    for p in perms:
      sh(f'pm revoke {pkg} android.permission.{p}; pm set-permission-flags {pkg} android.permission.{p} user-set user-fixed')
  for pkg, ops in APPOPS.items():
    for op, uid in ops:
      sh(f"appops set {'--uid ' if uid else ''}{pkg} {op} allow")
  sh(f'cmd role add-role-holder android.app.role.SMS {SMS_HOLDER}')
  sh('appops write-settings')  # flush appops to /data/system/appops.xml now
  print('restored; verifying\n')
  cmd_verify(args)


def cmd_verify(args):
  ensure_root()
  golden = json.load(open(GOLDEN_JSON)) if os.path.exists(GOLDEN_JSON) else None
  rows = check_state(golden)
  if getattr(args, 'screens', False) and golden:
    rows += compare_screens(screens(), golden['screens'])
  sys.exit(0 if report(rows) else 1)


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('--console-port', type=int, default=5554, help='emulator console port (device emulator-<port>)')
  ap.add_argument('--grpc-port', type=int, default=8554, help='emulator gRPC port (used by --screens)')
  sub = ap.add_subparsers(dest='cmd', required=True)
  sub.add_parser('capture')
  sub.add_parser('restore').add_argument('--screens', action='store_true')
  sub.add_parser('verify').add_argument('--screens', action='store_true')
  a = ap.parse_args()
  CONSOLE_PORT, GRPC_PORT = a.console_port, a.grpc_port
  {'capture': cmd_capture, 'restore': cmd_restore, 'verify': cmd_verify}[a.cmd](a)
