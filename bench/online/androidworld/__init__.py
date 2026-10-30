"""AndroidWorld integration: adapter (agent <-> env), upstream loop rules, suite runner."""

import os

# gRPC (grpcio >= 1.83 on macOS arm64, grpc/grpc#43062) registers pthread_atfork handlers when its fork support is on
# (the build default). They run inside every child that android_env forks for an adb command, before exec:
#   - the child's restarted poller logs "FD from fork parent still in poll list"; android_env reads adb output with
#     stderr merged in, so the line lands in command output (443 of 4,109 calls) and android_world's
#     file_utils.clear_directory takes an empty folder for a non-empty one -> `rm -r dir/*` fails -> the suite stops;
#   - the handler can block on a lock owned by a thread that does not exist in the child -> the child never execs and
#     the parent waits forever in subprocess._execute_child (4 hangs in 4,109 calls; a 116-task run cannot finish).
# Our children only exec adb and never use gRPC, so the handlers are not needed: with fork support off,
# 23,405 calls gave 0 hangs and 0 polluted outputs (2026-10-01). GRPC_VERBOSITY=ERROR stays as a second guard against
# stray gRPC INFO lines in adb output. Both must be set before grpc is imported; explicit values are kept.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
