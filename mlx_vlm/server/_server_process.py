# Copyright © 2026 Neywa Labs Inc.

"""CLI process owner. Keep this executable stdlib-only, outside the MLX runtime."""

import os
import signal
import subprocess
import sys
import time


def supervise(command, shutdown_timeout):
    deadline = None
    stopping_signal = None
    forwarded = False
    child = None

    def stop(signum, _frame):
        nonlocal deadline, stopping_signal
        # A second signal explicitly skips the remaining grace period.
        deadline = time.monotonic() + (
            shutdown_timeout if stopping_signal is None else 0
        )
        stopping_signal = signum

    handlers = {
        sig: signal.signal(sig, stop)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        child = subprocess.Popen(command, start_new_session=True)
        while True:
            if stopping_signal is not None and not forwarded:
                # Uvicorn handles TERM gracefully; HUP is also a stop request here.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                forwarded = True
            result = child.poll()
            if result is not None:
                return result if result >= 0 else 128 - result
            if deadline is not None and time.monotonic() >= deadline:
                print(
                    "MLX-VLM shutdown deadline reached; terminating the owned "
                    "server process group.",
                    file=sys.stderr,
                    flush=True,
                )
                return 124
            time.sleep(0.02)
    finally:
        if child is not None:
            # Includes descendants left by a reloader or a crashed server. Never
            # signal the caller's process group or another application process.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(supervise(sys.argv[2:], float(sys.argv[1])))
