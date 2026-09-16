"""Driving the daemon: record, reload, transcribe, type.

Lifted out of cli.py, which had grown to 1,949 lines and four
responsibilities. Nothing here changed on the way across.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .. import asr, config, ipc
from ..term import DIM, GREEN, RED, RESET, YELLOW
from ._support import load_wav, log, setup_logging
from .health import _print_entry


def cmd_daemon(args: argparse.Namespace) -> int:
    cfg = config.load()
    setup_logging(args.log_level or cfg["ui"].get("log_level", "INFO"), to_file=True)
    # The one command that runs the resident side, and so the one that needs
    # it imported. Above the try because the except clause names it: an
    # import failure here should say what failed to import, not NameError.
    from .. import daemon

    try:
        daemon.Daemon(cfg).run()
    except daemon.AlreadyRunning as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1
    except asr.NotReady as exc:
        # EX_CONFIG. The unit sets RestartPreventExitStatus=78, so a machine
        # that is simply not set up yet reports one line instead of five stack
        # traces and a start-limit — and the first-run screen can read it.
        print(f"{RED}not ready:{RESET} {exc}", file=sys.stderr)
        log.error("not ready: %s", exc)
        return 78
    except KeyboardInterrupt:
        pass
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    try:
        reply = ipc.request({"cmd": args.action})
    except (ConnectionError, OSError) as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1
    if not reply.get("ok"):
        print(f"{RED}{reply.get('error', reply)}{RESET}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(reply, ensure_ascii=False))
    return 0


def cmd_reload(args: argparse.Namespace) -> int:
    try:
        reply = ipc.request({"cmd": "reload"})
    except (ConnectionError, OSError) as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1
    if not reply.get("ok"):
        print(f"{RED}{reply.get('error')}{RESET}", file=sys.stderr)
        return 1
    print(f"{GREEN}config reloaded{RESET}")
    if reply.get("model_restart_required"):
        print(f"{YELLOW}model settings changed; restart the daemon{RESET}")
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    from .. import asr
    from ..audio import Capture
    from ..pipeline import Pipeline

    cfg = config.load()
    setup_logging(args.log_level or "WARNING")
    samples, rate = load_wav(Path(args.file), cfg["audio"]["rate"])

    backend = asr.build(cfg)
    backend.load()
    try:
        capture = Capture(samples, rate, 0.0, 0.0, False)
        entry = Pipeline(cfg, backend).process(
            capture, inject=args.inject, forced_mode=args.mode)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    if args.json:
        print(json.dumps(entry, ensure_ascii=False, indent=2))
    else:
        _print_entry(entry, verbose=args.verbose)
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    """Put known text into the focused window, without saying anything.

    Injection failures are hard to pin down from dictation: you cannot tell a
    bad transcript from a paste that never landed. This does only the last
    step, with text you chose, so the question becomes one thing at a time.
    """
    import time as _time

    from ..inject import Injector
    from ..window import active_window

    cfg = config.load()
    if args.method:
        cfg["inject"]["method"] = args.method
    if args.paste_via:
        cfg["inject"]["paste_method"] = args.paste_via

    text = args.text or "omavoi one line"
    if args.lines > 1:
        text = "\n".join(f"{text} {i + 1}" for i in range(args.lines))

    print(f"{DIM}focus the target window now — injecting in {args.delay}s{RESET}")
    _time.sleep(args.delay)

    if args.via_daemon:
        try:
            reply = ipc.request({"cmd": "inject", "text": text}, timeout=120)
        except (ConnectionError, OSError) as exc:
            print(f"{RED}{exc}{RESET}", file=sys.stderr)
            return 1
        inj = reply.get("inject", {})
        win_info = reply.get("window", {})
        print("  injected by  the daemon process")
        print(f"  window       {win_info.get('class', '?')}  "
              f"xwayland={win_info.get('xwayland')}")
        print(f"  route        {inj.get('method')}  paste_via={inj.get('paste_via') or '—'}")
        mark = f"{GREEN}ok{RESET}" if inj.get("ok") else f"{RED}failed{RESET}"
        print(f"  result       {mark}  {inj.get('error') or ''}")
        return 0 if inj.get("ok") else 1

    win = active_window()

    # XTEST delivers to whatever X11 thinks is focused. If the compositor has
    # not handed X11 focus to the client, keys land nowhere and every layer
    # above reports success.
    xfocus = "n/a"
    if shutil.which("xdotool"):
        env = dict(os.environ)
        env.setdefault("DISPLAY", ":0")
        probe = subprocess.run(["xdotool", "getwindowfocus", "getwindowname"],
                               capture_output=True, timeout=5, env=env, check=False)
        xfocus = (probe.stdout.decode("utf-8", "replace").strip()
                  or probe.stderr.decode("utf-8", "replace").strip() or "?")

    injector = Injector(cfg)
    profile: dict[str, Any] = {}
    if args.method:
        profile["inject"] = args.method
    result = injector.inject(text, win, profile)

    print(f"  window       {win.cls or '?'}  xwayland={win.xwayland}")
    print(f"  x11 focus    {xfocus}")
    print(f"  route        {result.method}"
          + (f" (fell back from {cfg['inject']['method']})" if result.fell_back else ""))
    print(f"  paste via    {cfg['inject'].get('paste_method', 'shortcut')}")
    print(f"  lines        {len(text.splitlines())}")
    mark = f"{GREEN}ok{RESET}" if result.ok else f"{RED}failed{RESET}"
    print(f"  result       {mark} in {result.seconds:.2f}s"
          + (f"  {result.error}" if result.error else ""))
    print(f"\n{DIM}omavoi reports what it did, not what the app accepted — "
          f"look at the window.{RESET}")
    return 0 if result.ok else 1
