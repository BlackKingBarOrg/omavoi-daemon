"""The push-to-talk key, and the secrets that are never in argv.

Lifted out of cli.py, which had grown to 1,949 lines and four
responsibilities. Nothing here changed on the way across.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .. import config, paths
from ..term import BOLD, DIM, GREEN, RED, RESET


def _why_not_that_hotkey(key: str, value: str) -> str:
    """Why this key name will not work, or "" if it will.

    validate() warned about it on the next load, by which time the daemon had
    already started with no hotkey at all — which from the outside looks
    exactly like a broken microphone.
    """
    if str(key) != "hotkey.key" or not str(value).strip():
        return ""
    from ..hotkey import HotkeyUnavailable, key_code

    try:
        key_code(value)
    except HotkeyUnavailable as exc:
        return str(exc)
    except ModuleNotFoundError:
        return ""
    return ""


def _hotkey_report() -> dict[str, Any]:
    """Everything that can be wrong with the hotkey, answered at once.

    The key not working has had four different causes in this program and one
    message for all of them. Each line below is a separate question, so the
    answer says which one it is instead of naming the likeliest.
    """
    import grp
    import os

    from .. import ipc
    from ..hotkey import HotkeyUnavailable, explain_missing, key_code

    cfg = config.load()
    want = str(cfg["hotkey"].get("key", ""))
    out: dict[str, Any] = {"configured": want,
                           "mode": str(cfg["hotkey"].get("mode", "")),
                           "enabled": bool(cfg["hotkey"].get("enabled", True))}

    try:
        code = key_code(want)
        out["code"] = code
        out["name_ok"] = True
    except HotkeyUnavailable as exc:
        out["name_ok"] = False
        out["problem"] = str(exc)
        return out
    except ModuleNotFoundError:
        out["problem"] = "evdev is not installed; reinstall omavoi"
        return out

    try:
        entry = grp.getgrnam("input")
        user = os.environ.get("USER") or ""
        out["group_listed"] = user in entry.gr_mem
        out["group_held"] = entry.gr_gid in os.getgroups()
    except KeyError:
        out["group_listed"] = out["group_held"] = False

    out["devices_problem"] = explain_missing(code, want)
    # Which layout is loaded, so the caller can judge the one thing this
    # program cannot: whether Right Alt is AltGr here. It is on German,
    # French, Spanish, Polish and Nordic layouts among others, where it
    # types characters — @ is AltGr+Q on a German keyboard — and holding it
    # to dictate means those characters cannot be typed without starting a
    # take. Reported rather than decided: the person typing on it knows, and
    # a list of layouts here would be a guess that goes stale.
    if want.upper() in ("RIGHTALT", "KEY_RIGHTALT"):
        out["layout"] = _keyboard_layout()

    info = ipc.ping()
    if info is None:
        out["daemon"] = "not running"
    else:
        hk = info.get("hotkey") or {}
        out["daemon"] = "running"
        # Live only when there is a listener: status() falls back to the
        # configured key for display, and reading that as a binding is how
        # a dead hotkey looked bound.
        out["bound"] = str(hk.get("key", "")) if hk.get("enabled") else ""
        out["bound_devices"] = hk.get("devices") or []
        # A listener with no devices is not listening. `enabled` only says an
        # object exists, and for one startup it said so about a listener that
        # had opened nothing at all.
        out["listener"] = bool(hk.get("enabled")) and bool(out["bound_devices"])
        # The one that hid a bug for a whole session: the file said one thing
        # and the listener was on another.
        out["matches"] = (bool(hk.get("enabled"))
                          and str(hk.get("key", "")) == want)
    return out


def _keyboard_layout() -> str:
    """The active xkb layout, or "" when nothing will say."""
    import os
    import subprocess

    explicit = os.environ.get("XKB_DEFAULT_LAYOUT")
    if explicit:
        return explicit
    try:
        out = subprocess.run(["hyprctl", "getoption", "input:kb_layout"],
                             capture_output=True, timeout=2, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in out.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("str:"):
            return line.split(":", 1)[1].strip()
    return ""


def cmd_hotkey(args: argparse.Namespace) -> int:
    """Read one key press, or say why the key is not working."""
    from ..hotkey import HotkeyUnavailable, capture

    if args.action == "check":
        r = _hotkey_report()
        if args.json:
            print(json.dumps(r, ensure_ascii=False, indent=2))
            # The same question the text output answers: is the daemon reading
            # the configured key. `devices_problem` describes *this* process's
            # access to /dev/input, and keying the exit status off it failed on
            # every working machine where the daemon holds the input group and
            # the shell does not — which is every machine where the group was
            # granted after login, including the one this was written on. The
            # text path demoted that to a parenthetical; the exit code was the
            # sixth place in this program to report the checking process's
            # state instead of the daemon's. The field stays in the payload;
            # it is real, it just is not the answer.
            return 0 if r.get("listener") and r.get("matches") else 1

        def line(ok: bool, text: str) -> None:
            mark = f"{GREEN}ok{RESET}  " if ok else f"{RED}no{RESET}  "
            print(f"  {mark}{text}")

        print(f"{BOLD}hotkey{RESET}    {r['configured']} ({r['mode']})")
        if not r.get("enabled"):
            line(False, "hotkey.enabled is false; nothing is listening by design")
            return 1
        if not r.get("name_ok"):
            line(False, r.get("problem", "the key name is not an evdev key"))
            print(f"{DIM}  omavoi hotkey capture   — press the key you want{RESET}")
            return 1
        line(True, f"{r['configured']} is a real evdev key (code {r['code']})")
        if r.get("layout"):
            print(f"{DIM}  Right Alt is AltGr on many non-US layouts, where it "
                  f"types characters and cannot be held without that. "
                  f"Yours is {r['layout']!r}.{RESET}")
        # The daemon is asked first, because the daemon is the thing that has
        # to work. Everything below this is about *this* process's access to
        # the devices, which is only ever evidence about the daemon's — and it
        # can be wrong in both directions. It was wrong here: the daemon was
        # started through `newgrp` and had the group, this shell did not, and
        # the check announced a broken hotkey over a working one.
        if r.get("listener"):
            line(True, "the daemon is reading it on: "
                       + ", ".join(r.get("bound_devices") or []))
            if not r.get("matches"):
                line(False, f"but on {r.get('bound')!r}, not {r['configured']!r}"
                            " — it has not picked up the change")
                print(f"{DIM}  systemctl --user restart omavoid{RESET}")
                return 1
            if not r.get("group_held"):
                # Worth saying, because it will surprise anyone who reads
                # /dev/input by hand — but no longer worth a warning: capture
                # goes through the daemon and doctor asks it too, so nothing
                # the program does depends on this shell's own access.
                print(f"{DIM}  (this login has no device access of its own — "
                      f"the daemon was started with the group and this session "
                      f"was not. Nothing here depends on it.){RESET}")
            return 0

        if r.get("daemon") != "running":
            line(False, "the daemon is not running")
            print(f"{DIM}  systemctl --user start omavoid{RESET}")
            return 1
        line(False, "the daemon is running and reading no device")

        # Why not. This process's own access is the best evidence available,
        # and the caveat above is why it is phrased as evidence.
        if not r.get("group_listed"):
            line(False, "you are not in the `input` group, so not one keyboard "
                        "can be opened")
            print(f"{DIM}  sudo usermod -aG input $USER   then log out and back in{RESET}")
            return 1
        if not r.get("group_held"):
            line(False, "you are in the `input` group, but this login started "
                        "before that")
            print(f"{DIM}  log out and back in — a group is granted at login "
                  f"and cannot be added to a session already running.{RESET}")
            print(f"{DIM}  To avoid that: start the daemon through `newgrp "
                  f"input`, which is setuid root and re-reads /etc/group, so it "
                  f"gets the group without a new login. README: \"the input "
                  f"group\".{RESET}")
            return 1

        dp = r.get("devices_problem") or ""
        if dp:
            line(False, dp)
            print(f"{DIM}  omavoi hotkey capture   — press a key that exists here{RESET}")
            return 1
        print(f"{DIM}  systemctl --user restart omavoid{RESET}")
        return 1

    if args.action != "capture":
        return 1

    # The daemon first, because it is the process that can read a key. This
    # one often cannot: a session that joined the `input` group after logging
    # in has no device access, and the console's "press a key" button is a
    # child of that session — so the button failed on exactly the machines
    # where rebinding was the thing you were trying to do.
    from .. import ipc

    # request() raises rather than returning None when there is no socket, and
    # a capture needs longer than its default read window allows for.
    reply: dict[str, Any] | None
    try:
        reply = ipc.request({"cmd": "capture", "timeout": args.timeout},
                                   timeout=args.timeout + 10.0)
    except (ConnectionError, OSError, ValueError):
        reply = None
    if reply is not None:
        if reply.get("ok") and reply.get("key"):
            if args.json:
                print(json.dumps({"ok": True, "key": str(reply["key"])}))
            else:
                print(str(reply["key"]))
            return 0
        # A running daemon that cannot read a key is the answer, not a reason
        # to try again here with strictly less access.
        why = str(reply.get("error") or "the daemon could not read a key")
        if args.json:
            print(json.dumps({"ok": False, "error": why}))
        else:
            print(f"{RED}{why}{RESET}", file=sys.stderr)
        return 1

    try:
        name = capture(timeout=args.timeout)
    except HotkeyUnavailable as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1
    except ModuleNotFoundError:
        msg = "evdev is not installed"
        print(json.dumps({"ok": False, "error": msg}) if args.json
              else f"{RED}{msg}{RESET}", file=sys.stderr if not args.json else sys.stdout)
        return 1
    if not name:
        if args.json:
            print(json.dumps({"ok": False, "error": "nothing was pressed"}))
        else:
            print(f"{DIM}nothing was pressed{RESET}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"ok": True, "key": name}))
    else:
        print(name)
    return 0


def cmd_secrets(args: argparse.Namespace) -> int:
    """Keys, read from stdin and never from a command line."""
    from .. import secrets

    if args.action == "list":
        stored = secrets.load_file()
        if not stored:
            print(f"{DIM}nothing stored in {paths.secrets_file()}{RESET}")
            return 0
        for name in sorted(stored):
            print(f"  {name:20s} {secrets.redact(stored[name])}")
        return 0

    if args.action == "set":
        if not args.name:
            print(f"{RED}usage: omavoi secrets set <name>   "
                  f"(the value comes from stdin){RESET}", file=sys.stderr)
            return 1
        if sys.stdin.isatty():
            print(f"{DIM}paste the key, then press ctrl-D{RESET}", file=sys.stderr)
        value = sys.stdin.read().strip()
        if not value:
            print(f"{RED}nothing on stdin{RESET}", file=sys.stderr)
            return 1
        secrets.store(args.name, value)
        print(f"{GREEN}ok{RESET} {args.name} = {secrets.redact(value)} "
              f"{DIM}in {paths.secrets_file()}{RESET}")
        return 0

    if args.action == "rm":
        if not args.name:
            print(f"{RED}usage: omavoi secrets rm <name>{RESET}", file=sys.stderr)
            return 1
        secrets.store(args.name, "")
        print(f"{GREEN}ok{RESET} {args.name} removed")
        return 0
    return 1
