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

    from .. import daemon as daemon_mod
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

    info = daemon_mod.ping()
    if info is None:
        out["daemon"] = "not running"
    else:
        hk = info.get("hotkey") or {}
        out["daemon"] = "running"
        out["bound"] = str(hk.get("key", ""))
        out["bound_devices"] = hk.get("devices") or []
        out["listener"] = bool(hk.get("enabled"))
        # The one that hid a bug for a whole session: the file said one thing
        # and the listener was on another.
        out["matches"] = str(hk.get("key", "")) == want
    return out


def cmd_hotkey(args: argparse.Namespace) -> int:
    """Read one key press, or say why the key is not working."""
    from ..hotkey import HotkeyUnavailable, capture

    if args.action == "check":
        r = _hotkey_report()
        if args.json:
            print(json.dumps(r, ensure_ascii=False, indent=2))
            return 0 if r.get("matches") and not r.get("devices_problem") else 1

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

        if r.get("group_listed") and not r.get("group_held"):
            line(False, "you are in the `input` group but this shell predates it "
                        "— log out and back in")
        elif not r.get("group_listed"):
            line(False, "you are not in the `input` group")
            print(f"{DIM}  sudo usermod -aG input $USER   then log out and back in{RESET}")
        else:
            line(True, "in the `input` group")

        dp = r.get("devices_problem") or ""
        line(not dp, dp or f"a device can emit {r['configured']}")

        if r.get("daemon") != "running":
            line(False, "the daemon is not running")
            print(f"{DIM}  systemctl --user start omavoid{RESET}")
            return 1
        if not r.get("listener"):
            line(False, "the daemon is running but has no key listener")
            print(f"{DIM}  systemctl --user restart omavoid{RESET}")
            return 1
        if not r.get("matches"):
            line(False, f"the daemon is listening on {r.get('bound')!r}, not "
                        f"{r['configured']!r} — it did not pick up the change")
            print(f"{DIM}  systemctl --user restart omavoid{RESET}")
            return 1
        line(True, "the daemon is listening on it: "
                   + ", ".join(r.get("bound_devices") or []))
        return 0 if not dp else 1

    if args.action != "capture":
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
