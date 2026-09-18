"""Command line: `omavoi <command>`."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__, term
from .commands._support import setup_logging
from .commands.catalogue import cmd_llm, cmd_model, cmd_speech
from .commands.health import (
    cmd_doctor,
    cmd_history,
    cmd_last,
    cmd_setup,
    cmd_stats,
    cmd_status,
)
from .commands.keys import cmd_hotkey, cmd_secrets
from .commands.modes import cmd_mode
from .commands.run import (
    cmd_daemon,
    cmd_inject,
    cmd_record,
    cmd_reload,
    cmd_transcribe,
)
from .commands.settings import cmd_config
from .commands.words import cmd_dict, cmd_names


def _color(enabled: bool) -> None:
    if not enabled:
        term.disable()


# -- commands ----------------------------------------------------------------

# -- parser ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omavoi",
        description="Voice dictation for Omarchy: hold a key, talk, the text lands in the focused window.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"omavoi {__version__}")
    parser.add_argument("--log-level", default="", help="DEBUG, INFO, WARNING or ERROR")
    parser.add_argument("--no-color", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("daemon", help="run the daemon (keeps the model resident)")
    p.set_defaults(func=cmd_daemon)

    p = sub.add_parser("record", help="drive recording from a keybinding or script")
    p.add_argument("action", choices=["start", "stop", "toggle", "cancel"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("status", help="daemon status (--json for a bar module)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("history", help="recent takes, and deleting them")
    # Optional, so `omavoi history -n 40 --json` -- what the console runs on
    # every refresh -- still means the listing it has always meant.
    p.add_argument("action", nargs="?", default="list", choices=["list", "rm", "clear"])
    p.add_argument("ids", nargs="*", help="for rm: the id each take is listed with")
    p.add_argument("-n", "--number", type=int, default=10)
    p.add_argument("-v", "--verbose", action="store_true",
                   help="include per-segment confidences and post-processing")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("last", help="full diagnostics for the most recent take")
    p.add_argument("--raw", action="store_true", help="print the model's raw text only")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_last)

    p = sub.add_parser("stats", help="aggregates: empty rate, RTF, input level")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("model", help="download and switch local models")
    p.add_argument("action", choices=["list", "pull", "rm", "use"])
    p.add_argument("models", nargs="*", help="e.g. large-v3 or ggml:large-v3")
    p.add_argument("--json", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="switch even to a backend that is not installed here")
    p.set_defaults(func=cmd_model)

    p = sub.add_parser(
        "hotkey",
        help="choose the push-to-talk key, or combination, by pressing it")
    p.add_argument("action", choices=["capture", "check"])
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_hotkey)

    p = sub.add_parser("secrets", help="API keys, read from stdin")
    p.add_argument("action", choices=["list", "set", "rm"])
    p.add_argument("name", nargs="?", help="which key")
    p.set_defaults(func=cmd_secrets)

    p = sub.add_parser("speech", help="the remote speech endpoint")
    p.add_argument("action", choices=["show", "check"], nargs="?", default="show")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_speech)

    p = sub.add_parser("llm", help="the [llm.<name>] entries a mode's steps name")
    # No add or rm: there are three configurations and migrate() folds anything
    # else away on the next load, so a command that made a fourth reported a
    # success that did not survive it.
    p.add_argument("action", choices=["list", "check"])
    p.add_argument("rest", nargs="*", help="check [name]")
    p.add_argument("--json", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_llm)

    p = sub.add_parser("config", help="inspect and change settings")
    p.add_argument("action", choices=["init", "show", "get", "set", "edit", "path"])
    p.add_argument("key", nargs="?", default="")
    p.add_argument("value", nargs="?", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("dict", help="term dictionary: pin words the model keeps mishearing")
    p.add_argument("action", choices=["list", "add", "rm"])
    p.add_argument("heard", nargs="?", default="", help="what the model produced")
    p.add_argument("meant", nargs="?", default="", help="what you actually said")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_dict)

    p = sub.add_parser("reload", help="make the daemon re-read the config")
    p.set_defaults(func=cmd_reload)

    p = sub.add_parser("transcribe", help="transcribe a file through the same pipeline")
    p.add_argument("file")
    p.add_argument("--inject", action="store_true", help="also type the result")
    p.add_argument("--mode", default="", help="force a mode instead of matching the window")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("names", help="proper nouns, written only in their correct form")
    p.add_argument("action", choices=["list", "add", "rm", "dryrun", "enable"])
    p.add_argument("names", nargs="*")
    p.add_argument("--group", default="", help="People, Places, Terms, ...")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_names)

    p = sub.add_parser("setup", help="what is still missing, and the command for each")
    p.add_argument("--run", action="store_true",
                   help="run the next step that does not need root")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("mode", help="inspect and edit the modes")
    p.add_argument("action", choices=["list", "show", "which", "use", "auto", "new",
                                      "rm", "set", "match", "unmatch", "step"])
    p.add_argument("rest", nargs="*", help="arguments for the action")
    p.add_argument("--json", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="switch even if the mode's models will not fit in VRAM")
    p.set_defaults(func=cmd_mode)

    p = sub.add_parser("inject", help="type test text into the focused window")
    p.add_argument("text", nargs="?", default="", help="what to inject")
    p.add_argument("--delay", type=float, default=3.0,
                   help="seconds to focus the target first (default 3)")
    p.add_argument("--lines", type=int, default=1, help="inject this many lines")
    p.add_argument("--method", default="", choices=["", "wtype", "clipboard"],
                   help="force a route instead of auto")
    p.add_argument("--paste-via", default="",
                   choices=["", "shortcut", "wtype", "xdotool"],
                   dest="paste_via", help="how the paste keystroke is sent")
    p.add_argument("--via-daemon", action="store_true", dest="via_daemon",
                   help="have the daemon inject instead of this process")
    p.set_defaults(func=cmd_inject)

    p = sub.add_parser("doctor", help="check the whole setup")
    p.add_argument("--mic", action="store_true", help="also measure 2s of microphone input")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _color(sys.stdout.isatty() and not args.no_color and os.environ.get("NO_COLOR") is None)
    if args.command != "daemon":
        setup_logging(args.log_level or "WARNING")
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
