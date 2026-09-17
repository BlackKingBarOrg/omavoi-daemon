"""What is running, what is missing, and what happened.

Lifted out of cli.py, which had grown to 1,949 lines and four
responsibilities. Nothing here changed on the way across.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from typing import Any

from .. import __version__, config, ipc, models, paths
from ..term import BOLD, DIM, GREEN, RED, RESET, YELLOW
from .catalogue import _print_engines


def _print_entry(entry: dict[str, Any], verbose: bool) -> None:
    import datetime

    ts = datetime.datetime.fromtimestamp(entry.get("ts", 0)).strftime("%m-%d %H:%M:%S")
    text = entry.get("text") or ""
    rejected = entry.get("rejected") or ""
    head = text if text else f"{DIM}(dropped: {rejected}){RESET}"
    print(f"{DIM}{ts}{RESET}  {head}")
    if not verbose:
        return

    audio = entry.get("audio", {})
    asr_info = entry.get("asr", {})
    print(f"          {DIM}audio{RESET}   {audio.get('seconds', 0):.2f}s  "
          f"rms {audio.get('rms_dbfs', 0):.1f} dBFS  peak {audio.get('peak_dbfs', 0):.1f} dBFS")
    if asr_info:
        print(f"          {DIM}model{RESET}   {asr_info.get('model', '?')} "
              f"[{asr_info.get('device', '?')}]  "
              f"decode {asr_info.get('decode_seconds', 0):.2f}s  RTF {asr_info.get('rtf', 0):.3f}  "
              f"lang {asr_info.get('language', '?')}")
        for seg in asr_info.get("segments", []):
            flag = RED if seg.get("avg_logprob", 0) < -1.0 else DIM
            print(f"            {flag}[{seg.get('start', 0):5.2f}-{seg.get('end', 0):5.2f}] "
                  f"logprob={seg.get('avg_logprob', 0):6.2f} "
                  f"no_speech={seg.get('no_speech_prob', 0):.3f}{RESET} {seg.get('text', '')}")
    raw = entry.get("raw_text") or ""
    if raw and raw != text:
        print(f"          {DIM}raw{RESET}     {raw}")
    changes = entry.get("post", {}).get("changes") or []
    if changes:
        print(f"          {DIM}post{RESET}    {' | '.join(changes)}")
    win = entry.get("window", {})
    if win:
        print(f"          {DIM}window{RESET}  {win.get('class', '?')} "
              f"profile={win.get('profile') or '-'}")
    inj = entry.get("inject", {})
    if inj:
        print(f"          {DIM}inject{RESET}  {inj.get('method')} "
              f"{'ok' if inj.get('ok') else 'FAILED ' + inj.get('error', '')}")
    for warning in entry.get("warnings", []):
        print(f"          {YELLOW}! {warning}{RESET}")


def cmd_status(args: argparse.Namespace) -> int:
    info = ipc.ping()
    if info is None:
        if args.json:
            print(json.dumps({"state": "stopped", "text": "", "class": "stopped"}))
        else:
            print(f"{DIM}daemon not running{RESET}")
        return 1

    if args.json:
        # Shaped for a bar module: text / tooltip / class.
        state = info["state"]
        label = {"idle": "", "recording": "recording", "transcribing": "transcribing"}
        print(json.dumps({
            "text": label.get(state, state),
            "tooltip": info["backend"],
            "class": state,
            "state": state,
            # The bar only needs the first four; anything scripting against
            # this wants to know what is actually loaded.
            "engines": info.get("engines", {}),
        }, ensure_ascii=False))
        return 0

    print(f"{BOLD}state{RESET}     {info['state']}")
    print(f"{BOLD}pid{RESET}       {info['pid']}  up {info['uptime']:.0f}s  {info['takes']} takes")
    _print_engines(info)
    hk = info["hotkey"]
    print(f"{BOLD}hotkey{RESET}    {hk['key']} ({hk['mode']})")
    for dev in hk["devices"]:
        print(f"          {DIM}{dev}{RESET}")
    au = info["audio"]
    mark = f"{GREEN}ok{RESET}" if au["healthy"] else f"{RED}unhealthy{RESET}"
    print(f"{BOLD}audio{RESET}     {mark}  level {au['level']:.3f}  "
          f"pre-roll {au['preroll']}s / tail {au['tail']}s")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    from ..history import History

    entries = History(config.load()).entries(args.number)
    if args.json:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
        return 0
    if not entries:
        print(f"{DIM}no takes recorded yet{RESET}")
        return 0
    for entry in entries:
        _print_entry(entry, args.verbose)
    return 0


def cmd_last(args: argparse.Namespace) -> int:
    from ..history import History

    entry = History(config.load()).last()
    if entry is None:
        print(f"{DIM}no takes recorded yet{RESET}")
        return 1
    if args.json:
        print(json.dumps(entry, ensure_ascii=False, indent=2))
    elif args.raw:
        print(entry.get("raw_text", ""))
    else:
        _print_entry(entry, verbose=True)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from ..history import History

    stats = History(config.load()).stats()
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    if not stats.get("count"):
        print(f"{DIM}no takes recorded yet{RESET}")
        return 0
    print(f"{BOLD}takes{RESET}         {stats['count']}")
    print(f"{BOLD}empty{RESET}         {stats['empty']} ({stats['empty_rate']:.1%})")
    if stats.get("median_rtf") is not None:
        print(f"{BOLD}median RTF{RESET}    {stats['median_rtf']:.4f}")
    if stats.get("median_rms_dbfs") is not None:
        print(f"{BOLD}median level{RESET}  {stats['median_rms_dbfs']:.1f} dBFS")
    print(f"{BOLD}injected via{RESET}  {stats['inject_methods']}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    from .. import setup as setup_mod

    cfg = config.load()
    report = setup_mod.check(cfg)

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0 if report.ready else 1

    print(f"{BOLD}Omavoi setup{RESET}  {report.done}/{len(report.steps)} done\n")
    for step in report.steps:
        if step.done:
            mark, colour = f"{GREEN}ok  {RESET}", ""
        elif step.optional:
            mark, colour = f"{YELLOW}--  {RESET}", YELLOW
        else:
            mark, colour = f"{RED}todo{RESET}", RED
        tail = f" {DIM}(optional){RESET}" if step.optional and not step.done else ""
        print(f"  {mark} {step.title}{tail}")
        print(f"       {DIM}{step.detail}{RESET}")
        if not step.done and step.command:
            root = f" {YELLOW}[needs root]{RESET}" if step.needs_root else ""
            print(f"       {colour}$ {step.command}{RESET}{root}")
        if not step.done and step.note:
            print(f"       {DIM}{step.note}{RESET}")
        print()

    if report.ready:
        print(f"{GREEN}Ready. Hold {cfg['hotkey']['key']} and talk.{RESET}")
        return 0

    nxt = report.blocking[0]
    if args.run:
        if nxt.needs_root:
            print(f"{YELLOW}This step needs root; run it yourself:{RESET}\n  {nxt.command}")
            return 1
        print(f"{BOLD}Running:{RESET} {nxt.command}")
        return subprocess.call(nxt.command, shell=True)

    print(f"Next: {BOLD}{nxt.title}{RESET}")
    print(f"{DIM}Run `omavoi setup --run` to do the next step that does not need root.{RESET}")
    return 1


def cmd_doctor(args: argparse.Namespace) -> int:
    from ..hotkey import find_devices, key_code

    cfg = config.load()
    ok = True

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        mark = f"{GREEN}ok  {RESET}" if good else f"{RED}FAIL{RESET}"
        # A padded field and nothing else runs the two together the moment a
        # label is longer than the column — "read by the daemonon 2 devices".
        # One space always, and the column is one narrower so short rows land
        # where they always did.
        print(f"  {mark} {label:<21} {detail}")

    print(f"{BOLD}Omavoi {__version__}{RESET}\n")

    print(f"{BOLD}config{RESET}")
    problems = config.validate(cfg)
    check("config file", not problems, str(paths.config_file()) +
          ("" if paths.config_file().exists() else f" {DIM}(absent, using defaults){RESET}"))
    for problem in problems:
        print(f"    {YELLOW}! {problem}{RESET}")

    print(f"\n{BOLD}external commands{RESET}")
    for tool, needed in [("pw-record", True), ("wtype", True), ("wl-copy", True),
                         ("wl-paste", False), ("hyprctl", True), ("xdotool", True),
                         ("notify-send", False), ("ffmpeg", False)]:
        found = shutil.which(tool)
        check(tool, bool(found) or not needed,
              found or (f"{RED}missing{RESET}" if needed else f"{DIM}optional, absent{RESET}"))

    print(f"\n{BOLD}hotkey{RESET}")
    try:
        from ..hotkey import explain_missing

        code = key_code(cfg["hotkey"]["key"])
        key = str(cfg["hotkey"]["key"])

        # The daemon first. This process opening a device says nothing about
        # whether the key works — the daemon is what reads it, and the two can
        # differ: a session that joined the `input` group after logging in
        # cannot open a keyboard while a daemon started later reads the same
        # ones. doctor said FAIL and exited 1 over a hotkey that was working,
        # which is the third place in this program to have its own copy of
        # that wrong answer.
        # Asked here rather than reusing the daemon block further down: that
        # one runs after this section, and one extra socket round trip is
        # cheaper than reordering a function that prints in a fixed order.
        live = (ipc.ping() or {}).get("hotkey") or {}
        bound = live.get("devices") or []
        if bound and live.get("enabled"):
            check(f"{key} read by the daemon", True, f"{len(bound)} device(s)")
            for name in bound:
                print(f"    {DIM}{name}{RESET}")
        else:
            devices = find_devices(code)
            # Not a guess. `id -nG` was the worst possible thing to point at
            # here: it reports the groups this session inherited, so someone
            # who ran usermod and did not log out sees `input` absent,
            # concludes they are not in the group, runs usermod again, and is
            # told the same thing forever. explain_missing asks separately.
            check(f"{key} readable on", bool(devices),
                  f"{len(devices)} device(s), but the daemon is reading none"
                  if devices
                  else f"{RED}nothing{RESET} — "
                       + (explain_missing(code, key)
                          or "no reason could be determined"))
            for dev in devices:
                print(f"    {DIM}{dev.path}  {dev.name}{RESET}")
                dev.close()
    except Exception as exc:
        check("hotkey", False, str(exc))

    print(f"\n{BOLD}asr{RESET}")
    from .. import asr as asr_mod

    backend_name = asr_mod.canonical(cfg["speech"]["backend"])
    check("backend", backend_name is not None, str(backend_name or cfg["speech"]["backend"]))

    if backend_name == "api":
        from .. import secrets
        from ..asr.api_whisper import PROVIDERS

        api = cfg["speech"]["api"]
        provider = api.get("provider", "openai")
        check("provider", provider in PROVIDERS or bool(api.get("base_url")), provider)
        key = secrets.resolve(
            api.get("key_env", "") or PROVIDERS.get(provider, {}).get("key_env", ""),
            api.get("key_name", "") or provider,
        )
        check("api key", bool(key), secrets.redact(key))
    elif backend_name == "local-whispercpp":
        from ..asr.local_whispercpp import find_server

        server = find_server()
        check("whisper.cpp server", bool(server),
              server or f"{RED}not found — pacman -S whisper-cpp ggml-vulkan{RESET}")
        key = cfg["speech"]["model"]
        if not key.startswith("ggml:"):
            key = f"ggml:{key}"
        check(f"model {key}", models.is_downloaded(key),
              str(models.local_path(key) or f"{YELLOW}omavoi model pull {key}{RESET}"))
    elif backend_name == "api":
        # The endpoint answers or it does not, and `omavoi speech check`
        # is the command that asks. Nothing local to inspect.
        api = cfg["speech"].get("api") or {}
        check("endpoint", bool(api.get("base_url") or api.get("provider")),
              str(api.get("base_url") or api.get("provider") or "not configured"))

    print(f"\n{BOLD}audio{RESET}")
    try:
        out = subprocess.run(["pactl", "get-default-source"], capture_output=True, timeout=2, check=False)
        source = out.stdout.decode().strip()
        check("default source", bool(source), source or f"{RED}none{RESET}")
    except (subprocess.SubprocessError, OSError) as exc:
        check("default source", False, str(exc))

    if args.mic:
        import time as _time

        from ..audio import RingCapture

        print(f"  {DIM}measuring 2s of input, say something...{RESET}")
        ring = RingCapture(cfg)
        try:
            ring.start()
            mark = ring.mark()
            _time.sleep(2.0)
            cap = ring.take(mark)
            good = cap.rms_dbfs > cfg["audio"]["warn_rms_dbfs"]
            check("input level", good,
                  f"rms {cap.rms_dbfs:.1f} dBFS  peak {cap.peak_dbfs:.1f} dBFS"
                  + ("" if good else f"  {YELLOW}too quiet, expect dropped words{RESET}"))
        finally:
            ring.stop()

    print(f"\n{BOLD}daemon{RESET}")
    info = ipc.ping()
    check("running", info is not None,
          f"pid {info['pid']}, {info['state']}" if info else f"{DIM}not running{RESET}")

    print()
    print(f"{GREEN}all good{RESET}" if ok else f"{YELLOW}problems above{RESET}")
    return 0 if ok else 1
