"""Command line: `omavoi <command>`."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, asr, config, daemon, models, paths

log = logging.getLogger("omavoi")

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def _color(enabled: bool) -> None:
    if not enabled:
        globals().update(GREEN="", RED="", YELLOW="", DIM="", BOLD="", RESET="")


def setup_logging(level: str = "INFO", to_file: bool = False) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if to_file:
        paths.state_dir().mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(paths.log_file(), encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def load_wav(path: Path, want_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Read an audio file to float32 mono at `want_rate`, via ffmpeg if needed."""
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == want_rate:
                raw = wav.readframes(wav.getnframes())
                return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, want_rate
    except (wave.Error, OSError):
        pass

    if shutil.which("ffmpeg") is None:
        raise SystemExit(f"{path} is not 16 kHz mono WAV and ffmpeg is not installed")
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(want_rate), "-"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0, want_rate


# -- commands ----------------------------------------------------------------

def cmd_daemon(args: argparse.Namespace) -> int:
    cfg = config.load()
    setup_logging(args.log_level or cfg["ui"].get("log_level", "INFO"), to_file=True)
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
        reply = daemon.request({"cmd": args.action})
    except (ConnectionError, OSError) as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1
    if not reply.get("ok"):
        print(f"{RED}{reply.get('error', reply)}{RESET}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(reply, ensure_ascii=False))
    return 0


def _print_engines(info: dict[str, Any]) -> None:
    """The two model families and whether each is actually up.

    Split from the old single `backend` line because a mode is a chain: the
    speech model being resident says nothing about whether the LLM step it
    hands off to has ever started.
    """
    engines = info.get("engines") or {}
    speech = engines.get("speech") or {}
    if not speech:
        # An older daemon still running after an upgrade.
        print(f"{BOLD}backend{RESET}   {info['backend']}")
        return

    where = f"  {DIM}{speech['url']}{RESET}" if speech.get("url") else ""
    live = f"{GREEN}running{RESET}" if speech.get("live") else f"{YELLOW}not loaded{RESET}"
    print(f"{BOLD}speech{RESET}    {speech['engine']} {speech['model']} "
          f"[{speech['device']}]  {live}{where}")

    llms = engines.get("llm") or []
    if not llms:
        return
    width = max(len(entry["name"]) for entry in llms)
    for i, llm in enumerate(llms):
        head = f"{BOLD}llm{RESET}       " if i == 0 else "          "
        if llm.get("problem"):
            note = f"{RED}{llm['problem']}{RESET}"
        elif llm.get("live"):
            note = f"{GREEN}running{RESET}" if not llm.get("remote") else f"{GREEN}ready{RESET}"
        else:
            # A local server starts on the first take that names it, so cold
            # is the resting state, not a fault.
            note = f"{DIM}cold{RESET}"
        where = f"  {DIM}{llm['url']}{RESET}" if llm.get("url") and not llm.get("remote") else ""
        print(f"{head}{llm['name']:<{width}}  {llm['engine']} {llm['model']}  {note}{where}")


def cmd_status(args: argparse.Namespace) -> int:
    info = daemon.ping()
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


def cmd_history(args: argparse.Namespace) -> int:
    from .history import History

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
    from .history import History

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
    from .history import History

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


def _is_remote(backend: str, base_url: str) -> bool:
    """Whether using this model sends text off the machine."""
    from .llm import ON_MACHINE

    backend = backend.strip().lower()
    if backend == "anthropic":
        return True
    if backend in ON_MACHINE and not base_url:
        # llama-local owns its own server on loopback and never writes a URL.
        return False
    if not base_url:
        # An OpenAI-compatible backend with no base_url means api.openai.com.
        return True
    host = base_url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    return host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def _vram_split(engines: dict[str, Any]) -> dict[str, Any]:
    """VRAM, with the part our own two model families hold broken out.

    One bar for "13.4 of 16.3 GB used" does not say whether the speech model
    or the LLM is the thing filling it, which is the only actionable question
    when a chain stops fitting.
    """
    from . import gpu

    info = gpu.vram()
    if not info:
        return info
    speech = engines.get("speech") or {}
    speech_pid = int(speech.get("pid", 0) or 0) if speech.get("live") else 0
    llms = [e for e in (engines.get("llm") or []) if e.get("live") and e.get("pid")]
    llm_pids = {int(e["pid"]) for e in llms}

    labels = {
        "speech": (f"{speech.get('engine', '')} {speech.get('model', '')}".strip()
                   if speech_pid else ""),
        "llm": ", ".join(f"{e['name']} {e['model']}" for e in llms),
        "other": "",
    }
    out = dict(info)
    out["segments"] = [
        seg | {"label": labels.get(seg["kind"], "")}
        for seg in gpu.segments(int(info.get("used_mb", 0)), speech_pid, llm_pids)
    ]
    return out


def _llm_live(engines: dict[str, Any], name: str) -> dict[str, Any]:
    """The daemon's view of one LLM, merged into its config row.

    Named `live_*` so it cannot be confused with the config fields beside it —
    `remote` is what the file says, `live_running` is what is happening.
    """
    for entry in engines.get("llm") or []:
        if entry.get("name") == name:
            return {
                "live_running": bool(entry.get("live")),
                "live_url": str(entry.get("url", "")),
                "live_problem": str(entry.get("problem", "")),
                "live_engine": str(entry.get("engine", "")),
            }
    return {"live_running": False, "live_url": "", "live_problem": "", "live_engine": ""}


def cmd_model(args: argparse.Namespace) -> int:
    if args.action == "list":
        cfg = config.load()
        active = cfg["speech"]["model"]
        if args.json:
            from . import gpu, i18n

            # The console renders these notes verbatim, so they are translated
            # here rather than there: the text lives beside the catalogue.
            lang = i18n.ui_lang(cfg)

            # The config says which model was asked for; only the daemon knows
            # which one is loaded, and the two differ after every edit until a
            # restart. Absent daemon leaves `engines` null rather than implying
            # nothing is running.
            live = daemon.ping()
            engines = (live or {}).get("engines") or {}
            speech_now = engines.get("speech") or {}
            running_speech = str(speech_now.get("model", "")) if speech_now.get("live") else ""
            running_llm = {
                str(e.get("model", "")) for e in (engines.get("llm") or []) if e.get("live")
            }

            def _is_running(spec: models.ModelSpec) -> bool:
                if spec.kind == models.LLM:
                    return spec.key in running_llm
                if not running_speech:
                    return False
                return (spec.key == running_speech
                        or (spec.fmt == models.CT2 and spec.id == running_speech))

            # Which weights an LLM configuration points at, so an llm row can
            # answer "is this the one in use?" the same way a speech row does.
            # It could not before: `active` compared every entry against the
            # speech model, so no llm entry was ever active and the UI grew a
            # second, different way of asking.
            llm_chosen = {str(e.get("model", ""))
                          for e in cfg.get("llm", {}).values()
                          if str(e.get("model", ""))}

            rows = []
            for spec in models.CATALOG:
                # Every model costs VRAM, so every model can fail to fit.
                # Asking only about LLMs meant a speech model that will not
                # load reported fits:true — and the speech table had no
                # warning to show because the data was never there.
                fit = gpu.fits(spec.size_mb)
                rows.append({
                    "fits": fit.get("fits", True),
                    "needed_mb": fit.get("needed_mb", 0),
                    "key": spec.key, "id": spec.id, "fmt": spec.fmt, "kind": spec.kind,
                    "backend": spec.backend, "size_mb": spec.size_mb,
                    "note": i18n.t(spec.note, lang), "tags": list(spec.tags),
                    "downloaded": models.is_downloaded(spec.key),
                    "path": str(models.local_path(spec.key) or ""),
                    "ours": models.owned_by_us(spec.key),
                    "active": (spec.key in llm_chosen
                               if spec.kind == models.LLM
                               else spec.key == active
                                    or (spec.fmt == models.CT2
                                        and spec.id == active)),
                    "running": _is_running(spec),
                })
            from . import gpu, secrets

            # Which modes name which LLM, so removing one shows what breaks.
            used_by: dict[str, list[str]] = {}
            for mode_name, mode in cfg.get("modes", {}).items():
                for step in mode.get("steps") or []:
                    used_by.setdefault(str(step.get("llm", "")), []).append(mode_name)

            llms = []
            for name, entry in sorted(cfg.get("llm", {}).items()):
                backend = str(entry.get("backend", "openai"))
                key_env = str(entry.get("key_env", ""))
                key = secrets.resolve(key_env, str(entry.get("key_name", "") or name))
                llms.append({
                    "name": name,
                    "backend": backend,
                    "model": entry.get("model", ""),
                    "base_url": entry.get("base_url", ""),
                    "remote": _is_remote(backend, str(entry.get("base_url", ""))),
                    "key_env": key_env,
                    "key_name": str(entry.get("key_name", "") or name),
                    "key": secrets.redact(key) if key_env else "",
                    "has_key": bool(key) or not key_env,
                    "used_by": sorted(used_by.get(name, [])),
                    **_llm_live(engines, name),
                })

            # The remote speech endpoint, in the same shape as an llm entry.
            # It was absent entirely, so the console could offer the engine and
            # then had nothing to configure it with — you could select "remote
            # API" for speech and there was nowhere to put the URL.
            from .asr.api_whisper import PROVIDERS

            sapi = dict(cfg["speech"].get("api") or {})
            sprov = str(sapi.get("provider", "") or "")
            preset = PROVIDERS.get(sprov, {})
            # What the backend will actually use: the explicit value, or the
            # provider's default underneath it.
            s_key_env = str(sapi.get("key_env", "") or preset.get("key_env", ""))
            # ApiWhisperBackend reads the key as `key_name or provider`, so the
            # console has to write it under exactly that. It was writing
            # "speech-api", which nothing would ever have looked up.
            s_key_name = str(sapi.get("key_name", "") or sprov or "speech-api")
            s_key = secrets.resolve(s_key_env, s_key_name)
            speech_api = {
                "provider": sprov,
                "key_name": s_key_name,
                "base_url": str(sapi.get("base_url", "") or ""),
                "model": str(sapi.get("model", "") or ""),
                "key_env": s_key_env,
                "key": secrets.redact(s_key) if s_key_env else "",
                "has_key": bool(s_key) or not s_key_env,
                # Shown greyed as the value that applies when the field is
                # left empty, so a preset is visible rather than magic.
                "default_base_url": str(preset.get("base_url", "")),
                "default_model": str(preset.get("model", "")),
                "default_key_env": str(preset.get("key_env", "")),
                "selected": cfg["speech"]["backend"] == "api",
            }

            print(json.dumps({"active": active,
                              "backend": cfg["speech"]["backend"],
                              "speech_api": speech_api,
                              "root": str(models.model_root()),
                              "models": rows,
                              "llm": llms,
                              "engines": engines or None,
                              "daemon": bool(live),
                              "vram": _vram_split(engines)}, ensure_ascii=False, indent=2))
            return 0
        print(f"{BOLD}  {'model':<24}{'size':>7}  {'state':<10}notes{RESET}")
        groups = [
            (models.CT2, models.SPEECH, "speech · local-whisper (CUDA)"),
            (models.GGML, models.SPEECH, "speech · local-whispercpp (Vulkan)"),
            ("", models.LLM, "llm · llama-local, started by the daemon"),
        ]
        for fmt, kind, engine in groups:
            entries = [s for s in models.CATALOG
                       if s.kind == kind and (not fmt or s.fmt == fmt)]
            if not entries:
                continue
            print(f"\n{DIM}{engine}{RESET}")
            for spec in entries:
                here = models.is_downloaded(spec.key)
                current = spec.key == active or (fmt == models.CT2 and spec.id == active)
                mark = f"{GREEN}*{RESET}" if current else (f"{DIM}.{RESET}" if here else " ")
                state = f"{GREEN}local{RESET}" if here else f"{DIM}remote{RESET}"
                tag = f" {YELLOW}[recommended]{RESET}" if "recommended" in spec.tags else ""
                print(f"{mark} {spec.key:<24}{spec.size_mb / 1024:>6.1f}G  {state:<18}{spec.note}{tag}")
        print(f"\n{DIM}* = in use   . = downloaded   stored in {models.model_root()}{RESET}")
        return 0

    if args.action == "pull":
        for key in args.models:
            spec = models.spec(key)
            if spec is None:
                print(f"{RED}unknown model {key}{RESET}", file=sys.stderr)
                return 1
            if models.is_downloaded(key):
                print(f"{DIM}{key} already present{RESET}")
                continue
            print(f"downloading {key} (~{spec.size_mb / 1024:.1f}G) ...")
            models.pull(key)
            print(f"{GREEN}ok{RESET} {key}")
        return 0

    if args.action == "rm":
        for key in args.models:
            if models.remove(key):
                print(f"{GREEN}removed{RESET} {key}")
            elif models.is_downloaded(key):
                print(f"{YELLOW}{key} lives outside our store, left alone:{RESET} "
                      f"{models.local_path(key)}")
            else:
                print(f"{DIM}{key} is not downloaded{RESET}")
        return 0

    if args.action == "use":
        key = args.models[0]
        spec = models.spec(key)
        if spec is None:
            print(f"{RED}unknown model {key}{RESET}", file=sys.stderr)
            return 1
        if spec.kind == models.LLM:
            print(f"{RED}{key} is an LLM, not a speech model{RESET}", file=sys.stderr)
            print(f"{DIM}`model use` sets speech.model. An LLM is named by a mode's "
                  f"step instead:{RESET}", file=sys.stderr)
            print(f"{DIM}  omavoi mode step <mode> add <llm-name>{RESET}", file=sys.stderr)
            return 1
        if not models.is_downloaded(key):
            print(f"{YELLOW}{key} is not downloaded yet, fetching{RESET}")
            models.pull(key)
        why = asr.why_unavailable(spec.backend)
        if why and not getattr(args, "force", False):
            print(f"{RED}{key} runs on {spec.backend}, which is not usable here"
                  f"{RESET}", file=sys.stderr)
            print(f"{DIM}{why}{RESET}", file=sys.stderr)
            print(f"{DIM}nothing was changed; repeat with --force to switch anyway"
                  f"{RESET}", file=sys.stderr)
            return 1
        config.set_path("speech.backend", spec.backend)
        config.set_path("speech.model", spec.key if spec.fmt == models.GGML else spec.id)
        print(f"{GREEN}ok{RESET} using {key} via {spec.backend} "
              f"{DIM}(restart the daemon to apply){RESET}")
        return 0
    return 1


def _why_not_that_hotkey(key: str, value: str) -> str:
    """Why this key name will not work, or "" if it will.

    validate() warned about it on the next load, by which time the daemon had
    already started with no hotkey at all — which from the outside looks
    exactly like a broken microphone.
    """
    if str(key) != "hotkey.key" or not str(value).strip():
        return ""
    from .hotkey import HotkeyUnavailable, key_code

    try:
        key_code(value)
    except HotkeyUnavailable as exc:
        return str(exc)
    except ModuleNotFoundError:
        return ""
    return ""


def _why_not_that_llm_model(key: str, value: str) -> str:
    """Why `llm.<name>.model = value` would not work, or "" if it would."""
    parts = str(key).split(".")
    if len(parts) != 3 or parts[0] != "llm" or parts[2] != "model":
        return ""
    # Only the local configuration runs weights from the catalogue. For the
    # API and the agent, `model` is the provider's own name — gpt-4o-mini,
    # haiku — and checking those against the catalogue refused every valid
    # value, which is how the remote API became impossible to configure.
    entry = (config.load().get("llm") or {}).get(parts[1]) or {}
    if str(entry.get("backend", "")).strip().lower() not in (
            "llama-local", "llama.cpp", "llamacpp"):
        return ""
    spec = models.spec(value)
    if spec is None:
        known = ", ".join(m.key for m in models.CATALOG if m.kind == models.LLM)
        return f"{value} is not a model in the catalogue. Local ones: {known}"
    if spec.kind != models.LLM:
        return f"{value} is a speech model, not an LLM"
    if not models.is_downloaded(value):
        gb = spec.size_mb / 1024
        return (f"{value} is not downloaded ({gb:.1f} GB). "
                f"Run: omavoi model pull {value}")
    return ""


def _legal_values(key: str, cfg: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    """The values a key accepts, when it accepts a fixed set of them.

    Returns (values, what) or ((), "").

    152 settable keys and four of them were checked. The rest took anything —
    `speech.backend = nonsens` was written happily and then stopped the daemon
    from starting at all, and `switching.mode = typo` fell back to default with
    one line in a journal nobody reads. Every set below is read from wherever
    the code already keeps it, so this table cannot drift away from the thing
    it is describing.
    """
    from . import asr, i18n
    from .llm import BACKENDS as LLM_BACKENDS

    parts = key.split(".")

    def at(*shape: str) -> bool:
        """`llm.*.backend` matches ("llm", "*", "backend")."""
        return (len(parts) == len(shape)
                and all(a == "*" or a == b for a, b in zip(shape, parts, strict=True)))

    if key == "speech.backend":
        return tuple(sorted(asr.BACKENDS)), "a speech engine"
    if at("llm", "*", "backend"):
        return tuple(sorted(LLM_BACKENDS)), "an LLM backend"
    if key == "hotkey.mode":
        return ("push_to_talk", "toggle"), "a hotkey behaviour"
    if key == "inject.method" or at("modes", "*", "inject"):
        return ("auto", "clipboard", "wtype", "xdotool"), "an injection route"
    if at("modes", "*", "rules", "punctuation"):
        return ("keep", "strip"), "a punctuation policy"
    if key == "ui.hud_dwell":
        return ("always", "changed", "never"), "an overlay dwell"
    if key == "ui.language":
        # "" follows the system locale, which is the shipped default.
        return ("", *i18n.LANGUAGES), "an interface language"
    if key == "speech.local_whisper.device":
        return ("auto", "cpu", "cuda"), "a device"
    if key == "switching.mode":
        return tuple(sorted(cfg.get("modes", {}))), "a mode that exists"
    return (), ""


def _why_not_that_choice(key: str, value: str) -> str:
    """Why this value is not one of the ones the key accepts."""
    cfg = config.load()
    legal, what = _legal_values(str(key), cfg)
    if not legal or value in legal:
        return ""
    shown = ", ".join(repr(v) if v == "" else v for v in legal)
    return f"{value!r} is not {what}. One of: {shown}"


def _why_not_that_provider(key: str, value: str) -> str:
    """Why `speech.api.provider = value` would not work, or "" if it would.

    The presets are the only thing this name selects, so a name outside them
    silently selects nothing: no base_url, no model, no key_env, and a remote
    engine that fails at the first take with no clue that a typo was the
    cause.
    """
    if str(key) != "speech.api.provider":
        return ""
    from .asr.api_whisper import PROVIDERS

    if value in PROVIDERS:
        return ""
    return (f"{value} is not a known provider. "
            f"One of: {', '.join(sorted(PROVIDERS))} "
            f"— or leave it and set speech.api.base_url yourself")


def _check_endpoint(name: str, entry: dict[str, Any], *, as_json: bool = False,
                    label: str = "") -> int:
    """Ask an OpenAI-compatible endpoint for its model list.

    One GET answers everything that can be wrong before a take: whether the
    host resolves, whether TLS holds, whether the key is accepted, whether the
    thing at the other end speaks this API at all — and it returns the model
    names, so the model does not have to be typed from memory. A chat
    completion would answer the same questions and bill for the privilege.
    """
    import httpx

    from . import secrets

    base = str(entry.get("base_url") or "").rstrip("/")
    if not base:
        out = {"ok": False,
               "error": f"{label or f'llm.{name}'}.base_url is not set"}
    else:
        key_env = str(entry.get("key_env", ""))
        key = secrets.resolve(key_env, str(entry.get("key_name", "") or name))
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            r = httpx.get(f"{base}/models", headers=headers, timeout=12.0)
            if r.status_code in (401, 403):
                out = {"ok": False, "status": r.status_code,
                       "error": ("the endpoint refused the key" if key else
                                 f"this endpoint needs a key; set {key_env or 'one'}")}
            elif r.status_code >= 400:
                out = {"ok": False, "status": r.status_code,
                       "error": f"{base}/models returned {r.status_code}"}
            else:
                body = r.json()
                got = body.get("data") if isinstance(body, dict) else body
                ids = [str(m.get("id", "")) for m in (got or []) if isinstance(m, dict)]
                out = {"ok": True, "status": r.status_code, "models": [i for i in ids if i],
                       "key": secrets.redact(key) if key_env else ""}
        except httpx.HTTPError as exc:
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        except ValueError:
            out = {"ok": False, "status": r.status_code,
                   "error": "the answer was not JSON — is this an OpenAI-compatible API?"}

    if as_json:
        print(json.dumps(out | {"name": name, "base_url": base}, ensure_ascii=False))
        return 0 if out.get("ok") else 1
    if out.get("ok"):
        models_seen = out.get("models") or []
        print(f"{GREEN}ok{RESET} {base} answered with {len(models_seen)} models")
        for m in models_seen[:12]:
            print(f"  {m}")
        if len(models_seen) > 12:
            print(f"{DIM}  … and {len(models_seen) - 12} more{RESET}")
        return 0
    print(f"{RED}{out.get('error')}{RESET}", file=sys.stderr)
    return 1


def _hotkey_report() -> dict[str, Any]:
    """Everything that can be wrong with the hotkey, answered at once.

    The key not working has had four different causes in this program and one
    message for all of them. Each line below is a separate question, so the
    answer says which one it is instead of naming the likeliest.
    """
    import grp
    import os

    from . import daemon as daemon_mod
    from .hotkey import HotkeyUnavailable, explain_missing, key_code

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


def cmd_speech(args: argparse.Namespace) -> int:
    """The remote speech endpoint: what it is set to, and whether it answers.

    The remote engine could be selected and never configured — the console
    offered the card and had no fields behind it, and there was no way to ask
    whether the endpoint worked either. Same two questions the remote LLM
    already answered, so the same two answers.
    """
    from .asr.api_whisper import PROVIDERS

    cfg = config.load()
    api = dict(cfg["speech"].get("api") or {})
    provider = str(api.get("provider", "") or "")
    preset = PROVIDERS.get(provider, {})
    # What the backend will actually use: the explicit value, else the
    # provider's. Reporting the config alone would say "not set" about a URL
    # that is about to work.
    entry = {
        "base_url": str(api.get("base_url", "") or preset.get("base_url", "")),
        "model": str(api.get("model", "") or preset.get("model", "")),
        "key_env": str(api.get("key_env", "") or preset.get("key_env", "")),
        "key_name": str(api.get("key_name", "") or provider or "speech-api"),
    }

    if args.action == "check":
        return _check_endpoint("api", entry, as_json=args.json,
                               label="speech.api")

    # "show"
    if args.json:
        print(json.dumps({"provider": provider, "providers": sorted(PROVIDERS),
                          "selected": cfg["speech"]["backend"] == "api",
                          **entry}, ensure_ascii=False, indent=2))
        return 0
    print(f"{BOLD}speech.api{RESET}")
    print(f"  provider   {provider or DIM + 'unset' + RESET}")
    for field in ("base_url", "model", "key_env"):
        shown = entry[field] or f"{DIM}unset{RESET}"
        explicit = "" if api.get(field) else f"  {DIM}(from {provider}){RESET}"
        print(f"  {field:<10} {shown}{explicit if entry[field] else ''}")
    print(f"  in use     {'yes' if cfg['speech']['backend'] == 'api' else 'no'}")
    print(f"{DIM}  omavoi speech check      ask the endpoint for its models{RESET}")
    return 0


def cmd_hotkey(args: argparse.Namespace) -> int:
    """Read one key press, or say why the key is not working."""
    from .hotkey import HotkeyUnavailable, capture

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
    from . import secrets

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


def cmd_llm(args: argparse.Namespace) -> int:
    """Inspect the three configurations a mode's steps reach for by name.

    Three and only three — the agent, the local model, one remote endpoint —
    so there is nothing to add or remove. Which weights the local one runs is
    a per-step choice: `omavoi mode step <mode> model <index> <key>`.
    """
    cfg = config.load()
    entries: dict[str, Any] = cfg.setdefault("llm", {})
    rest = list(args.rest)

    if args.action == "list":
        if args.json:
            print(json.dumps(entries, ensure_ascii=False, indent=2))
            return 0
        used: dict[str, list[str]] = {}
        for mode_name, mode in (cfg.get("modes") or {}).items():
            for step in mode.get("steps") or []:
                used.setdefault(str(step.get("llm", "")), []).append(mode_name)
        for name, entry in sorted(entries.items()):
            by = ", ".join(sorted(used.get(name, []))) or f"{DIM}unused{RESET}"
            print(f"  {name:12s} {entry.get('backend', '?'):14s} "
                  f"{entry.get('model', '')!s:24s} {by}")
        return 0

    if args.action == "check":
        name = rest[0] if rest else "api"
        entry = entries.get(name)
        if entry is None:
            print(f"{RED}no [llm.{name}]{RESET}", file=sys.stderr)
            return 1
        return _check_endpoint(name, entry, as_json=args.json)

    return 1


def cmd_config(args: argparse.Namespace) -> int:
    path = paths.config_file()
    if args.action == "path":
        print(path)
        return 0
    if args.action == "init":
        if path.exists() and not args.force:
            print(f"{YELLOW}{path} exists; pass --force to overwrite{RESET}", file=sys.stderr)
            return 1
        config.write(config.defaults(), path)
        print(f"{GREEN}wrote{RESET} {path}")
        return 0
    if args.action == "show":
        if args.json:
            print(json.dumps(config.load(), ensure_ascii=False, indent=2))
        else:
            print(config.dumps(config.load()))
        return 0
    if args.action == "get":
        try:
            print(json.dumps(config.get_path(config.load(), args.key), ensure_ascii=False))
        except KeyError:
            print(f"{RED}no such setting: {args.key}{RESET}", file=sys.stderr)
            return 1
        return 0
    if args.action == "set":
        # Pointing an LLM at weights that are not there is accepted by the
        # config and then falls through on every take, which reads as the LLM
        # step doing nothing rather than as a missing download.
        why = _why_not_that_hotkey(args.key, args.value) \
            or _why_not_that_llm_model(args.key, args.value) \
            or _why_not_that_provider(args.key, args.value) \
            or _why_not_that_choice(args.key, args.value)
        if why and not getattr(args, "force", False):
            print(f"{RED}{why}{RESET}", file=sys.stderr)
            print(f"{DIM}nothing was changed; repeat with --force to set it anyway"
                  f"{RESET}", file=sys.stderr)
            return 1
        try:
            value = config.set_path(args.key, args.value)
        except KeyError:
            print(f"{RED}no such setting: {args.key}{RESET}", file=sys.stderr)
            return 1
        except ValueError as exc:
            print(f"{RED}{exc}{RESET}", file=sys.stderr)
            return 1
        print(f"{GREEN}ok{RESET} {args.key} = {json.dumps(value, ensure_ascii=False)}")
        return 0
    if args.action == "edit":
        if not path.exists():
            config.write(config.defaults(), path)
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
        return subprocess.call([editor, str(path)])
    return 1


def cmd_dict(args: argparse.Namespace) -> int:
    cfg = config.load()
    dictionary: dict[str, str] = dict(cfg.setdefault("dictionary", {}).get("rules", {}))
    if args.action == "list":
        if args.json:
            order = sorted(dictionary, key=len, reverse=True)
            rows = []
            for i, src in enumerate(order):
                shadow = next((o for o in order[:i] if o.lower() in src.lower()
                               or src.lower().startswith(o.lower())), "")
                rows.append({"heard": src, "meant": dictionary[src], "shadowed_by": shadow})
            print(json.dumps({"rules": rows}, ensure_ascii=False, indent=2))
            return 0
        for src in sorted(dictionary, key=len, reverse=True):
            print(f"  {src}  ->  {dictionary[src]}")
        print(f"\n{DIM}{len(dictionary)} entries{RESET}")
        return 0
    if args.action == "add":
        if not args.heard or not args.meant:
            print(f"{RED}usage: omavoi dict add <heard> <meant>{RESET}", file=sys.stderr)
            return 1
        dictionary[args.heard] = args.meant
        cfg["dictionary"]["rules"] = dictionary
        config.write(cfg)
        print(f"{GREEN}ok{RESET} {args.heard} -> {args.meant}  "
              f"{DIM}(omavoi reload to apply){RESET}")
        return 0
    if args.action == "rm":
        if dictionary.pop(args.heard, None) is None:
            print(f"{DIM}{args.heard} is not in the dictionary{RESET}")
            return 1
        cfg["dictionary"]["rules"] = dictionary
        config.write(cfg)
        print(f"{GREEN}removed{RESET} {args.heard}")
        return 0
    return 1


def cmd_reload(args: argparse.Namespace) -> int:
    try:
        reply = daemon.request({"cmd": "reload"})
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
    from . import asr
    from .audio import Capture
    from .pipeline import Pipeline

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


_MODE_FIELDS = ("language", "speech_model", "prompt", "inject", "paste_key")


def _mode_wants(cfg: dict[str, Any], mode: Any) -> list[dict[str, Any]]:
    """The local LLM weights a mode would need resident on the GPU.

    Remote steps need no VRAM. The speech model is already loaded whenever the
    daemon is up, so it belongs to used_mb rather than here — counting it again
    would refuse every mode on a machine that is working fine.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in getattr(mode, "steps", []) or []:
        name = str(getattr(step, "llm", "") or "")
        entry = (cfg.get("llm") or {}).get(name) or {}
        backend = str(entry.get("backend", "")).strip().lower()
        if backend not in ("llama-local", "llama.cpp", "llamacpp"):
            continue
        key = str(entry.get("model", "") or "llm:qwen3-8b")
        if key in seen:
            continue
        seen.add(key)
        spec = models.spec(key)
        out.append({
            "key": key,
            "llm": name,
            "weights_mb": spec.size_mb if spec is not None else 0,
            "ctx_size": int(entry.get("ctx_size", 4096)),
            "gpu_layers": int(entry.get("n_gpu_layers", 99)),
        })
    return out


def _live_model_keys() -> set[str]:
    """What the daemon has resident right now, so it is not double-counted."""
    info = daemon.ping()
    engines = (info or {}).get("engines") or {}
    live = {str(e.get("model", "")) for e in (engines.get("llm") or []) if e.get("live")}
    speech = engines.get("speech") or {}
    if speech.get("live"):
        live.add(str(speech.get("model", "")))
    return {k for k in live if k}


def _reclaimable_mb(engines: dict[str, Any], wanted_keys: set[str]) -> int:
    """VRAM held by local LLM servers the target mode does not name.

    Switching modes unloads those, so counting them as taken would refuse the
    switch over memory the switch itself frees.
    """
    from . import gpu

    live = [e for e in (engines.get("llm") or [])
            if e.get("live") and not e.get("remote") and e.get("pid")]
    if not live:
        return 0
    by_pid = gpu.usage_by_pid()
    return sum(by_pid.get(int(e["pid"]), 0) for e in live
               if str(e.get("model", "")) not in wanted_keys)


def _mode_fit(cfg: dict[str, Any], mode: Any, engines: dict[str, Any] | None = None
              ) -> dict[str, Any]:
    from . import gpu

    if engines is None:
        engines = ((daemon.ping() or {}).get("engines") or {})
    wants = _mode_wants(cfg, mode)
    keys = {w["key"] for w in wants}
    return gpu.fits_chain(wants, _live_model_keys(),
                          _reclaimable_mb(engines, keys))


def cmd_mode(args: argparse.Namespace) -> int:
    from . import modes as modes_mod
    from .window import active_window

    cfg = config.load()
    table = cfg.setdefault("modes", {})
    rest = list(args.rest)

    def need(n: int, usage: str) -> bool:
        if len(rest) < n:
            print(f"{RED}usage: omavoi mode {usage}{RESET}", file=sys.stderr)
            return False
        return True

    def save(msg: str) -> int:
        config.write(cfg)
        # The daemon watches config.toml, so this applies on its own within
        # about a second. Saying otherwise sent people looking for a command
        # they did not need.
        print(f"{GREEN}ok{RESET} {msg}")
        return 0

    switching = cfg.setdefault("switching", {"by_window": False, "mode": "default"})

    if args.action == "use":
        force = bool(getattr(args, "force", False))
        if not need(1, "use <mode> [--force]   — the mode every take uses"):
            return 1
        name = rest[0]
        if name not in table:
            print(f"{RED}no such mode: {name}{RESET}", file=sys.stderr)
            return 1
        # Switching into a mode whose LLM will not fit does not fail at the
        # switch — it fails on the next take, minutes later, as a step that
        # silently falls through. Refuse here, where the cause is obvious.
        fit = _mode_fit(cfg, modes_mod.resolve(cfg, None, forced=name))
        if fit.get("known") and not fit.get("fits"):
            need_gb = fit["needed_mb"] / 1024
            free_gb = fit.get("available_mb", fit.get("free_mb", 0)) / 1024
            which = ", ".join(fit.get("pending") or [])
            print(f"{RED}{name} needs {need_gb:.1f} GB of VRAM and only "
                  f"{free_gb:.1f} GB is free{RESET}", file=sys.stderr)
            print(f"{DIM}it would have to load {which}{RESET}", file=sys.stderr)
            for h in fit.get("holders") or []:
                print(f"{DIM}  {h['used_mb'] / 1024:.1f} GB  {h['name']} "
                      f"(pid {h['pid']}){RESET}", file=sys.stderr)
            if not force:
                print(f"{DIM}close one of those, pick a mode with a smaller model, "
                      f"or repeat with --force{RESET}", file=sys.stderr)
                return 1
            print(f"{YELLOW}--force given, switching anyway{RESET}", file=sys.stderr)
        switching["mode"] = name
        if switching.get("by_window"):
            print(f"{YELLOW}note: switching by window is on, so this only applies "
                  f"where nothing matches{RESET}")
        return save(f"every take now uses {name}")

    if args.action == "auto":
        if not need(1, "auto on|off"):
            return 1
        want = rest[0].lower() in ("on", "true", "yes", "1")
        switching["by_window"] = want
        return save("mode follows the focused window" if want
                    else f"mode is fixed at {switching.get('mode', 'default')}")

    if args.action == "list":
        active = modes_mod.resolve(cfg, active_window())
        if args.json:
            from . import gpu

            # Resolved once: ping and nvidia-smi per mode would be a dozen
            # subprocesses for a list of six.
            live = _live_model_keys()
            engines = ((daemon.ping() or {}).get("engines") or {})
            rows = []
            for name in modes_mod.names(cfg):
                mode = modes_mod.resolve(cfg, None, forced=name)
                raw = table.get(name, {})
                wants = _mode_wants(cfg, mode)
                fit = gpu.fits_chain(wants, live,
                                     _reclaimable_mb(engines, {w["key"] for w in wants}))
                rows.append(mode.as_dict() | {
                    "match": list(raw.get("match") or []),
                    "prompt": str(raw.get("prompt", "") or ""),
                    "paste_key": str(raw.get("paste_key", "") or ""),
                    "active": name == active.name,
                    "fits": fit.get("fits", True),
                    "vram_known": fit.get("known", False),
                    "needs_mb": fit.get("needed_mb", 0),
                    "free_mb": fit.get("free_mb", 0),
                    "available_mb": fit.get("available_mb", fit.get("free_mb", 0)),
                    "pending": fit.get("pending") or [],
                })
            print(json.dumps({"active": active.name, "modes": rows,
                              "llm": sorted(cfg.get("llm", {})),
                              "fields": list(_MODE_FIELDS),
                              "switching": dict(switching)},
                             ensure_ascii=False, indent=2))
            return 0
        by_window = bool(switching.get("by_window"))
        for name in modes_mod.names(cfg):
            mode = modes_mod.resolve(cfg, None, forced=name)
            mark = f"{GREEN}*{RESET}" if name == active.name else " "
            match = ", ".join(table.get(name, {}).get("match") or []) or "fallback"
            chain = " -> ".join(["speech", *(st.llm for st in mode.steps)])
            trigger = match if by_window else f"{DIM}{match}{RESET}"
            print(f"{mark} {name:<12}{chain:<34}{DIM}{trigger}{RESET}")
        print()
        if by_window:
            print(f"{DIM}The mode follows the focused window.{RESET}")
        else:
            print(f"Every take uses {BOLD}{switching.get('mode', 'default')}{RESET}. "
                  f"{DIM}The window lists above are inert — omavoi mode auto on{RESET}")
        return 0

    if args.action == "show":
        name = (args.rest[0] if args.rest else "") or modes_mod.resolve(cfg, active_window()).name
        if name not in table:
            print(f"{RED}no such mode: {name}{RESET}", file=sys.stderr)
            return 1
        print(json.dumps(table[name], ensure_ascii=False, indent=2))
        return 0

    if args.action == "which":
        win = active_window()
        mode = modes_mod.resolve(cfg, win)
        # The config on disk and the daemon's copy of it can differ for a
        # moment, and only the daemon's answer is the one that types.
        live, reachable = "", True
        try:
            live = str(daemon.request({"cmd": "status"}, timeout=3).get("mode", ""))
        except (ConnectionError, OSError):
            reachable = False
        print(f"{mode.name}  {DIM}window={win.cls or '?'} "
              f"matched={mode.matched_on or '-'}{RESET}")
        if not reachable:
            print(f"{DIM}(daemon not running; this is the config on disk){RESET}")
        elif live and live != mode.name:
            print(f"{YELLOW}the daemon is still on {live} — it picks up a change "
                  f"within about a second{RESET}")
        elif live:
            print(f"{DIM}the daemon agrees{RESET}")
        return 0

    # -- mutations ---------------------------------------------------------

    if args.action == "new":
        if not need(1, "new <name> [copy-from]"):
            return 1
        name = rest[0]
        if name in table:
            print(f"{RED}{name} already exists{RESET}", file=sys.stderr)
            return 1
        source = rest[1] if len(rest) > 1 else ""
        if source and source not in table:
            print(f"{RED}no such mode to copy: {source}{RESET}", file=sys.stderr)
            return 1
        base = copy.deepcopy(table[source]) if source else {
            "match": [], "language": "", "prompt": "", "inject": "auto",
            "rules": dict(config.DEFAULTS["modes"]["default"]["rules"]), "steps": [],
        }
        base["match"] = [] if not source else list(base.get("match") or [])
        table[name] = base
        return save(f"created mode {name}" + (f" from {source}" if source else ""))

    if args.action == "rm":
        if not need(1, "rm <name>"):
            return 1
        name = rest[0]
        if name == "default":
            print(f"{RED}default is the fallback and cannot be removed{RESET}", file=sys.stderr)
            return 1
        if table.pop(name, None) is None:
            print(f"{DIM}no such mode: {name}{RESET}")
            return 1
        return save(f"removed mode {name}")

    if args.action == "set":
        if not need(3, "set <mode> <field> <value>   fields: " + ", ".join(_MODE_FIELDS)):
            return 1
        name, field, value = rest[0], rest[1], " ".join(rest[2:])
        if name not in table:
            print(f"{RED}no such mode: {name}{RESET}", file=sys.stderr)
            return 1
        if field not in _MODE_FIELDS:
            print(f"{RED}unknown field {field!r}; one of {', '.join(_MODE_FIELDS)}{RESET}",
                  file=sys.stderr)
            return 1
        # Refused where it is written, not warned about on the next load: a
        # mode whose speech_model does not exist keeps the previous weights and
        # sounds exactly like a mode that works.
        if field == "speech_model" and value.strip():
            spec = models.spec(value.strip())
            if spec is None or spec.kind != models.SPEECH:
                known = ", ".join(m.key for m in models.CATALOG if m.kind == models.SPEECH)
                print(f"{RED}{value} is not a speech model{RESET}", file=sys.stderr)
                print(f"{DIM}one of: {known}{RESET}", file=sys.stderr)
                return 1
            if not models.is_downloaded(value.strip()):
                print(f"{RED}{value} is not downloaded "
                      f"({spec.size_mb / 1024:.1f} GB){RESET}", file=sys.stderr)
                print(f"{DIM}omavoi model pull {value.strip()}{RESET}", file=sys.stderr)
                return 1
            # A mode changes the weights, never the engine: the running engine
            # reads one format, and the swap would be refused at mode-switch
            # time instead of here, which is the wrong place to find out.
            running = str(cfg["speech"].get("backend", ""))
            wants_ggml = running == "local-whispercpp"
            if (spec.fmt == models.GGML) != wants_ggml:
                print(f"{RED}{value} is a {spec.fmt} model and the engine in use "
                      f"({running}) reads {'ggml' if wants_ggml else 'ct2'}{RESET}",
                      file=sys.stderr)
                print(f"{DIM}a mode can change the weights, not the engine"
                      f"{RESET}", file=sys.stderr)
                return 1
        table[name][field] = value
        return save(f"{name}.{field} set")

    if args.action in ("match", "unmatch"):
        if not need(2, f"{args.action} <mode> <window-class> [...]"):
            return 1
        name, tokens = rest[0], rest[1:]
        if name not in table:
            print(f"{RED}no such mode: {name}{RESET}", file=sys.stderr)
            return 1
        current = list(table[name].get("match") or [])
        if args.action == "match":
            for t in tokens:
                if t not in current:
                    current.append(t)
        else:
            current = [c for c in current if c not in tokens]
        table[name]["match"] = current
        return save(f"{name}.match = {current or '[]'}")

    if args.action == "step":
        if not need(2, "step <mode> add|rm|prompt|llm|model [...]"):
            return 1
        name, op = rest[0], rest[1]
        if name not in table:
            print(f"{RED}no such mode: {name}{RESET}", file=sys.stderr)
            return 1
        steps = list(table[name].get("steps") or [])

        if op == "add":
            if len(rest) < 3:
                print(f"{RED}usage: omavoi mode step <mode> add <llm> [prompt]{RESET}",
                      file=sys.stderr)
                return 1
            llm = rest[2]
            if llm not in cfg.get("llm", {}):
                print(f"{RED}no [llm.{llm}] defined; see omavoi model list{RESET}",
                      file=sys.stderr)
                return 1
            # Never leave a step without instructions: an LLM handed a bare
            # transcript answers it, and the answer is what gets typed.
            steps.append({"llm": llm,
                          "prompt": " ".join(rest[3:]) or config.DEFAULT_STEP_PROMPT})
        elif op in ("rm", "prompt", "llm", "model"):
            if len(rest) < 3 or not rest[2].isdigit():
                print(f"{RED}usage: omavoi mode step <mode> {op} <index> [...]{RESET}",
                      file=sys.stderr)
                return 1
            index = int(rest[2])
            if not 0 <= index < len(steps):
                print(f"{RED}no step {index}; this mode has {len(steps)}{RESET}",
                      file=sys.stderr)
                return 1
            if op == "rm":
                steps.pop(index)
            elif op == "prompt":
                steps[index]["prompt"] = " ".join(rest[3:]) or config.DEFAULT_STEP_PROMPT
            elif op == "model":
                # Which weights this step runs, when it names the local
                # configuration. Empty goes back to the configuration's own.
                want = rest[3] if len(rest) > 3 else ""
                if want:
                    spec = models.spec(want)
                    if spec is None or spec.kind != models.LLM:
                        known = ", ".join(m.key for m in models.CATALOG
                                          if m.kind == models.LLM)
                        print(f"{RED}{want} is not an LLM in the catalogue{RESET}",
                              file=sys.stderr)
                        print(f"{DIM}one of: {known}{RESET}", file=sys.stderr)
                        return 1
                    if not models.is_downloaded(want):
                        print(f"{RED}{want} is not downloaded "
                              f"({spec.size_mb / 1024:.1f} GB){RESET}", file=sys.stderr)
                        print(f"{DIM}omavoi model pull {want}{RESET}", file=sys.stderr)
                        return 1
                    entry = cfg.get("llm", {}).get(steps[index].get("llm", ""), {})
                    if str(entry.get("backend", "")) not in (
                            "llama-local", "llama.cpp", "llamacpp"):
                        print(f"{RED}step {index} names "
                              f"{steps[index].get('llm')!r}, which is not the local "
                              f"configuration — only that one runs weights from the "
                              f"catalogue{RESET}", file=sys.stderr)
                        return 1
                if want:
                    steps[index]["model"] = want
                else:
                    # Inherit is the absence of an override, not an override
                    # that happens to be blank.
                    steps[index].pop("model", None)
            else:
                llm = rest[3] if len(rest) > 3 else ""
                if llm not in cfg.get("llm", {}):
                    print(f"{RED}no [llm.{llm}] defined{RESET}", file=sys.stderr)
                    return 1
                steps[index]["llm"] = llm
        else:
            print(f"{RED}unknown step op {op!r}: add, rm, prompt, llm, model{RESET}",
                  file=sys.stderr)
            return 1

        table[name]["steps"] = steps
        return save(f"{name} chain: " + " -> ".join(
            ["speech", *(s["llm"] + (f"({s['model'].replace('llm:', '')})"
                                     if s.get("model") else "")
                         for s in steps)]))

    print(f"{RED}unknown action {args.action!r}{RESET}", file=sys.stderr)
    return 1


def cmd_names(args: argparse.Namespace) -> int:
    from . import names as names_mod
    from .history import History

    cfg = config.load()
    entries = cfg.setdefault("dictionary", {}).setdefault("names", [])

    if args.action == "list":
        index = names_mod.NameIndex(cfg)
        if args.json:
            rows = []
            for e in index.entries:
                key = (names_mod.pinyin_key(e.name) if e.resolved_match() == "pinyin"
                       else names_mod.phonetic_key(e.name))
                rows.append(e.as_dict() | {"key": key})
            print(json.dumps({"names": rows, "seed": index.seed_text(),
                              "budget": index.budget}, ensure_ascii=False, indent=2))
            return 0
        if not index.entries:
            print(f"{DIM}no names yet — omavoi names add <name> [...]{RESET}")
            return 0
        for e in index.entries:
            state = f"{GREEN}matching{RESET}" if e.enabled else f"{DIM}seed only{RESET}"
            key = (names_mod.pinyin_key(e.name) if e.resolved_match() == "pinyin"
                   else names_mod.phonetic_key(e.name))
            print(f"  {e.name:<18}{key:<22}{e.resolved_match():<10}{state:<20}{DIM}{e.group}{RESET}")
        print(f"\n{DIM}{len(index.entries)} names · decoder prompt: {index.seed_text()[:60] or '(none)'}{RESET}")
        return 0

    if args.action == "add":
        if not args.names:
            print(f"{RED}usage: omavoi names add <name> [<name> ...]{RESET}", file=sys.stderr)
            return 1
        existing = {e.name for e in names_mod.load(cfg)}
        added = 0
        for name in args.names:
            name = name.strip()
            if not name or name in existing:
                continue
            entries.append({"name": name, "group": args.group, "seed": True, "enabled": False})
            added += 1
        config.write(cfg)
        print(f"{GREEN}added {added}{RESET} — seeded into the decoder prompt now.")
        print(f"{DIM}Sound matching stays off until `omavoi names dryrun` shows what it would do.{RESET}")
        return 0

    if args.action == "rm":
        target = set(args.names)
        kept = [e for e in entries if (e.get("name") if isinstance(e, dict) else e) not in target]
        removed = len(entries) - len(kept)
        cfg["dictionary"]["names"] = kept
        config.write(cfg)
        print(f"{GREEN}removed {removed}{RESET}")
        return 0

    if args.action in ("dryrun", "enable"):
        texts = [e.get("raw_text") or "" for e in History(cfg).iter_entries()]
        texts = [t for t in texts if t.strip()]
        if not texts:
            print(f"{YELLOW}no stored transcripts to test against yet{RESET}")
            return 1
        found = names_mod.dry_run(cfg, texts)
        print(f"{BOLD}{len(found)}{RESET} of {len(texts)} stored takes would change\n")
        for item in found[:20]:
            print(f"  {DIM}{item['before']}{RESET}")
            print(f"  {GREEN}{item['after']}{RESET}")
            print(f"    {DIM}{', '.join(h['found'] + ' -> ' + h['name'] for h in item['hits'])}{RESET}\n")
        if args.action == "dryrun":
            print(f"{DIM}Look these over. `omavoi names enable` turns matching on for all of them.{RESET}")
            return 0
        for e in entries:
            if isinstance(e, dict):
                e["enabled"] = True
        config.write(cfg)
        print(f"{GREEN}sound matching enabled for all names{RESET}")
        return 0
    return 1


def cmd_setup(args: argparse.Namespace) -> int:
    from . import setup as setup_mod

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


def cmd_inject(args: argparse.Namespace) -> int:
    """Put known text into the focused window, without saying anything.

    Injection failures are hard to pin down from dictation: you cannot tell a
    bad transcript from a paste that never landed. This does only the last
    step, with text you chose, so the question becomes one thing at a time.
    """
    import time as _time

    from .inject import Injector
    from .window import active_window

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
            reply = daemon.request({"cmd": "inject", "text": text}, timeout=120)
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


def cmd_doctor(args: argparse.Namespace) -> int:
    from . import cudaenv
    from .hotkey import find_devices, key_code

    cfg = config.load()
    ok = True

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        mark = f"{GREEN}ok  {RESET}" if good else f"{RED}FAIL{RESET}"
        print(f"  {mark} {label:<22}{detail}")

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
        from .hotkey import explain_missing

        code = key_code(cfg["hotkey"]["key"])
        devices = find_devices(code)
        # Not a guess. `id -nG` was the worst possible thing to point at
        # here: it reports the groups this session inherited, so someone who
        # ran usermod and did not log out sees `input` absent, concludes they
        # are not in the group, runs usermod again, and is told the same thing
        # forever. explain_missing asks the four questions separately.
        check(f"{cfg['hotkey']['key']} readable on", bool(devices),
              f"{len(devices)} device(s)" if devices
              else f"{RED}nothing{RESET} — "
                   + (explain_missing(code, cfg["hotkey"]["key"])
                      or "no reason could be determined"))
        for dev in devices:
            print(f"    {DIM}{dev.path}  {dev.name}{RESET}")
            dev.close()
    except Exception as exc:
        check("hotkey", False, str(exc))

    print(f"\n{BOLD}asr{RESET}")
    from . import asr as asr_mod

    backend_name = asr_mod.canonical(cfg["speech"]["backend"])
    check("backend", backend_name is not None, str(backend_name or cfg["speech"]["backend"]))

    if backend_name == "api":
        from . import secrets
        from .asr.api_whisper import PROVIDERS

        api = cfg["speech"]["api"]
        provider = api.get("provider", "openai")
        check("provider", provider in PROVIDERS or bool(api.get("base_url")), provider)
        key = secrets.resolve(
            api.get("key_env", "") or PROVIDERS.get(provider, {}).get("key_env", ""),
            api.get("key_name", "") or provider,
        )
        check("api key", bool(key), secrets.redact(key))
    elif backend_name == "local-whispercpp":
        from .asr.local_whispercpp import find_server

        server = find_server()
        check("whisper.cpp server", bool(server),
              server or f"{RED}not found — pacman -S whisper-cpp ggml-vulkan{RESET}")
        key = cfg["speech"]["model"]
        if not key.startswith("ggml:"):
            key = f"ggml:{key}"
        check(f"model {key}", models.is_downloaded(key),
              str(models.local_path(key) or f"{YELLOW}omavoi model pull {key}{RESET}"))
    else:
        info = cudaenv.diagnose()
        check("ctranslate2", "ctranslate2" in info,
              str(info.get("ctranslate2", info.get("ctranslate2_error"))))
        devices = int(info.get("cuda_devices", 0) or 0)
        check("cuda devices", devices > 0,
              f"{devices}" if devices else f"{YELLOW}none, will run on CPU{RESET}")
        check("cuda runtime libs", not info.get("preload_failures"),
              f"{info.get('preloaded_count', 0)} preloaded from wheels")
        key = cfg["speech"]["model"]
        check(f"model {key}", models.is_downloaded(key),
              str(models.local_path(key) or f"{YELLOW}omavoi model pull {key}{RESET}"))

    print(f"\n{BOLD}audio{RESET}")
    try:
        out = subprocess.run(["pactl", "get-default-source"], capture_output=True, timeout=2, check=False)
        source = out.stdout.decode().strip()
        check("default source", bool(source), source or f"{RED}none{RESET}")
    except (subprocess.SubprocessError, OSError) as exc:
        check("default source", False, str(exc))

    if args.mic:
        import time as _time

        from .audio import RingCapture

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
    info = daemon.ping()
    check("running", info is not None,
          f"pid {info['pid']}, {info['state']}" if info else f"{DIM}not running{RESET}")

    print()
    print(f"{GREEN}all good{RESET}" if ok else f"{YELLOW}problems above{RESET}")
    return 0 if ok else 1


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

    p = sub.add_parser("history", help="recent takes")
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

    p = sub.add_parser("hotkey", help="choose the push-to-talk key by pressing it")
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
