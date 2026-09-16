"""Models, weights and the endpoints that serve them.

Lifted out of cli.py, which had grown to 1,949 lines and four
responsibilities. Nothing here changed on the way across.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .. import asr, config, ipc, models
from ..term import BOLD, DIM, GREEN, RED, RESET, YELLOW


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


def _is_remote(backend: str, base_url: str) -> bool:
    """Whether using this model sends text off the machine."""
    from ..llm import ON_MACHINE

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
    from .. import gpu

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
            from .. import gpu, i18n

            # The console renders these notes verbatim, so they are translated
            # here rather than there: the text lives beside the catalogue.
            lang = i18n.ui_lang(cfg)

            # The config says which model was asked for; only the daemon knows
            # which one is loaded, and the two differ after every edit until a
            # restart. Absent daemon leaves `engines` null rather than implying
            # nothing is running.
            live = ipc.ping()
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
                # Bytes so far when one is in flight, so a three-gigabyte wait
                # can show progress instead of the word "downloading".
                "bytes_now": models.bytes_in_flight(spec.key),
                    "path": str(models.local_path(spec.key) or ""),
                    "ours": models.owned_by_us(spec.key),
                    "active": (spec.key in llm_chosen
                               if spec.kind == models.LLM
                               else spec.key == active
                                    or (spec.fmt == models.CT2
                                        and spec.id == active)),
                    "running": _is_running(spec),
                })
            from .. import gpu, secrets

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
                    # Which of the two it came from. The environment wins over
                    # the file, so a stale variable silently beats the key you
                    # just pasted — and "a key is stored" said nothing about
                    # which one was in use.
                    "key_source": secrets.source_of(key_env, str(entry.get("key_name", "") or name)),
                    "used_by": sorted(used_by.get(name, [])),
                    **_llm_live(engines, name),
                })

            # The remote speech endpoint, in the same shape as an llm entry.
            # It was absent entirely, so the console could offer the engine and
            # then had nothing to configure it with — you could select "remote
            # API" for speech and there was nowhere to put the URL.
            from ..asr.api_whisper import PROVIDERS

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
                "key_source": secrets.source_of(s_key_env, s_key_name),
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

    from .. import secrets

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


def cmd_speech(args: argparse.Namespace) -> int:
    """The remote speech endpoint: what it is set to, and whether it answers.

    The remote engine could be selected and never configured — the console
    offered the card and had no fields behind it, and there was no way to ask
    whether the endpoint worked either. Same two questions the remote LLM
    already answered, so the same two answers.
    """
    from ..asr.api_whisper import PROVIDERS

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


def _why_not_that_provider(key: str, value: str) -> str:
    """Why `speech.api.provider = value` would not work, or "" if it would.

    The presets are the only thing this name selects, so a name outside them
    silently selects nothing: no base_url, no model, no key_env, and a remote
    engine that fails at the first take with no clue that a typo was the
    cause.
    """
    if str(key) != "speech.api.provider":
        return ""
    from ..asr.api_whisper import PROVIDERS

    if value in PROVIDERS:
        return ""
    return (f"{value} is not a known provider. "
            f"One of: {', '.join(sorted(PROVIDERS))} "
            f"— or leave it and set speech.api.base_url yourself")
