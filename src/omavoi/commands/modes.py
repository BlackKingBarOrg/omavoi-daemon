"""A mode is a chain: the weights, the rules and the steps.

Lifted out of cli.py, which had grown to 1,949 lines and four
responsibilities. Nothing here changed on the way across.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from typing import Any

from .. import config, i18n, ipc, models
from ..term import BOLD, DIM, GREEN, RED, RESET, YELLOW

_MODE_FIELDS = ("language", "speech_model", "prompt", "inject", "paste_key", "newline_key")
# The rules a mode carries, for `set <mode> rules.<key> <value>`.
_RULE_FLAGS = ("hallucinations", "fillers", "dictionary", "names", "vocabulary", "cjk_spacing")
_RULE_KEYS = (*_RULE_FLAGS, "punctuation", "joiner")


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
    info = ipc.ping()
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
    from .. import gpu

    live = [e for e in (engines.get("llm") or [])
            if e.get("live") and not e.get("remote") and e.get("pid")]
    if not live:
        return 0
    by_pid = gpu.usage_by_pid()
    return sum(by_pid.get(int(e["pid"]), 0) for e in live
               if str(e.get("model", "")) not in wanted_keys)


def _mode_fit(cfg: dict[str, Any], mode: Any, engines: dict[str, Any] | None = None
              ) -> dict[str, Any]:
    from .. import gpu

    if engines is None:
        engines = ((ipc.ping() or {}).get("engines") or {})
    wants = _mode_wants(cfg, mode)
    keys = {w["key"] for w in wants}
    return gpu.fits_chain(wants, _live_model_keys(),
                          _reclaimable_mb(engines, keys))


def cmd_mode(args: argparse.Namespace) -> int:
    from .. import modes as modes_mod
    from ..window import active_window

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
            from .. import gpu

            # Resolved once: ping and nvidia-smi per mode would be a dozen
            # subprocesses for a list of six.
            live = _live_model_keys()
            engines = ((ipc.ping() or {}).get("engines") or {})
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
                              "switching": dict(switching), "vocabulary_supported": True,
                              "post_enabled": cfg.get("post", {}).get("enabled", True)},
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
            live = str(ipc.request({"cmd": "status"}, timeout=3).get("mode", ""))
        except (ConnectionError, OSError):
            reachable = False
        # --json was accepted here and ignored, so a script that asked for
        # it got two lines of prose and parsed the word "the".
        if args.json:
            print(json.dumps({"mode": mode.name, "window": win.cls or "",
                              "matched": mode.matched_on or "", "daemon": live,
                              "reachable": reachable}, ensure_ascii=False))
            return 0
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
        # rules.<key>, so a rule can be set where the other fields are set.
        # It could only be reached through `omavoi config set
        # modes.<mode>.rules.<key>`, which nothing pointed at -- getting a
        # two-line LLM answer to survive to the window meant knowing that
        # `joiner = keep` existed and where it lived.
        if field.startswith("rules."):
            key = field[len("rules."):]
            if key not in _RULE_KEYS:
                print(f"{RED}unknown rule {key!r}; one of {', '.join(_RULE_KEYS)}{RESET}",
                      file=sys.stderr)
                return 1
            rules = dict(table[name].get("rules") or {})
            low = value.strip().lower()
            if key in _RULE_FLAGS:
                if low not in ("on", "off", "true", "false", "yes", "no", "1", "0"):
                    print(f"{RED}{key} is on or off{RESET}", file=sys.stderr)
                    return 1
                rules[key] = low in ("on", "true", "yes", "1")
            elif key == "punctuation":
                if low not in ("keep", "strip"):
                    print(f"{RED}punctuation is keep or strip{RESET}", file=sys.stderr)
                    return 1
                rules[key] = low
            else:  # joiner: keep, or the literal text a newline becomes
                rules[key] = "keep" if low == "keep" else value
            table[name]["rules"] = rules
            return save(f"{name}.rules.{key} set")
        if field not in _MODE_FIELDS:
            print(f"{RED}unknown field {field!r}; one of {', '.join(_MODE_FIELDS)} "
                  f"or rules.<{'|'.join(_RULE_KEYS)}>{RESET}",
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
                          "prompt": (" ".join(rest[3:])
                                     or config.default_step_prompt(i18n.ui_lang(cfg)))})
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
                steps[index]["prompt"] = (
                    " ".join(rest[3:])
                    or config.default_step_prompt(i18n.ui_lang(cfg)))
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
