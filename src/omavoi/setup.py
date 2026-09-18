"""What is still missing before dictation works, and how to fix each thing.

Both `omavoi setup` and the plugin's first-run screen read from here, so the
two never disagree about what is done. Nothing runs on its own: a step is
either already satisfied, or it hands back the exact command it would run.
"""

from __future__ import annotations

import grp
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

from . import models


@dataclass(slots=True)
class Step:
    key: str
    title: str
    done: bool
    detail: str = ""
    command: str = ""
    needs_root: bool = False
    # A step that can be skipped without blocking dictation.
    optional: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "title": self.title, "done": self.done,
            "detail": self.detail, "command": self.command,
            "needs_root": self.needs_root, "optional": self.optional, "note": self.note,
        }


@dataclass(slots=True)
class Report:
    steps: list[Step] = field(default_factory=list)

    @property
    def done(self) -> int:
        return sum(1 for s in self.steps if s.done)

    @property
    def blocking(self) -> list[Step]:
        return [s for s in self.steps if not s.done and not s.optional]

    @property
    def ready(self) -> bool:
        return not self.blocking

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "done": self.done,
            "total": len(self.steps),
            "steps": [s.as_dict() for s in self.steps],
        }


def _ggml_backends() -> list[str]:
    """Which ggml compute plugins are installed. Arch ships one package each."""
    from pathlib import Path

    out: list[str] = []
    for directory in (Path("/usr/lib/ggml"), Path("/usr/lib64/ggml")):
        if directory.is_dir():
            out.extend(p.name for p in directory.glob("libggml-*.so*"))
    return out


def _input_group() -> tuple[bool, bool]:
    """(listed in the group, held by this session).

    Two answers, because they disagree for as long as it takes to log out and
    they need opposite advice. Collapsing them into one bool is how the setup
    step told someone who had already run usermod to run usermod: the command
    succeeded, changed nothing, and the step still said "not in the input
    group" — a loop with no exit.
    """
    try:
        entry = grp.getgrnam("input")
    except KeyError:
        return (False, False)
    user = os.environ.get("USER") or ""
    return (user in entry.gr_mem, entry.gr_gid in os.getgroups())



def _daemon_reads_the_key() -> bool:
    """Whether the running daemon has a key listener with devices open.

    Asked of the daemon rather than of this process, because they can differ
    and only one of them matters. Absent or unreachable counts as no, which
    leaves the checklist to answer from this process's own access — the best
    evidence there is when there is nothing to ask.
    """
    try:
        from . import ipc

        info = ipc.ping()
    except Exception:
        return False
    hk = (info or {}).get("hotkey") or {}
    return bool(hk.get("enabled")) and bool(hk.get("devices"))

def _unit_active() -> bool:
    if shutil.which("systemctl") is None:
        return False
    try:
        out = subprocess.run(
            ["systemctl", "--user", "is-enabled", "omavoid.service"],
            capture_output=True, timeout=3, check=False,
        )
        return out.stdout.decode().strip() in ("enabled", "static", "linked")
    except (subprocess.SubprocessError, OSError):
        return False


def check(cfg: dict[str, Any]) -> Report:
    from . import asr
    from .asr.local_whispercpp import find_server

    steps: list[Step] = []
    backend = asr.canonical(cfg["speech"]["backend"]) or ""

    # 1. What we type with. Omarchy ships all of these, so this is normally green.
    # xdotool is not optional despite only mattering to X11 windows: without
    # it a take into one of those lands nowhere, and says nothing.
    missing = [t for t in ("pw-record", "wtype", "wl-copy", "hyprctl", "xdotool")
               if not shutil.which(t)]
    steps.append(Step(
        "tools", "Typing and audio tools", not missing,
        detail="pw-record, wtype, wl-clipboard, hyprctl, xdotool"
        if not missing else "missing: " + ", ".join(missing),
        command="sudo pacman -S --needed pipewire wtype wl-clipboard xdotool"
        if missing else "",
        needs_root=bool(missing),
        note="xdotool is what carries a paste into an X11 window — WeChat, "
             "Feishu, Steam. Without it those get nothing." if missing else "",
    ))

    # 2. The speech engine, which depends on which backend is selected.
    if backend == "local-whispercpp":
        server = find_server()
        # ggml loads its compute backends as plugins, and whisper still asks
        # for a CPU device for the tensors it does not offload. With only the
        # GPU plugin installed it aborts on GGML_ASSERT(device) part-way
        # through loading the model — so check for both, or the failure is a
        # core dump with no clue in it.
        backends = _ggml_backends()
        has_cpu = any("cpu" in b for b in backends)
        has_gpu = any(b.split("libggml-")[-1].split(".")[0] in
                      ("vulkan", "cuda", "hip", "sycl", "metal") for b in backends)
        ok = bool(server) and has_cpu
        if not server:
            detail = "whisper.cpp is not installed"
        elif not has_cpu:
            detail = ("whisper.cpp is installed but no CPU ggml backend is — "
                      "it will abort while loading the model")
        else:
            # Naming every plugin listed a dozen CPU microarchitectures twice,
            # on the first screen a new user reads. What matters is which
            # accelerators are there, not which -march variants.
            kinds = sorted({
                b.split("libggml-")[-1].split(".")[0].split("-")[0] for b in backends
            })
            detail = f"{server}, backends: {', '.join(kinds) or 'none'}"
        steps.append(Step(
            "engine", "Speech engine (Vulkan)", ok,
            detail=detail,
            command="sudo pacman -S --needed whisper-cpp ggml ggml-vulkan",
            needs_root=True,
            # `ggml` and not `ggml-cpu`. Arch once shipped the CPU backend as
            # its own optional package and this said so; it is folded into
            # `ggml` now, which declares `provides ggml-cpu` and `conflicts
            # ggml-cpu` both. Naming the old one pins a stale 0.21.0 split
            # package that cannot coexist with the `ggml` every one of these
            # depends on, and pacman rejects the whole transaction with
            # "unresolvable package conflicts". It is also no longer possible
            # to be missing: whisper-cpp depends on it.
            note="About 10 MB. ggml carries the CPU backend, which whisper "
                 "needs for the tensors it does not offload — it comes with "
                 "whisper-cpp either way, and is listed so the reason is "
                 "visible."
                 + ("" if has_gpu else " Swap ggml-vulkan for ggml-cuda or "
                    "ggml-hip if you would rather use the vendor backend."),
        ))
    else:
        steps.append(Step("engine", "Speech engine (remote API)", True,
                          detail=f"provider {cfg['speech']['api'].get('provider', '?')}"))

    # 3. Weights. Never shipped: far too large for a package or a plugin repo.
    key = cfg["speech"]["model"]
    have = models.is_downloaded(key)
    spec = models.spec(key)
    size = f"{spec.size_mb / 1024:.1f} GB" if spec else "unknown size"
    steps.append(Step(
        "model", f"Model weights ({key})", have,
        detail=str(models.local_path(key)) if have else f"not downloaded, {size}",
        # Quoted because `omavoi setup --run` passes this to a shell, and
        # the key is a config value. `config set speech.model` refuses
        # anything outside the catalogue now, which is the real fix; this is
        # the one that holds when a config file is edited by hand.
        command=f"omavoi model pull {shlex.quote(key)}",
    ))

    # 3b. The LLM engine, but only when a mode actually reaches for a local
    # one. Nothing needs it for plain dictation, and the failure it causes is
    # the quietest in the program: the step falls through, the previous text is
    # kept, and the take is indistinguishable from one with no LLM at all.
    wants_local = sorted({
        name for name, entry in (cfg.get("llm") or {}).items()
        if str(entry.get("backend", "")).strip().lower()
        in ("llama-local", "llama.cpp", "llamacpp")
        and any(str(step.get("llm", "")) == name
                for mode in (cfg.get("modes") or {}).values()
                for step in (mode.get("steps") or []))
    })
    if wants_local:
        from .llm.llama_local import find_server

        binary = find_server()
        used_by = ", ".join(wants_local)
        steps.append(Step(
            "llm-engine", "LLM engine (llama.cpp)", binary is not None,
            detail=binary if binary else f"llama-server is not installed, needed by {used_by}",
            command="sudo pacman -S --needed llama-cpp",
            needs_root=True,
            # Optional: plain dictation does not need it. Without this the
            # console hid its own tabs behind the checklist because one engine
            # nobody had asked for yet was absent.
            optional=True,
            note="About 7 MB. Without it a mode's LLM step falls through and the "
                 "take arrives as plain dictation with nothing on screen to say why "
                 "— the reason is in the History tab.",
        ))

    # 4. The hotkey. This is the one step that cannot be finished in place.
    listed, held = _input_group()
    # The daemon first, and for the fourth time in this program: a step that
    # reports "not done" about a feature that is working sends the user to fix
    # something that is not broken. This process not holding the group says
    # nothing about the daemon's access — the daemon can have been started
    # through newgrp, or simply started after the group was granted.
    if _daemon_reads_the_key():
        steps.append(Step(
            "hotkey", f"Hotkey ({cfg['hotkey']['key']} via evdev)", True,
            detail="the daemon is reading it",
            command="",
            needs_root=False,
            optional=True,
            note="",
        ))
    elif listed and not held:
        # A group is granted at login, so this session cannot see it however
        # many times usermod is run. The daemon can, though: install.sh starts
        # it through newgrp, which re-reads /etc/group. Offer that when the
        # plugin is here to offer it, and the README's recipe when it is not.
        installer = os.path.expanduser(
            "~/.config/omarchy/plugins/ai.bkblab.omavoi/install.sh")
        steps.append(Step(
            "hotkey", f"Hotkey ({cfg['hotkey']['key']} via evdev)", False,
            detail="in the input group, but this session started before that",
            command=installer if os.path.exists(installer) else "",
            needs_root=False,
            optional=True,
            note=("Starts the daemon through newgrp, so it has the group now; "
                  "logging out and back in also works."
                  if os.path.exists(installer) else
                  "Nothing left to install. Start the daemon through newgrp — "
                  "the recipe is in the README under \"the input group\" — or "
                  "log out and back in."),
        ))
    else:
        steps.append(Step(
            "hotkey", f"Hotkey ({cfg['hotkey']['key']} via evdev)", held,
            detail="in the input group" if held else "not in the input group",
            command="sudo usermod -aG input $USER",
            needs_root=True,
            optional=True,
            note="A group is granted at login. The plugin's install.sh starts the "
                 "daemon through newgrp so the key works at once; failing that, bind a "
                 "non-modifier key such as F9 in Hyprland until your next login — "
                 "modifier keys cannot be bound that way, because pressing one fires "
                 "the release binding at once.",
        ))

    # 5. Run it at login.
    #
    # The unit ships with the plugin rather than with this package, so someone
    # who installed the daemon by hand has nothing to enable — and this step
    # used to hand them `systemctl --user enable --now omavoid.service`
    # regardless, which answers "Unit omavoid.service does not exist". That is
    # the last step of the manual install, and the manual install exists for
    # people who want the daemon and none of the desktop pieces.
    have_unit = os.path.exists(os.path.join(
        os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
        "systemd", "user", "omavoid.service"))
    steps.append(Step(
        "service", "Start at login", _unit_active(),
        detail=("omavoid.service is enabled" if _unit_active()
                else "not enabled" if have_unit
                else "no unit file yet"),
        command=(
            "systemctl --user enable --now omavoid.service" if have_unit else
            "curl -fsSL -o ~/.config/systemd/user/omavoid.service --create-dirs "
            "https://raw.githubusercontent.com/BlackKingBarOrg/"
            "omavoi-shell-plugin/master/omavoid.service"
            " && systemctl --user daemon-reload"
            " && systemctl --user enable --now omavoid.service"
        ),
        optional=True,
        note=("" if have_unit else
              "The unit ships with the plugin, which installs it for you. "
              "This fetches just the unit, for a daemon-only install."),
    ))
    return Report(steps)


