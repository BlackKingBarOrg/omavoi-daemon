"""The coding agent Omarchy is already set up with.

Omarchy has a notion of a default agent — `omarchy default agent` picks one of
pi, omp, opencode, claude, codex, grok, gemini, copilot or crush, and writes it
to ~/.config/omarchy/defaults/agent. It deliberately picks none for you. So
this backend asks which one is configured rather than assuming, and an earlier
version of it that ran `claude` unconditionally was wrong about the platform,
not just about the name.

What it cannot assume is how to call one. These are interactive coding agents
first, and their non-interactive modes disagree: some take the prompt as an
argument, some read stdin, one appends stdin to the argument. INVOCATIONS holds
only the shapes read out of each agent's own --help on a machine that had all
nine installed; the rest is `argv` in the config, because guessing an argv is
how you get a backend that looks supported and returns nothing.

No key is needed either way — the agent is already logged in — at the cost of
several seconds of process startup per take.

One thing about these shapes is worth saying out loud, because this program
refuses it elsewhere: an agent whose only non-interactive mode takes the
prompt as an argument gets the transcript in its argv, and argv is readable
from /proc by every process running as this user for as long as the command
lives. `omavoi secrets set` reads a key from stdin for exactly that reason.
Four of the seven shapes below are argv-only — codex, grok, crush, copilot —
so `state()` reports it per route and `omavoi llm list` prints it, rather
than leaving it to be discovered. It is computed from the template, not
listed, so a user-supplied `argv` is judged the same way.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .base import LlmResult

log = logging.getLogger(__name__)

# Placeholders an argv template may use:
#   {system}    the mode's prompt
#   {text}      the transcript
#   {combined}  the two joined, for agents with nowhere to put a system prompt
#
# "stdin" says where the transcript goes when it is not in argv.
INVOCATIONS: dict[str, dict[str, Any]] = {
    "claude": {"argv": ["-p", "--allowed-tools", "", "--system-prompt", "{system}"],
               "stdin": "{text}"},
    "pi": {"argv": ["--print"], "stdin": "{combined}"},
    # -p is the prompt and stdin is appended to it, which is exactly the split
    # this needs.
    "gemini": {"argv": ["--prompt", "{system}"], "stdin": "{text}"},
    "codex": {"argv": ["exec", "{combined}"]},
    "grok": {"argv": ["--single", "{combined}"]},
    "crush": {"argv": ["run", "{combined}"]},
    "copilot": {"argv": ["--prompt", "{combined}"]},
    # Deliberately absent: opencode offers both `run` and `--prompt` and it was
    # not clear from its help which takes a bare prompt, and `omp --help` did
    # not return at all. Both work through `argv` once someone has checked.
}

_AGENT_FILE = "defaults/agent"


def puts_transcript_in_argv(entry: dict[str, Any], agent: str = "") -> bool:
    """Whether this configuration puts the dictated text on a command line.

    A module-level function because two callers need the answer without a
    backend in hand — the `[llm]` payload the console reads, and `omavoi llm
    list` — and a second copy of the rule is how the two would come to
    disagree.

    Computed from the template rather than listed, so a user-supplied `argv`
    is judged the same way as the shapes above.
    """
    argv = entry.get("argv")
    if not argv:
        agent = str(entry.get("agent", "") or agent or configured_agent())
        argv = (INVOCATIONS.get(agent) or {}).get("argv") or []
    return any("{text}" in str(a) or "{combined}" in str(a) for a in argv)


def configured_agent() -> str:
    """The agent Omarchy is set to, or "" if the user has chosen none."""
    from .. import paths

    try:
        base = Path(paths.config_file()).parent.parent / "omarchy" / _AGENT_FILE
        if base.exists():
            return base.read_text().strip().split("\n")[0].strip()
    except OSError:
        pass
    found = shutil.which("omarchy-default-agent")
    if not found:
        return ""
    try:
        out = subprocess.run([found], capture_output=True, timeout=5, check=False)
        return out.stdout.decode().strip()
    except (subprocess.SubprocessError, OSError):
        return ""


class AgentCliBackend:
    """Runs the configured coding agent once per take, non-interactively."""

    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        self.backend = "agent-cli"
        # Empty follows Omarchy. Naming one pins this entry to it, which is
        # what you want if a mode depends on a particular agent's behaviour.
        self.agent = str(cfg.get("agent", "") or "")
        self.model = str(cfg.get("model", "") or "")
        self.timeout = float(cfg.get("timeout", 90.0))
        self.argv_template: list[str] = list(cfg.get("argv") or [])
        self.stdin_template = str(cfg.get("stdin", "") or "")

    # -- what it is ---------------------------------------------------------

    def which(self) -> str:
        return self.agent or configured_agent()

    def transcript_in_argv(self) -> bool:
        """Whether this route puts the dictated text on a command line.

        Readable from /proc by anything running as this user while the
        command lives. Unavoidable for an agent whose non-interactive mode
        has nowhere else to take a prompt; worth knowing before dictating
        something into it either way.
        """
        return puts_transcript_in_argv(
            {"argv": self.argv_template, "agent": self.which()}, self.which())

    def why_not(self) -> str:
        agent = self.which()
        if not agent:
            return ("Omarchy has no default agent set. Run `omarchy default agent`, "
                    "or name one with llm.%s.agent" % self.name)
        if shutil.which(agent) is None:
            return f"{agent} is not on PATH"
        if not self.argv_template and agent not in INVOCATIONS:
            return (f"how to call {agent} non-interactively is not known here; "
                    f"set llm.{self.name}.argv (use {{system}}, {{text}} or "
                    f"{{combined}})")
        return ""

    def state(self) -> dict[str, Any]:
        agent = self.which()
        problem = self.why_not()
        return {
            "name": self.name,
            "backend": self.backend,
            "engine": agent or "agent-cli",
            "model": self.model,
            # The process is local; the inference is not.
            "remote": True,
            "live": not problem,
            "url": shutil.which(agent) or "" if agent else "",
            "pid": 0,
            "problem": problem,
            "transcript_in_argv": self.transcript_in_argv(),
        }

    def describe(self) -> str:
        st = self.state()
        agent = self.which() or "no agent set"
        return f"{self.name}: {agent} {self.model}".rstrip() + f" ({st['url'] or 'unavailable'})"

    def close(self) -> None:
        """Nothing stays resident: each take is one process."""

    # -- inference ----------------------------------------------------------

    def _plan(self, system: str, text: str) -> tuple[list[str], bytes]:
        agent = self.which()
        shape = ({"argv": self.argv_template, "stdin": self.stdin_template}
                 if self.argv_template else INVOCATIONS[agent])
        combined = (system.strip() + "\n\n" + text).strip() if system.strip() else text

        def fill(s: str) -> str:
            return (s.replace("{system}", system)
                     .replace("{combined}", combined)
                     .replace("{text}", text))

        argv = [shutil.which(agent) or agent]
        if self.model:
            argv += ["--model", self.model]
        argv += [fill(a) for a in shape.get("argv", [])]
        stdin = fill(str(shape.get("stdin", "") or "")).encode()
        return argv, stdin

    def complete(self, system: str, text: str, *, timeout: float = 0.0) -> LlmResult:
        started = time.monotonic()
        problem = self.why_not()
        if problem:
            return LlmResult("", self.model or self.which(), self.backend, error=problem)

        argv, stdin = self._plan(system, text)
        try:
            proc = subprocess.run(
                argv, input=stdin, capture_output=True,
                timeout=timeout or self.timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return LlmResult("", self.model or self.which(), self.backend,
                             error=f"{self.which()} did not answer within "
                                   f"{timeout or self.timeout:.0f}s")
        except OSError as exc:
            return LlmResult("", self.model or self.which(), self.backend, error=str(exc))

        out = proc.stdout.decode(errors="replace").strip()
        if proc.returncode != 0 or not out:
            # The agent's own last line is more use than a return code.
            err = proc.stderr.decode(errors="replace").strip().split("\n")[-1:]
            return LlmResult(
                "", self.model or self.which(), self.backend,
                error=(err[0] if err and err[0] else
                       f"{self.which()} exited {proc.returncode} with no output"),
            )
        return LlmResult(out, self.model or self.which(), self.backend,
                         seconds=time.monotonic() - started)
