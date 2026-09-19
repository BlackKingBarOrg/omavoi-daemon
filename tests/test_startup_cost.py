"""The commands the console runs must not load the recording stack.

Quickshell opens the console by running seven `omavoi` processes, and the
plugin's every refresh runs more. They were each paying for numpy and evdev:

  omavoi --version      76 ms, of which numpy was 30
  config show --json    48 ms   (evdev, imported by config.validate)
  names list --json     82 ms   (history, and numpy behind it)
  setup --json          87 ms   (asr.local_whispercpp -> numpy)
  model list --json    124 ms   (asr.api_whisper, imported for PROVIDERS)

None of them records or decodes anything. The imports came in four ways —
a module-level import for a function-local use, a module-level import for
an annotation that `from __future__ import annotations` already made a
string, a function-level import at the top of a function that needs it in
one branch of four, and a validity check that no caller had asked for — and
each is a one-line regression to reintroduce.

Asserted in a subprocess because `sys.modules` in this one is already full
of everything the rest of the suite imported — and against a home that has
a config.toml, because `config.load` returns `defaults()` without calling
`validate` when there is no file. Written without one, this test passed with
the unconditional evdev check put back.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

# What the console runs when it opens, plus the floor.
CONSOLE_COMMANDS = [
    [],                                 # --version, the bare import
    ["config", "show", "--json"],
    ["dict", "list", "--json"],
    ["vocabulary", "list", "--json"],
    ["names", "list", "--json"],
    ["mode", "list", "--json"],
    ["history", "-n", "40", "--json"],
    ["setup", "--json"],
    ["model", "list", "--json"],
]

HEAVY = ("numpy", "evdev")

PROBE = """
import sys
sys.argv = ["omavoi"] + {argv!r}
import omavoi.cli
if {argv!r}:
    try:
        omavoi.cli.main()
    except SystemExit:
        pass
print("LOADED:" + ",".join(m for m in {heavy!r} if m in sys.modules))
"""


# A hotkey key name, so config.validate actually runs: it is the check that
# used to import evdev, and it is skipped entirely when the file is absent.
CONFIG = '[hotkey]\nkey = "RIGHTALT"\nmode = "push_to_talk"\n'


@pytest.mark.parametrize("argv", CONSOLE_COMMANDS,
                         ids=lambda a: " ".join(a) or "import only")
def test_no_console_command_loads_numpy_or_evdev(argv, home):
    cfg_dir = home / "config" / "omavoi"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.toml").write_text(CONFIG)

    proc = subprocess.run(
        [sys.executable, "-c", PROBE.format(argv=argv, heavy=HEAVY)],
        # check=False: the assertion below reports the child's stderr, which
        # is more use than a CalledProcessError with the output swallowed.
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("LOADED:")]
    assert line, f"probe printed nothing usable:\n{proc.stdout}\n{proc.stderr}"
    loaded = [m for m in line[-1][len("LOADED:"):].split(",") if m]
    assert not loaded, (
        f"`omavoi {' '.join(argv) or '--version'}` imported {', '.join(loaded)}. "
        "Neither is needed to print configuration: numpy is 25-30 ms and evdev "
        "25 ms, and this command runs every time the console opens. Look for a "
        "module-level import that only a function body needs, or an annotation "
        "that `from __future__ import annotations` already keeps lazy."
    )
