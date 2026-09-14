"""--no-color has to reach the module the output is actually formatted in.

It used to work by rewriting cli.py's own globals, which was enough only while
every command lived in cli.py. The moment they moved into their own modules
each took a copy of the escape codes, and the flag would have gone on
reporting success while the escapes kept coming out. This is the test that
would have caught that.
"""
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

MODULES = ["omavoi.commands.catalogue", "omavoi.commands.modes",
           "omavoi.commands.health", "omavoi.commands.keys",
           "omavoi.commands.settings", "omavoi.commands.words"]


def test_disable_reaches_every_command_module():
    for name in MODULES:
        importlib.import_module(name)
    term = importlib.import_module("omavoi.term")
    importlib.reload(term)
    # Re-import so the modules hold fresh, coloured copies.
    for name in MODULES:
        importlib.reload(sys.modules[name])
    assert any(getattr(sys.modules[n], "RED", "") for n in MODULES), \
        "no command module imports the codes any more — has the split changed?"
    term.disable()
    still = [n for n in MODULES if getattr(sys.modules[n], "RED", "") != ""]
    assert not still, f"these kept their escape codes: {still}"


def test_disable_does_not_invent_globals():
    import types
    term = importlib.import_module("omavoi.term")
    bare = types.ModuleType("omavoi.commands.notacommand")
    sys.modules["omavoi.commands.notacommand"] = bare
    try:
        term.disable()
        assert not hasattr(bare, "RED")
    finally:
        del sys.modules["omavoi.commands.notacommand"]


@pytest.mark.parametrize("argv", [
    ["--no-color", "config", "show"],
    ["--no-color", "llm", "list"],
    ["--no-color", "mode", "list"],
])
def test_the_flag_produces_no_escapes(argv):
    out = subprocess.run([sys.executable, "-m", "omavoi", *argv],
                         capture_output=True, check=False)
    assert b"\033[" not in out.stdout, "escape codes survived --no-color"
