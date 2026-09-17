"""Every setting in DEFAULTS must be read by something.

This is the project's oldest bug wearing its plainest clothes. A key gets
added to the defaults with a comment explaining what it does, `config set`
accepts it, the console may even grow a control for it, and nothing anywhere
reads it. Four were found at once:

  ui.hud               the overlay could not be turned off
  ui.hud_dwell         three buttons in the settings page, one behaviour
  ui.hud_size          declared xs | s | m, implemented as the constant 28
  ui.hud_position      declared bottom | cursor | window, only bottom built
  hotkey.force_mode    validated by config.validate on every load, and the
                       daemon set forced_mode to "" unconditionally

and one hiding a level deeper:

  inject.avoid_wtype_on_xwayland   read from config into an Injector
                                   attribute that nothing then consulted

The second shape is why the first test here is not enough on its own: a
`.get("key")` proves the key was fetched, not that the fetch mattered.
"""
from __future__ import annotations

import ast
import collections
import pathlib
import re

from omavoi import config

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "omavoi"

# dictionary.rules and dictionary.names are the user's own content, and
# modes.* are mode definitions — data the program carries, not settings it
# consults by name.
DATA_PREFIXES = (("dictionary", "rules"), ("dictionary", "names"))


def _leaves(node: dict, prefix: str = ""):
    for key, value in node.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _leaves(value, name)
        else:
            yield name


def _is_setting(dotted: str) -> bool:
    parts = tuple(dotted.split("."))
    return not any(parts[: len(p)] == p for p in DATA_PREFIXES)


def test_no_setting_is_declared_and_never_read():
    """A key in DEFAULTS that no code fetches is a promise with nothing behind it."""
    text = "\n".join(p.read_text() for p in SRC.rglob("*.py"))
    # The defaults table itself does not count as a reader.
    without_defaults = text.replace(
        (SRC / "config.py").read_text(), "", 1
    ) + (SRC / "config.py").read_text().split("DEFAULTS", 1)[0]

    orphans = []
    for dotted in sorted(set(_leaves(config.DEFAULTS))):
        if not _is_setting(dotted):
            continue
        leaf = dotted.rsplit(".", 1)[-1]
        read = (re.search(rf'\["{re.escape(leaf)}"\]', without_defaults)
                or re.search(rf'\.get\(\s*"{re.escape(leaf)}"', without_defaults)
                or re.search(rf'"{re.escape(dotted)}"', without_defaults))
        if not read:
            orphans.append(dotted)
    assert not orphans, (
        "declared in DEFAULTS and read by nothing: " + ", ".join(orphans)
        + ".\nEither wire it up or take it out — a setting that does nothing "
        "is worse than a missing one, because it looks like it works."
    )


def test_no_config_value_is_stored_on_self_and_never_used():
    """The same bug one level in: fetched, assigned, and never consulted.

    `self.avoid_wtype_on_xwayland = bool(inject.get(...))` satisfied the test
    above and changed nothing about how anything behaved.
    """
    orphans = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        assigned: dict[str, int] = {}
        for node in ast.walk(tree):
            targets, value = [], None
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            if value is None:
                continue
            source = ast.unparse(value)
            if ".get(" not in source and "[" not in source:
                continue
            for target in targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    assigned.setdefault(target.attr, node.lineno)
        if not assigned:
            continue
        # Any load of self.X anywhere in the file, plus any mention of the
        # name elsewhere in the package — an attribute may legitimately be
        # read by another module.
        loads = collections.Counter(
            n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
            and isinstance(n.value, ast.Name) and n.value.id == "self")
        elsewhere = "\n".join(p.read_text() for p in SRC.rglob("*.py")
                              if p != path)
        for attr, line in sorted(assigned.items(), key=lambda kv: kv[1]):
            if loads[attr] == 0 and f".{attr}" not in elsewhere:
                orphans.append(f"{path.name}:{line} self.{attr}")
    assert not orphans, (
        "assigned from config and never read: " + ", ".join(orphans)
    )


def test_the_forced_mode_from_config_reaches_the_daemon():
    """hotkey.force_mode was validated on every load and applied never."""
    import inspect

    from omavoi.daemon import Daemon

    src = inspect.getsource(Daemon.__init__)
    assert 'self.forced_mode = str(cfg["hotkey"].get("force_mode"' in src, (
        "the daemon is back to hardcoding forced_mode"
    )


def test_a_numeric_fallback_agrees_with_the_shipped_default():
    """`cfg.get("key", 60)` where DEFAULTS says 150 documents two answers.

    The fallback never fires — config.load merges DEFAULTS into everything —
    so this is not a behaviour bug; it is the file saying one thing while the
    program does another, which is the shape everything else here turns out
    to be. Someone reading inject.py learned that the paste settle was 60 ms.
    It is 150, and the reason is written beside the 150.

    Numbers only. A string fallback is usually a display marker for absence —
    "?" for an unknown device, "-" for a method that was never chosen — and
    those are deliberately not the shipped value.
    """
    numbers = {}
    for leaf, dotted, value in _walk(config.DEFAULTS):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numbers.setdefault(leaf, []).append((dotted, value))

    wrong = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "config.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                continue
            entries = numbers.get(node.args[0].value)
            # Skip a leaf name that appears under more than one section: which
            # default was meant is then genuinely ambiguous.
            if not entries or len(entries) != 1:
                continue
            try:
                fallback = ast.literal_eval(node.args[1])
            except (ValueError, SyntaxError):
                continue
            if isinstance(fallback, bool) or not isinstance(fallback, (int, float)):
                continue
            dotted, shipped = entries[0]
            if float(shipped) != float(fallback):
                wrong.append(f"{path.name}:{node.lineno} {dotted} "
                             f"ships {shipped!r}, falls back to {fallback!r}")
    assert not wrong, "\n".join(wrong)


def _walk(node, prefix=""):
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _walk(value, dotted)
        else:
            yield key, dotted, value
