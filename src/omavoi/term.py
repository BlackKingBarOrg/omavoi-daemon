"""The six escape codes, and the one way to turn them off.

They lived in cli.py, and `--no-color` worked by rewriting that module's own
globals. That is exactly as far as it reached: the moment a command moved into
its own module it would take a copy of the codes with it, and the flag would
go on reporting success while the escapes kept coming out.

So the names are still imported plainly — 454 sites read `RED` and not
`term.RED`, and they should stay that way — and `disable()` reaches every
module that imported them. Walking sys.modules is not elegant; it is, however,
the thing that is true, and it cannot fall out of step with a list somebody
forgets to add a module to.
"""

from __future__ import annotations

import sys

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

_NAMES = ("GREEN", "RED", "YELLOW", "DIM", "BOLD", "RESET")


def disable() -> None:
    """Blank the codes here and in every omavoi module that imported them."""
    blanks = dict.fromkeys(_NAMES, "")
    globals().update(blanks)
    for name, module in list(sys.modules.items()):
        if not (name == "omavoi" or name.startswith("omavoi.")) or module is None:
            continue
        if module is sys.modules[__name__]:
            continue
        ns = getattr(module, "__dict__", None)
        if ns is None:
            continue
        # Only where the module actually imported them, so this cannot
        # invent globals in a module that never wanted any.
        for key in _NAMES:
            if key in ns:
                ns[key] = ""
