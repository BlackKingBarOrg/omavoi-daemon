"""Every test runs against its own empty home.

The paths module reads the XDG variables at call time, so redirecting them is
enough to keep a test off the real config — which matters more than usual
here, because the thing most worth testing is the code that writes it.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    for var, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                     ("XDG_STATE_HOME", "state"), ("XDG_CACHE_HOME", "cache"),
                     ("XDG_RUNTIME_DIR", "run")):
        d = tmp_path / sub
        d.mkdir()
        monkeypatch.setenv(var, str(d))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path
