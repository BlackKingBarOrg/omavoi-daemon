"""The registry's own methods must at least run.

`retain()` carried two references to a name that no longer existed after the
keys became "name@model", so it raised NameError — from reload(), which the
console calls after every settings change. Changing any setting killed the
command handler and the UI hung waiting for a reply that was never coming.
Nothing had ever called it in a test.
"""
from __future__ import annotations

from omavoi import config
from omavoi.llm import Registry


def _registry(home):
    return Registry(config.load())


def test_retain_runs_and_returns_names(home):
    reg = _registry(home)
    freed = reg.retain(set())
    assert isinstance(freed, list)
    assert all(isinstance(name, str) for name in freed)


def test_retain_with_everything_needed_frees_nothing(home):
    cfg = config.load()
    reg = Registry(cfg)
    assert reg.retain(set(cfg.get("llm", {}))) == []


def test_states_runs_for_every_configuration(home):
    cfg = config.load()
    states = _registry(home).states()
    assert {s["name"] for s in states} >= set(cfg.get("llm", {}))
    for s in states:
        # The fields the console reads off these.
        for field in ("name", "backend", "model", "remote", "live"):
            assert field in s, f"{s['name']} has no {field}"


def test_get_keys_on_name_and_model(home):
    """Two steps naming one configuration with different weights are two
    instances, not one that keeps changing under them."""
    reg = _registry(home)
    a = reg.get("local", "llm:qwen3-4b")
    b = reg.get("local", "llm:gemma-3-4b")
    c = reg.get("local", "llm:qwen3-4b")
    assert a is c, "the same name and model must give the same instance"
    assert a is not b, "a different model must give a different instance"


def test_update_survives_a_config_change(home):
    reg = _registry(home)
    reg.get("local", "llm:qwen3-4b")
    cfg = config.load()
    cfg["llm"]["local"]["model"] = "llm:gemma-3-4b"
    reg.update(cfg)
    assert reg.states()
