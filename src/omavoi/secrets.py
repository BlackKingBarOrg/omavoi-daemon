"""API keys: from the environment, or from a 0600 file we never log."""

from __future__ import annotations

import logging
import os
import re
import stat
import tomllib

from . import paths

log = logging.getLogger(__name__)


def store(name: str, value: str) -> None:
    """Write one key into the 0600 file, leaving the others alone.

    Never through a command line: a value in argv is readable from /proc by
    every process running as this user for as long as the command lives. The
    only caller reads it from stdin.
    """
    path = paths.secrets_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = load_file()
    if value:
        stored[str(name)] = str(value)
    else:
        stored.pop(str(name), None)
    body = "".join(
        f'{k} = "{v.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"\n'
        for k, v in sorted(stored.items())
    )
    # Created 0600 before anything is in it, rather than written and then
    # chmodded, which leaves a window where it is readable.
    tmp = path.with_suffix(".toml.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("# Written by `omavoi secrets set`. Never in config.toml.\n")
            fh.write(body)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def source_of(key_env: str = "", key_name: str = "") -> str:
    """Where a key would come from: "env", "file", or "" if there is none.

    Worth saying out loud, because the environment wins over the file — a
    stale env var silently beats the key you just pasted.
    """
    if key_env and os.environ.get(key_env):
        return "env"
    stored = load_file()
    if (key_name and key_name in stored) or (key_env and key_env in stored):
        return "file"
    return ""


def load_file() -> dict[str, str]:
    path = paths.secrets_file()
    if not path.exists():
        return {}
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        log.warning("%s is world- or group-readable; chmod 600 it", path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def resolve(key_env: str = "", key_name: str = "") -> str:
    """Look up a key: explicit env var first, then secrets.toml.

    Keys are never written into config.toml, so a config you paste into a
    chat or a gist does not leak credentials.
    """
    if key_env:
        value = os.environ.get(key_env, "")
        if value:
            return value
    stored = load_file()
    if key_name and key_name in stored:
        return stored[key_name]
    if key_env and key_env in stored:
        return stored[key_env]
    return ""


# Shapes that are a credential wherever they appear: the vendor prefixes,
# and a long bearer-looking run. Deliberately not a general high-entropy
# hunt — that flags model ids and base64 audio and teaches people to ignore
# it. Anchored on a word boundary so it does not eat half of a URL path.
_KEYISH = re.compile(
    r"\b("
    r"sk-[A-Za-z0-9_-]{12,}"          # openai, anthropic, deepseek, many more
    r"|gsk_[A-Za-z0-9_-]{12,}"        # groq
    r"|xai-[A-Za-z0-9_-]{12,}"        # xai
    r"|AIza[A-Za-z0-9_-]{20,}"        # google
    r"|hf_[A-Za-z0-9]{12,}"           # hugging face
    r"|Bearer\s+[A-Za-z0-9._-]{16,}"
    r")"
)


def scrub(text: str, *known: str) -> str:
    """Take credentials out of something about to be stored or shown.

    An HTTP error body goes into the take's warnings, into history.jsonl and
    into a desktop notification, and some providers reflect the key back in
    the error that says it is wrong: "Incorrect API key provided: sk-...".
    Truncating the body to 160 characters does not help, because the key is
    at the front of that sentence.

    `known` is for the values this process actually holds, which is the only
    reliable half. The pattern above catches the rest, including a key from
    a provider whose error we have never seen.
    """
    for value in known:
        if value and len(value) >= 8:
            text = text.replace(value, redact(value))
    return _KEYISH.sub(lambda m: redact(m.group(0)), text)


def redact(value: str) -> str:
    if not value:
        return "(unset)"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"
