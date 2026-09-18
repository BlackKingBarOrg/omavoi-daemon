"""Configuration: typed defaults, a user TOML on top, and a dotted-path editor.

The shape mirrors what the UI presents, deliberately:

  [audio] [hotkey]     global, physical      -> Settings
  [speech]             one active engine     -> Models, left
  [llm.<name>]         several, by name      -> Models, right
  [modes.<name>]       a chain of the above  -> Modes
  [dictionary.rules]   heard -> meant        -> Dictionary / Rules
  [[dictionary.names]] canonical only        -> Dictionary / Names
"""

from __future__ import annotations

import copy
import logging
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

from . import i18n, paths

log = logging.getLogger(__name__)

# Rules every mode starts from; a mode overrides only what it cares about.
_RULE_DEFAULTS: dict[str, Any] = {
    "hallucinations": True,
    "fillers": True,
    "dictionary": True,
    "names": True,
    "cjk_spacing": True,
    "punctuation": "keep",   # keep | strip
    # What happens to newlines in the text about to be injected, whoever put
    # them there — the transcript or an LLM step. In a chat window a newline
    # is the send key, so the safe default is to fold them away. "keep" is
    # for modes that only ever type into an editor.
    "joiner": " ",
}

# Re-exported: the English and its seven translations live together in
# i18n, and everything that fills a step's prompt reads it from here.
DEFAULT_STEP_PROMPT = i18n.DEFAULT_STEP_PROMPT


def default_step_prompt(lang: str = "") -> str:
    """The starting prompt for a new LLM step, in the interface language.

    The interface language and not the speech language, because this is a
    paragraph someone reads and edits on the Modes tab. "In its original
    language", inside the prompt, is what keeps a Chinese prompt correct
    over an English take.
    """
    return i18n.t(DEFAULT_STEP_PROMPT, lang)


def is_default_step_prompt(text: str) -> bool:
    """Whether this is still a shipped prompt, in any of the languages.

    What makes it safe to move a stored prompt when the interface language
    changes: one nobody has touched follows the language, one somebody has
    edited is theirs and is never rewritten.
    """
    stripped = (text or "").strip()
    return any(stripped == default_step_prompt(lang).strip()
               for lang in ("", *i18n.LANGUAGES))

DEFAULTS: dict[str, Any] = {
    "audio": {
        "target": "",
        "rate": 16000,
        # Audio kept from *before* the key went down. This is what stops the
        # first syllable being lost while PipeWire starts the stream.
        "preroll_seconds": 0.6,
        "tail_seconds": 0.25,
        "max_seconds": 300,
        "min_seconds": 0.35,
        "warn_rms_dbfs": -45.0,
    },
    "hotkey": {
        "enabled": True,
        # Physical evdev key name, unaffected by xkb remapping. Requires
        # membership of the `input` group; `omavoi setup` explains the
        # alternative, which is a Hyprland binding on a non-modifier key.
        "key": "RIGHTALT",
        "mode": "push_to_talk",   # push_to_talk | toggle
        "devices": [],
        "rescan_seconds": 5.0,
        # Always this mode, whatever window is in front. Empty means resolve
        # it from the window as usual. `force_modifier` was here too — hold a
        # modifier to force a mode for one take — and it needs the listener to
        # track a second key's state across the whole take. Nothing read it,
        # so it is gone rather than sitting in the file looking available.
        "force_mode": "",
    },
    "speech": {
        # Exactly one speech engine is active at a time.
        # local-whispercpp | api
        #
        # whisper.cpp on Vulkan runs on NVIDIA, AMD and Intel alike, and on
        # CPU where there is no GPU, out of ~8 MB of packages. There used to
        # be a faster-whisper/CTranslate2 engine beside it, NVIDIA-only, for
        # ~2.2 GB of CUDA wheels — and on the short push-to-talk takes this
        # program is for, the two measured within a tenth of a second of each
        # other, because fixed overhead dominates at that length. It was a
        # second engine, a second model format, a second set of runtime
        # library problems and a 2.2 GB install, for nothing anyone could
        # feel.
        "backend": "local-whispercpp",
        # large-v3-turbo: half the download of large-v3 and faster, for one
        # measured cost that models.py spells out — its no_speech_prob is
        # always 0, so the silence check never fires.
        "model": "ggml:large-v3-turbo",
        "language": "",
        "local_whispercpp": {
            "binary": "",
            "port": 0,
            "threads": 0,
            "gpu": True,
            "ggml_backend_path": "",
            "beam_size": 5,
            "startup_timeout": 120.0,
        },
        "api": {
            "provider": "openai",
            "base_url": "",
            "model": "",
            "key_env": "",
            "key_name": "",
            "timeout": 30.0,
            "response_format": "verbose_json",
        },
    },
    # Any number of LLMs, referenced from a mode by these names. An entry
    # costs nothing until a mode names it.
    # Three configurations, one per kind of thing an LLM step can be, mirroring
    # the three speech engines. Named for the kind, not for a vendor or a
    # model: which weights the local one runs is a per-step choice, so two
    # modes share this configuration and each gets its own server.
    "llm": {
        # Started and owned by the daemon, lazily: a mode with no LLM step
        # costs no VRAM. Models come from the same catalogue as the speech
        # ones — `omavoi model list` shows both.
        "local": {
            "backend": "llama-local",
            # gemma-3-4b, not qwen3-8b: the shipped `prose` mode has a local
            # LLM step, so this is the model a fresh install actually runs —
            # and qwen is the strongest Chinese at its size and weaker
            # elsewhere, which is the wrong thing to assume about someone who
            # has just installed a dictation tool. Half the size, too.
            "model": "llm:gemma-3-4b",
            "n_gpu_layers": 99,
            "ctx_size": 4096,
            "threads": 0,
            "port": 0,
            "startup_timeout": 180.0,
            "timeout": 60.0,
            "max_tokens": 1024,
            "temperature": 0.2,
            # Reasoning models answer an editing prompt by thinking at length
            # and then running out of budget. Nothing here needs deliberation.
            "thinking": False,
        },
        # Whichever coding agent Omarchy is set to. It is already logged in, so
        # this needs no key; it costs several seconds of process startup, which
        # is why it suits a deliberate pass rather than every take.
        "agent": {
            "backend": "agent-cli",
            "agent": "",
            "model": "",
            "timeout": 60.0,
            "max_tokens": 1024,
            "temperature": 0.2,
        },
        # One remote endpoint. `backend` picks the wire format, because these
        # are not all the same protocol: "openai" covers OpenAI, Groq,
        # SiliconFlow, vLLM and most others, "anthropic" is Claude's own.
        # Unconfigured on purpose — it is the one kind that sends your words
        # off the machine, so it takes a deliberate URL and key.
        "api": {
            "backend": "openai",
            "model": "",
            "base_url": "",
            "key_env": "OPENAI_API_KEY",
            "key_name": "openai",
            "timeout": 20.0,
            "max_tokens": 1024,
            "temperature": 0.2,
        },
    },
    # How a mode gets picked. Separate from [modes.*], which only defines them.
    "switching": {
        # Choose the mode from the focused window. Off by default: it is a good
        # idea that needs per-application tuning before it earns its keep, and
        # until then a mode that changes under you is worse than one that does
        # not. The match lists stay where they are — turning this on is one
        # flag, not a rebuild.
        "by_window": False,
        # Which mode every take uses while by_window is off.
        "mode": "default",
    },
    "modes": {
        # The fallback. Every other mode inherits anything it omits.
        "default": {
            "match": [],
            "language": "",
            "prompt": "",
            "inject": "auto",
            "rules": dict(_RULE_DEFAULTS),
            "steps": [],
        },
        "terminal": {
            "match": ["alacritty", "foot", "kitty", "ghostty", "org.wezfurlong.wezterm"],
            "inject": "auto",
            "paste_key": "CTRL+SHIFT+V",
            # A command line does not want a trailing full stop, and it cannot
            # afford an LLM round-trip either.
            "rules": dict(_RULE_DEFAULTS) | {"punctuation": "strip"},
            "steps": [],
        },
        "code": {
            "match": ["code", "cursor", "dev.zed.Zed"],
            "inject": "clipboard",
            "rules": dict(_RULE_DEFAULTS),
            "steps": [],
        },
        "prose": {
            "match": ["obsidian", "slack", "com.anthropic.claude", "thunderbird"],
            "inject": "auto",
            "rules": dict(_RULE_DEFAULTS),
            # Zero or more LLM passes, run in order. Each names an [llm.*].
            "steps": [
                {
                    "llm": "local",
                    "prompt": DEFAULT_STEP_PROMPT,
                },
            ],
        },
    },
    "dictionary": {
        # heard -> meant. Unlike a decoder prompt this is a guarantee, not a
        # hint. Case-insensitive; the longest key is tried first.
        "rules": {
            "hyperland": "Hyprland",
            "hyper land": "Hyprland",
            "wayland": "Wayland",
            "omarchy": "Omarchy",
            "github": "GitHub",
            "gitlab": "GitLab",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "kubernetes": "Kubernetes",
            "postgres": "Postgres",
        },
        # Proper nouns, written only in their correct form. You cannot know
        # how a model will mangle a name, and for CJK the manglings are an
        # open set of homophones, so these are matched by sound instead.
        "names": [],
        "names_settings": {
            # Seed the most-used names into the decoder prompt, which is what
            # makes the model produce them rather than fixing them after.
            "seed_prompt": True,
            "seed_budget_tokens": 224,
            # Sound matching can damage text that was already correct, so a
            # new name stays inert until its dry run has been accepted.
            "match_new_names": False,
            "pinyin_require_tones": False,
            "min_chars": 2,
        },
    },
    "post": {
        "enabled": True,
        # Reject the whole transcript when the model itself says it heard
        # nothing. Its own verdict beats any string matching.
        # Whisper segments text the way subtitles are cut, one line each. Left
        # alone, a sentence arrives as several lines — and in a chat window a
        # newline can send the message. So boundaries become punctuation.
        "newlines": "space",              # space | keep
        "add_missing_punctuation": True,
        "no_speech_threshold": 0.8,
        # A quiet take *and* a raised no_speech is silence almost every time,
        # while either signal alone is too weak to reject on: a strict
        # threshold lets "Thank you." through, and a loose one eats real
        # softly-spoken words. So this lower bound applies only below
        # audio.warn_rms_dbfs.
        "quiet_no_speech_threshold": 0.5,
        # Fillers, split by how the text is written rather than by language:
        # a space-separated script can be matched on a word boundary, and one
        # written without spaces has to be bracketed by punctuation instead.
        # The console offers eight interface languages and this had lists for
        # two of them, so six were dictating into a rule that did nothing.
        #
        # Every entry here is a token that is *not* a word in its language.
        # The spaced list is removed wherever it appears as a word, so a real
        # word on it would be deleted out of the middle of a sentence — which
        # is why German has "äh" and not "also", and Spanish "em" and not
        # "pues". The unspaced list is safer, because a filler there only
        # goes when a sentence boundary or a comma brackets it: that is what
        # lets 那个 be a filler in 那个，我想说 and a demonstrative in 那个函数.
        "fillers_en": ["um", "uh", "erm", "hmm", "er"],
        "fillers_de": ["äh", "ähm", "öh", "öhm", "hm"],
        "fillers_fr": ["euh", "heu", "hum"],
        "fillers_es": ["eh", "em", "ehm"],
        "fillers_cjk": ["嗯", "呃", "啊", "唉", "那个", "这个", "就是说"],
        "fillers_ja": ["えーと", "えっと", "ええと", "あのー", "えー", "うーん"],
        # Left empty on purpose. Thai and Vietnamese fillers are mostly
        # overloaded particles — Thai แบบ and คือ, Vietnamese à and ừ are
        # ordinary words as often as they are hesitation — and a wrong entry
        # on a filler list deletes a real word silently. The key exists so
        # someone who speaks the language can fill it; guessing it here would
        # be worse than leaving it off.
        "fillers_th": [],
        "fillers_vi": [],
        # What whisper writes when it hears nothing: the subtitle credits and
        # sign-offs its training data is full of. Matched per sentence and
        # whole, after folding away punctuation, spacing and case — so a
        # 谢谢观看 said inside a longer sentence survives and a bare one goes.
        #
        # This covers the languages the console offers, which it did not:
        # seven English entries, five Chinese, one Japanese and one Russian,
        # for an interface that speaks eight. The cost of a wrong entry here
        # is only that it never fires — a subtitle credit is not a sentence
        # anyone dictates — which is the opposite of the filler lists above,
        # where a wrong entry deletes a real word. So imperfect recall of an
        # exact credit string is worth having; a guessed filler is not.
        "hallucinations": [
            # en
            "Thanks for watching!",
            "Thank you for watching!",
            "Thank you.",
            "Thank you",
            "you",
            "Bye.",
            "Subtitles by the Amara.org community",
            # zh
            "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
            "字幕由Amara.org社区提供",
            "由 Amara.org 社群提供的字幕",
            "谢谢观看",
            "感谢观看",
            "字幕志愿者 李宗盛",
            "谢谢大家",
            # ja
            "ご視聴ありがとうございました",
            "ご視聴ありがとうございます",
            "最後までご視聴いただきありがとうございました",
            # de
            "Untertitel der Amara.org-Community",
            "Untertitelung aufgrund der Amara.org-Community",
            "Vielen Dank für das Zuschauen",
            "Danke fürs Zuschauen!",
            # fr
            "Sous-titres réalisés par la communauté d'Amara.org",
            "Merci d'avoir regardé cette vidéo !",
            "Sous-titrage Société Radio-Canada",
            # es
            "Subtítulos realizados por la comunidad de Amara.org",
            "Gracias por ver el video",
            "¡Gracias por ver el vídeo!",
            # vi
            "Phụ đề được thực hiện bởi cộng đồng Amara.org",
            "Hãy subscribe cho kênh Ghiền Mì Gõ để không bỏ lỡ những video hấp dẫn",
            # th
            "คำบรรยายโดยชุมชน Amara.org",
            # ru, which was already here
            "Продолжение следует...",
            "Субтитры сделал DimaTorzok",
        ],
    },
    "inject": {
        # auto | wtype | clipboard
        "method": "auto",
        # XWayland clients never receive the keymap wtype installs for its
        # virtual keyboard, so they decode its keycodes against the system
        # layout and a sentence arrives as "1234567890-=". Detecting the
        # client beats listing them: it covers every X11 app at once.
        # XTEST is the only route into an X11 client that depends on neither
        # wtype's keymap nor the compositor's clipboard bridge. Where that
        # bridge is broken — and it is, on some setups — pasting silently
        # produces nothing at all.
        #
        # `xdotool_for_xwayland` and `avoid_wtype_on_xwayland` used to be here
        # as switches for that, both defaulting on and neither read: the first
        # by nothing at all, the second by an attribute assigned in Injector
        # and never consulted. Turning either off selects a route these very
        # comments say cannot arrive — wtype's keymap is what X11 clients
        # cannot read, and the clipboard bridge does not carry the selection —
        # so they were switches for producing silence. The xdotool-absent case
        # is still handled, by asking shutil.which.
        "xdotool_delay_ms": 12,
        "clipboard_classes": [
            "code", "cursor", "electron", "slack", "discord", "obsidian",
            "chrome", "chromium", "brave", "vivaldi", "com.anthropic.claude",
            # Belt and braces for compositors that do not report xwayland.
            "wechat", "weixin", "feishu", "lark", "qq", "dingtalk",
        ],
        "paste_key": "CTRL+V",
        # What a newline in the text becomes when it is typed rather than
        # pasted. In an editor a newline is Return. In a chat window Return
        # is the send key, so a two-line take posted its first line and left
        # the second in the box -- and the earlier answer, folding every
        # newline away after the LLM, made two lines impossible. Modes that
        # type into chat windows set SHIFT+RETURN; a paste keeps newlines as
        # they are and never consults this.
        "newline_key": "RETURN",
        # Windows where Return sends the message, so a typed newline goes as
        # SHIFT+RETURN whatever newline_key says -- a mode's own newline_key
        # still wins. Matched against the window class, case-insensitively,
        # as a substring, like clipboard_classes.
        "chat_classes": [
            "wechat", "weixin", "feishu", "lark", "qq", "dingtalk",
            "slack", "discord", "telegram", "element", "signal",
        ],
        # How the paste keystroke is delivered:
        #   shortcut  the compositor synthesises it (hyprctl send_shortcut)
        #   wtype     the virtual keyboard sends it — wrong keys on X11
        #   xdotool   XTEST, the only one an X11 client reads correctly
        # Empty follows the window: xdotool for X11, shortcut for Wayland.
        "paste_method": "",
        # XWayland mirrors the Wayland selection lazily, so an X11 client can
        # ask for the clipboard noticeably after the keystroke arrives.
        "paste_settle_ms": 150,
        "restore_clipboard_after": 4.0,
        "wtype_delay_ms": 0,
    },
    "history": {
        "enabled": True,
        "keep": 500,
        # Stored audio is what makes Test, re-running on another model, and
        # the Names dry run possible. 0 turns those off with it.
        "keep_audio": 20,
    },
    "ui": {
        "notify": True,
        "notify_on_empty": True,
        "log_level": "INFO",
        # "" follows the environment. Otherwise one of: en, de, es, fr, vi,
        # zh, ja, th — the console's dropdown writes this.
        "language": "",
        "hud": True,
        # bottom is the only one implemented. "cursor" and "window" were
        # written here and never built: they need the pointer position or the
        # focused window's geometry at the moment the key goes down, which is
        # a hyprctl call per take and a multi-monitor question, and a config
        # key that names a behaviour nothing implements is worse than no key.
        # config set rejects the other two rather than accepting them quietly.
        "hud_position": "bottom",     # bottom
        "hud_size": "s",              # xs | s | m
        # always | changed | never — "changed" dwells only when the text was
        # altered or flagged, which is the only version that stays useful
        # when you dictate several sentences in a row.
        "hud_dwell": "changed",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def defaults() -> dict[str, Any]:
    return copy.deepcopy(DEFAULTS)


def load(path: Path | None = None) -> dict[str, Any]:
    path = path or paths.config_file()
    if not path.exists():
        return defaults()
    try:
        with path.open("rb") as fh:
            user = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"Bad TOML in {path}:\n  {exc}") from exc
    merged = _deep_merge(defaults(), user)
    # A user-defined mode should not have to restate every rule.
    for name, mode in merged.get("modes", {}).items():
        if isinstance(mode, dict):
            mode["rules"] = _RULE_DEFAULTS | dict(mode.get("rules") or {})
    # Entries from before there were three of them. Reported, not silent: a
    # fold rewrites the steps that named them.
    folded = (migrate(merged) + retire_cuda_engine(merged)
              + normalise_hotkey(merged) + follow_ui_language(merged))
    for note in folded:
        log.info("config: %s", note)
    if folded and path.exists():
        # Written back, or the fold happens again on every load and the file
        # keeps disagreeing with what is running.
        try:
            write(merged, path)
        except OSError as exc:
            log.warning("could not write the folded config back: %s", exc)
    for problem in validate(merged):
        log.warning("config: %s", problem)
    return merged


def _speech_spec(key: str) -> Any:
    """The catalogue entry for a speech model key, or None."""
    from . import models

    spec = models.spec(key)
    return spec if spec is not None and spec.kind == models.SPEECH else None


# The three configurations an LLM step can name. Anything else in [llm.*] is
# from before they existed and gets folded into whichever of these matches its
# backend, with the modes that named it rewritten to point at the fold.
_LLM_KINDS = {
    "local": ("llama-local", "llama.cpp", "llamacpp"),
    "agent": ("agent-cli", "agent", "claude-cli", "claude-code", "claude"),
    "api": ("openai", "openai-compatible", "ollama", "vllm", "llama-cpp", "anthropic"),
}


def _kind_of(backend: str) -> str:
    b = str(backend).strip().lower()
    for kind, names in _LLM_KINDS.items():
        if b in names:
            return kind
    return ""


def retire_cuda_engine(cfg: dict[str, Any]) -> list[str]:
    """Move a config off the faster-whisper engine, which no longer exists.

    Without this an upgraded install has `speech.backend = "local-whisper"`,
    asr.build raises on an unknown backend, and the daemon fails on every
    start — the loudest possible way to deliver a removal, and to someone who
    did nothing but update.

    The model key moves with it. A ct2 model was named bare — `large-v3` —
    and the ggml catalogue has the same names, so the equivalent is the one
    with the prefix. Where there is no equivalent the shipped default is
    used, because a key that names nothing stops the daemon just as dead.

    Reported, like the [llm.*] fold: the config file changed underneath.
    """
    from . import models

    notes: list[str] = []
    speech = cfg.get("speech") or {}
    if str(speech.get("backend", "")) in ("local-whisper", "faster-whisper", "ct2"):
        speech["backend"] = "local-whispercpp"
        notes.append("speech.backend was the removed CUDA engine, "
                     "now local-whispercpp (whisper.cpp on Vulkan)")
    # Dropped whether or not the backend was in use: it configures nothing.
    if speech.pop("local_whisper", None) is not None:
        notes.append("[speech.local_whisper] configured the removed engine and is gone")

    want = str(speech.get("model", "") or "")
    if want and ":" not in want:
        # A bare name was the ct2 spelling. It still resolves — parse_key
        # reads one as ggml now — but leaving it bare means `config show`
        # names the model in a form that no longer has a meaning of its own.
        # Six of the eight ct2 names exist verbatim in the ggml catalogue.
        # These two do not, and falling through to the shipped default would
        # hand someone who had deliberately chosen 75 MB a 3 GB download.
        nearest = {"tiny": "base", "distil-large-v3": "large-v3-turbo"}
        moved = f"{models.GGML}:{nearest.get(want, want)}"
        if models.spec(moved) is not None:
            speech["model"] = moved
            notes.append(f"speech.model {want!r} was a ct2 model, now {moved!r}")
        else:
            speech["model"] = DEFAULTS["speech"]["model"]
            notes.append(f"speech.model {want!r} has no whisper.cpp equivalent, "
                         f"now {speech['model']!r}")
    elif want and models.spec(want) is None:
        speech["model"] = DEFAULTS["speech"]["model"]
        notes.append(f"speech.model {want!r} names nothing in the catalogue, "
                     f"now {speech['model']!r}")
    return notes


def normalise_hotkey(cfg: dict[str, Any]) -> list[str]:
    """Write the hotkey the way everything else spells it.

    A combination can be typed in any order and any case — `super+v`,
    `shift+ctrl+space` — and the listener canonicalises it internally, so
    both work. But then `config get` shows what you typed while `hotkey
    check` and the daemon both report CTRL+SHIFT+SPACE, and the one
    comparison that matters — has the daemon picked up the change — is
    between two spellings of the same thing.

    A name that resolves to nothing is uppercased like any other and left
    alone otherwise; validate() is what reports that it is not a key.
    """
    # canonical_name and not parse_chord: this runs on every config.load, and
    # parse_chord resolves the names to evdev codes, which costs 25 ms of
    # importing evdev on every command — the exact regression the startup
    # test exists to catch, and did. The spelling needs no codes: it is the
    # modifier order and the case.
    from .hotkey import canonical_name

    want = str((cfg.get("hotkey") or {}).get("key", "") or "")
    if not want:
        return []
    canonical = canonical_name(want.split("+"))
    if canonical == want:
        return []
    cfg["hotkey"]["key"] = canonical
    return [f"hotkey.key {want!r} written as {canonical!r}"]


def follow_ui_language(cfg: dict[str, Any]) -> list[str]:
    """Move step prompts that are still shipped defaults into ui.language.

    The prompt a step starts with is filled in when the step is made, so a
    console switched to Chinese afterwards kept every English prompt it had
    already written — which is the whole of what someone means by asking for
    the prompts to be translated too.

    Only a prompt that still matches a shipped default, in any language, is
    moved. One with a single character changed is the user's, and the cost of
    guessing wrong here is somebody's own instructions silently replaced.

    Reported rather than done quietly, and in place, like the fold above.
    """
    lang = i18n.ui_lang(cfg)
    want = default_step_prompt(lang)
    notes: list[str] = []
    for name, mode in (cfg.get("modes") or {}).items():
        for index, step in enumerate(mode.get("steps") or []):
            if not isinstance(step, dict):
                continue
            current = str(step.get("prompt", "") or "")
            if current == want or not is_default_step_prompt(current):
                continue
            step["prompt"] = want
            notes.append(
                f"modes.{name}.steps[{index}].prompt was the default prompt, "
                f"moved to {lang or 'en'}"
            )
    return notes


def migrate(cfg: dict[str, Any]) -> list[str]:
    """Fold stray [llm.*] entries into the three, in place.

    An open-ended list of named entries put an implementation detail on screen
    as a configuration surface: six rows, two of them called haiku for
    different reasons. Folding is not lossless when someone had two remote
    endpoints, so every fold is reported rather than done quietly.
    """
    notes: list[str] = []
    entries: dict[str, Any] = cfg.setdefault("llm", {})
    modes: dict[str, Any] = cfg.get("modes") or {}

    for name in [n for n in list(entries) if n not in _LLM_KINDS]:
        entry = dict(entries[name])
        kind = _kind_of(entry.get("backend", ""))
        if not kind:
            continue
        model = str(entry.get("model", "") or "")
        target = entries.setdefault(kind, {})

        if kind == "local":
            # The weights move to the steps that wanted them, which is what a
            # second local entry was standing in for.
            for mode_name, mode in modes.items():
                for step in mode.get("steps") or []:
                    if str(step.get("llm", "")) == name:
                        step["llm"] = kind
                        if model and model != str(target.get("model", "")):
                            step["model"] = model
            notes.append(f"llm.{name} folded into llm.local; "
                         f"the steps that used it now name their own weights")
        else:
            # For the other two there is one endpoint and one agent, so the
            # first stray entry configures it and later ones only re-point.
            if not str(target.get("model", "")) and model:
                for field in ("backend", "model", "base_url", "key_env", "key_name"):
                    if entry.get(field):
                        target[field] = entry[field]
                notes.append(f"llm.{name} became llm.{kind}")
            else:
                notes.append(f"llm.{name} dropped; llm.{kind} was already configured")
            for mode_name, mode in modes.items():
                for step in mode.get("steps") or []:
                    if str(step.get("llm", "")) == name:
                        step["llm"] = kind
        del entries[name]

    return notes


def validate(cfg: dict[str, Any]) -> list[str]:
    """Non-fatal complaints, surfaced by `omavoi doctor`."""
    from . import asr

    problems: list[str] = []

    if asr.canonical(cfg["speech"]["backend"]) is None:
        problems.append(f"speech.backend={cfg['speech']['backend']!r} is not a known engine")
    if cfg["hotkey"]["mode"] not in ("push_to_talk", "toggle"):
        problems.append(f"hotkey.mode={cfg['hotkey']['mode']!r} must be push_to_talk or toggle")
    # An unresolvable key name was accepted silently: the daemon logged
    # "hotkey unavailable" once at startup and then ran with no hotkey at all,
    # which looks exactly like a broken microphone.
    key = str(cfg["hotkey"].get("key", ""))
    # Checked only where evdev is already loaded, which is not a shortcut but
    # the whole of the answer to "who reads this warning".
    #
    # Every command calls config.load, which calls this, and `from evdev
    # import ecodes` costs 25 ms — there is no cheap submodule, importing any
    # part of it runs the package. So `config show`, `dict list` and `names
    # list` each paid 25 ms to tell someone their keyboard shortcut was
    # misspelled, which is not a thing any of them is doing.
    #
    # Everything that could act on it has evdev loaded already: the daemon
    # imports it to bind the key, `hotkey check` and `doctor` import it to
    # diagnose exactly this, and `config set hotkey.key` refuses a bad name
    # outright, so a wrong one can now only arrive by hand-editing the file.
    if key and "evdev" in sys.modules:
        from .hotkey import HotkeyUnavailable, parse_chord

        try:
            parse_chord(key)
        except HotkeyUnavailable as exc:
            problems.append(f"hotkey.key={key!r} is not an evdev key ({exc})")
    if cfg["audio"]["rate"] != 16000:
        problems.append("whisper needs 16000 Hz; audio.rate has been changed")
    if cfg["audio"]["preroll_seconds"] < 0:
        problems.append("audio.preroll_seconds cannot be negative")

    modes = cfg.get("modes", {})
    for name, mode in modes.items():
        want = str((mode or {}).get("speech_model", "") or "")
        if not want:
            continue
        if _speech_spec(want) is None:
            problems.append(
                f"modes.{name}.speech_model={want!r} is not a speech model in the catalogue"
            )
    if "default" not in modes:
        problems.append("no [modes.default]; there must be a fallback mode")
    for name, mode in modes.items():
        for index, step in enumerate(mode.get("steps") or []):
            ref = step.get("llm")
            if ref not in cfg.get("llm", {}):
                problems.append(
                    f"modes.{name}.steps[{index}] names llm={ref!r}, which is not defined"
                )
    force = cfg["hotkey"].get("force_mode")
    if force and force not in modes:
        problems.append(f"hotkey.force_mode={force!r} is not a mode")
    fixed = cfg.get("switching", {}).get("mode", "default")
    if fixed not in modes:
        problems.append(f"switching.mode={fixed!r} is not a mode")
    return problems


# -- dotted-path access, for `omavoi config get/set` ------------------------

def get_path(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def coerce(current: Any, raw: str) -> Any:
    """Type the new value like the one it replaces, so the TOML stays valid."""
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in ("true", "yes", "on", "1"):
            return True
        if low in ("false", "no", "off", "0"):
            return False
        raise ValueError(f"{raw!r} is not a boolean")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key: str) -> str:
    """TOML bare keys are ASCII-only, so anything else has to be quoted.

    Two real cases: dictionary entries in a non-Latin script, and profile
    names like org.wezfurlong.wezterm, which would otherwise be read as
    three nested tables.
    """
    if _BARE_KEY.match(key):
        return key
    return '"' + key.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_toml_key(k)} = {_toml_scalar(v)}" for k, v in value.items()) + "}"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{text}"'


def _is_table_array(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)


def dumps(cfg: dict[str, Any]) -> str:
    """Serialise a config back to TOML.

    Scalars first, then sub-tables, then arrays-of-tables — otherwise a
    scalar written after a [table] header would be swallowed by it.
    """
    lines: list[str] = []

    def emit(node: dict[str, Any], prefix: str) -> None:
        scalars = {k: v for k, v in node.items()
                   if not isinstance(v, dict) and not _is_table_array(v)}
        tables = {k: v for k, v in node.items() if isinstance(v, dict)}
        arrays = {k: v for k, v in node.items() if _is_table_array(v)}

        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalars.items():
            lines.append(f"{_toml_key(key)} = {_toml_scalar(value)}")
        if scalars or prefix:
            lines.append("")
        for key, value in tables.items():
            emit(value, f"{prefix}.{_toml_key(key)}" if prefix else _toml_key(key))
        for key, value in arrays.items():
            path = f"{prefix}.{_toml_key(key)}" if prefix else _toml_key(key)
            for item in value:
                lines.append(f"[[{path}]]")
                for k, v in item.items():
                    lines.append(f"{_toml_key(k)} = {_toml_scalar(v)}")
                lines.append("")

    emit(cfg, "")
    return "\n".join(lines).rstrip() + "\n"


def write(cfg: dict[str, Any], path: Path | None = None) -> Path:
    path = path or paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # The pid is in the name because the console starts seven `omavoi`
    # processes at once and any of them can rewrite the config — a fold, or a
    # prompt following the interface language. They all compute the same
    # bytes, so sharing one temp file has never actually corrupted anything,
    # but one writer truncating another's half-written temp and then renaming
    # it is not a thing to leave standing on the strength of that.
    tmp = path.with_suffix(f".toml.{os.getpid()}.tmp")
    try:
        tmp.write_text(dumps(cfg), encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def set_path(dotted: str, raw: str, path: Path | None = None) -> Any:
    """Edit one key on disk. Rewrites the file from the merged config."""
    path = path or paths.config_file()
    cfg = load(path)
    current = get_path(cfg, dotted)  # raises KeyError on a typo
    value = coerce(current, raw)

    node: Any = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    write(cfg, path)
    return value
