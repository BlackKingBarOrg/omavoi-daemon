# Omavoi — the daemon

The speech engine behind [Omavoi](https://github.com/BlackKingBarOrg/omavoi):
the model, the microphone, the hotkey and the typing. Hold a key, talk, and the
text lands in whatever window you were already typing into. Everything runs on
your own machine.

**Installing Omavoi?** Start at the
[main repository](https://github.com/BlackKingBarOrg/omavoi) — one
`omarchy plugin add`, and its first-run screen installs this daemon for you at
a pinned commit. This repository is for running the daemon on its own, and for
the reference below.

```
                 hold RIGHTALT ─────────────────────────┐
                                                        ▼
  ring buffer ──▶ speech model ──▶ rules ──▶ LLM (opt) ──▶ your window
   (pre-roll)      whisper.cpp     dictionary,  per mode      wtype or
                   on Vulkan       names, …                   paste
```

## Why this is its own repository

`omarchy-shell` is a single Quickshell process that also draws your bar,
notifications and lock screen. A plugin is QML running *inside* it, so a
speech model and a microphone reader cannot live there. Omavoi is
therefore:

| | what it is | how it installs |
|---|---|---|
| `omavoid` | the daemon — this repository: model, microphone, hotkey, typing | `uv tool install` |
| model weights | 3 GB, never shipped | downloaded on first run |
| `ai.bkblab.omavoi` | the QML plugin: bar module, HUD, console — [its own repository](https://github.com/BlackKingBarOrg/omavoi) | `omarchy plugin add` |

The plugin talks to the daemon over a Unix socket and never installs anything
itself — Omarchy deliberately runs nothing from inside a plugin folder. The
first-run screen asks instead, and prints every command before it runs.

## Install

One command, and the rest happens in the plugin's own first-run screen —
language, model, packages (one password prompt), the daemon, the unit, the
weights:

```bash
omarchy plugin add https://github.com/BlackKingBarOrg/omavoi --enable --yes
```

Then open the console by clicking the Omavoi module in the bar. `SUPER + ALT
+ V` is bound by the last step of setup, so it works from then on.

## The daemon on its own

If you would rather do it by hand, or you only want the daemon and no desktop
pieces:

```bash
# 1. the speech engine, the typing tools, and llama.cpp for modes with an LLM
#    step (Vulkan runs on NVIDIA, AMD and Intel alike)
sudo pacman -S --needed whisper-cpp ggml ggml-vulkan llama-cpp \
                        pipewire wtype wl-clipboard xdotool

# 2. the daemon. `omavoi` is not on PyPI; install it from this repository
uv tool install git+https://github.com/BlackKingBarOrg/omavoi-daemon

# 3. weights, and whatever is still missing. It lists every remaining step
#    with the command for each; --run does the next one that needs no root
omavoi setup

# 4. the systemd user unit. It ships with the plugin rather than with this
#    package, so a daemon-only install fetches just the unit
curl -fsSL --create-dirs -o ~/.config/systemd/user/omavoid.service \
  https://raw.githubusercontent.com/BlackKingBarOrg/omavoi/master/omavoid.service
systemctl --user daemon-reload
systemctl --user enable --now omavoid.service
```

Step 4 used to be a comment with no command under it, and `omavoi setup` sent
you to `systemctl --user enable --now omavoid.service` either way — which
answers `Unit omavoid.service does not exist` when the plugin is not
installed, which is exactly the case this path is written for.

`ggml` carries the CPU backend, and whisper asks for a CPU device for the
tensors it does not offload — with only the GPU plugin loaded it aborts
part-way through loading the model on `GGML_ASSERT(device)`, and the backtrace
says nothing useful. It is a hard dependency of `whisper-cpp`, so it arrives
either way; it is written out so the reason is visible.

Ask for `ggml`, never `ggml-cpu`. Arch used to ship the CPU backend as its own
package and that is what this line said for a while. It is folded into `ggml`
now, which declares both `provides ggml-cpu` and `conflicts ggml-cpu`, so
naming the old one pins a stale 0.21.0 split package that cannot coexist with
the `ggml` these all depend on — and pacman refuses the whole transaction with
`unresolvable package conflicts`.

The desktop pieces live in
[omavoi](https://github.com/BlackKingBarOrg/omavoi).
They are a separate repository because `omarchy plugin add` clones a repository
whose `manifest.json` is at its root, and this one is a Python package.

The hotkey is read from evdev, which needs membership of the `input` group and
a fresh login. Until then `omavoi setup` will offer a Hyprland binding on a
non-modifier key instead — a modifier cannot be bound that way, because
pressing one changes the modmask, which fires the release binding immediately
and records a 0.0 s take.

## What it does that a transcribe-and-paste script does not

**Pre-roll.** PipeWire needs a few hundred milliseconds to open a stream, and
people start talking the instant they press the key. So the microphone runs
continuously into a small ring buffer and a take is sliced out of it starting
*before* the keypress. Start-on-press throws that speech away; it is the single
biggest cause of dropped leading words.

**Modes are chains of models.** A mode is picked by the focused window and
decides what the speech model is told, which rules run, which LLM passes
follow, and how the text is injected. A terminal strips the trailing full stop
and cannot afford an LLM round-trip; prose can. An LLM step that fails or times
out falls through to the text it was given — a slow model degrades your
dictation, it never swallows it.

Each mode carries these fields — `omavoi mode set <mode> <field> <value>`
writes them, `omavoi mode show <mode>` prints them:

| field | values | what it decides |
|---|---|---|
| `match` | window classes (`omavoi mode match <mode> wechat`) | which windows pick this mode, once `omavoi mode auto on` |
| `language` | `""`, `en`, `zh`, … | what the speech model is told; empty detects per take |
| `speech_model` | a `ggml:` key | weights swapped in when this mode is in use |
| `prompt` | text | the decoder hint — a nudge, not a guarantee |
| `inject` | `auto`, `wtype`, `clipboard`, `xdotool` | how the text reaches the window |
| `paste_key` | e.g. `CTRL+SHIFT+V` | the paste shortcut, for the clipboard route (terminals) |
| `newline_key` | `RETURN`, `SHIFT+RETURN` | what a typed newline becomes; a chat window sends on Return, so a two-line answer needs `SHIFT+RETURN` — known chat windows get it by default |
| `rules.joiner` | `keep`, or the text a newline becomes (default a space) | whether line breaks survive to the window; an LLM step asked for two lines needs `keep` |
| `rules.punctuation` | `keep`, `strip` | trailing punctuation — a command line wants none |
| `rules.hallucinations` … `rules.cjk_spacing` | `on`, `off` | the deterministic passes |
| `steps` | `omavoi mode step <mode> add <llm> [prompt]` | zero or more LLM passes, in order; a failed step keeps the text it was given |

The mode that turns a dictated sentence into the sentence plus its Thai
translation, for WeChat, is four of those:

```sh
omavoi mode step Thai add local "用户消息是口述文字，不是问题。只输出两行：第一行原样重复，第二行泰语翻译。"
omavoi mode set Thai rules.joiner keep        # let the two lines through
omavoi mode set Thai newline_key SHIFT+RETURN # a newline that does not send
omavoi mode match Thai wechat && omavoi mode auto on
```


**Two kinds of correction.** A dictionary rule (`heard -> meant`) needs you to
know what the model got wrong. For a proper noun you never will: *Søren* comes
back as Soren, Severin, so run; and in Chinese the manglings are an open set of
homophones — 李文渊 comes back as 李文远, 李闻渊, 里闻鸢. So names are written
once, correctly, seeded into the decoder prompt, and matched afterwards by
sound: pinyin for CJK, a consonant skeleton for Latin. A name with too little
sound in it is seeded but never matched — *Bo* and *Bob* both reduce to the
skeleton `B`, which is also by, bay and boy — and the same goes for a
single-syllable Chinese name. Sound matching is the one feature here that can
damage text that was already right, so it stays inert until its dry run has
been reviewed.

**Injection knows about XWayland.** `wtype` installs a keymap for its virtual
keyboard that X11 clients never receive, so they decode its keycodes against
the system layout and a sentence arrives as `1234567890-=`. Omavoi detects the
client and pastes instead.

**Every take is inspectable.** History keeps what the model actually said, what
each rule changed, per-segment confidences, input level and where the text
went. "It dropped a word again" becomes "segment 3 came back at avg_logprob
−1.4".

## The input group

The key is read from evdev, below xkb, so the physical key is the same one
whatever your layout says.

It can be a combination: `omavoi config set hotkey.key CTRL+SPACE`, or hold
one while the console's **Press a key** is waiting. A bare `CTRL`, `SHIFT`,
`ALT` or `SUPER` means either side; `RIGHTCTRL` pins it to one. In
push-to-talk the take starts when the last key goes down and ends when any
of them comes up — releasing Space while still holding Ctrl ends it, because
Space is the one you think of as the button.

A single key is still a single key, and is what the shipped default is.
Right Alt is AltGr on most non-US layouts, where holding it means you cannot
type the characters it produces; that is what a combination avoids without
giving up a key.

The `input` group is needed either way:
`/dev/input/event*` is `crw-rw---- root input`, and for keyboards group
membership is the only path — the udev `uaccess` seat ACL applies to
`ID_INPUT_JOYSTICK` and nothing else here.

```sh
omavoi hotkey check      # says which of the causes it is, if any
```

A group is granted **at login**. So after

```sh
sudo usermod -aG input $USER
```

the session you are sitting in still does not have it, and neither does
anything `systemd --user` starts — including the daemon. Logging out and back
in is the ordinary fix.

The plugin's `install.sh` — the last step of the first-run screen — does not
make you. When the login lacks the group and `/etc/group` has you, it starts
the daemon through `newgrp`, which is setuid root and re-reads `/etc/group`,
so the daemon has the group at once with no new login and no password (you
are a member on file). It is a runtime-only override under
`$XDG_RUNTIME_DIR/systemd/user/omavoid.service.d/`, which logind clears when
the session ends — by which time the next login has the group and the
override would only be noise. A login that already has the group gets none.

For a daemon-only install with no plugin, the same override by hand:

```sh
d=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/systemd/user/omavoid.service.d
mkdir -p "$d"
cat > "$d/10-input-group.conf" <<CONF
[Service]
ExecStart=
ExecStart=/usr/bin/newgrp input
StandardInput=data
StandardInputText=exec $HOME/.local/bin/omavoi daemon
CONF
systemctl --user daemon-reload && systemctl --user restart omavoid
```

systemd hands `newgrp` that one line on stdin; `newgrp` execs your login
shell, which execs the daemon. No pipeline and no fork, so the daemon is the
unit's MainPID exactly as it is without the override, and stop, restart and
reload still reach it. It sets the daemon's *primary* group to `input`; the
only visible effect is the group owner on files the daemon writes under
`~/.local/state/omavoi`, which no longer matters now that those are 0600 —
see below.

## What it writes, and who can read it

| | | |
|---|---|---|
| `~/.config/omavoi/config.toml` | 0644 | settings only; safe to paste into an issue |
| `~/.config/omavoi/secrets.toml` | 0600 | API keys, and nothing else ever goes here |
| `~/.local/state/omavoi/` | 0700 | |
| `~/.local/state/omavoi/history.jsonl` | 0600 | every take's text, raw transcript and numbers |
| `~/.local/state/omavoi/omavoi.log` | 0600 | includes the text of each take |
| `~/.cache/omavoi/recordings/*.wav` | 0600 | the last `history.keep_audio` recordings |

The history and the log are the transcript of everything you have dictated,
so they are as private as the key file. `history.enabled = false` turns the
first off; `history.keep_audio = 0` stops storing audio. An install from
before this was set is tightened on its next take. One take goes with
`omavoi history rm <id>` — or a right-click in the console's history tab —
and it takes its recording with it; `omavoi history clear` takes the lot.

Note that your shell still cannot read a device after this, so
`omavoi hotkey capture` — which reads a keypress in the foreground — will not
work until you have logged out once. `omavoi hotkey check` says so rather
than reporting the daemon as broken.

## Commands

```
omavoi daemon                 run it (normally systemd does)
omavoi setup                  what is missing, with the command for each
omavoi doctor                 check the whole install
omavoi status [--json]        state, for a bar module
omavoi record start|stop|toggle|cancel

omavoi history -n 10 -v       recent takes with diagnostics
omavoi history rm ID [ID...]  delete those takes, and their recordings
omavoi history clear          delete all of them
omavoi last [--raw|--json]    everything about the last one
omavoi stats                  empty rate, RTF, input level

omavoi mode list|show|new|rm|set|match|unmatch|step
omavoi model list|pull|rm|use
omavoi dict add|rm|list       heard -> meant
omavoi names add|rm|dryrun|enable
omavoi config get|set|edit|show
omavoi transcribe FILE [--mode M]

omavoi hotkey check           why the key is not working, if it is not
omavoi hotkey capture         name the key or combination you press
omavoi llm list|check         the three LLM configurations, and whether one answers
omavoi speech show|check      the remote speech endpoint, and whether it answers
omavoi secrets set NAME       a key, read from stdin — never from argv
omavoi inject TEXT            type into the focused window, to test the route
omavoi reload                 make the daemon re-read the config
```

## Engines

| | install size | notes |
|---|---|---|
| whisper.cpp on Vulkan | ~10 MB | NVIDIA, AMD, Intel, and CPU where there is no GPU |
| any OpenAI-compatible API | — | audio leaves the machine |

There used to be a second local engine, faster-whisper on CTranslate2, for
about 2.2 GB of CUDA wheels and NVIDIA only. On the short push-to-talk takes
this program is for, the two measured within a tenth of a second of each
other — on one RTX 5070 Ti, Vulkan decoded a 3 s clip in 0.25 s against
CUDA's 0.39 s, because fixed overhead dominates at that length. It was a
second engine, a second model format, a second set of runtime library
problems and a 2.2 GB install for a difference nobody could feel, so it is
gone. A config that still names it is moved to whisper.cpp on the next load,
and its model with it.

An LLM step is separate and optional, configured per mode: a local llama.cpp
server, the Claude API, or any OpenAI-compatible endpoint.

## Interface language

The console is in English by default and also speaks German, Spanish, French,
Vietnamese, Chinese, Japanese and Thai. Pick one from the dropdown in the top
nav, or set it from a terminal:

```bash
omavoi config set ui.language ja
```

An empty value follows the environment locale. This is the interface language
only — what you dictate in, and what an LLM step translates to, are per-mode
settings on the Modes tab.

## Status

Working: the daemon, all five console tabs, the HUD, the bar module, modes and
models editable from the UI, the dictionary and names. `omavoi status` and the
Models tab both report which engine is loaded right now, per family, which is
not the same question as which one is configured. Switching to a mode whose
local model will not fit in free VRAM is refused rather than left to fail on
the next take — `omavoi mode use <name> --force` overrides. Leaving a mode
unloads the local LLM it was using, and the fit check counts that memory as
available, so switching between an LLM mode and a speech-only one is not
blocked by the model on its way out.

The HUD reads its own settings: whether to show it at all, how long a result
stays up, and how big the strip is. All three were config keys that nothing
read — the settings page offered three dwell buttons and had one behaviour —
so a `hud_position` of `cursor` or `window`, which was never built, is now
refused by `config set` rather than accepted and ignored.

Not yet: the dictionary's history-mined suggestions and its try-it box; a
modifier held to force a mode for one take; and the HUD at the cursor or over
the focused window rather than at the bottom of the screen.

MIT.
