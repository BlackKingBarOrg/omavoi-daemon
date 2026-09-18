#!/usr/bin/env bash
#
# Put this machine back to before Omavoi was ever installed, so the install can
# be tested again from nothing.
#
#   dev/reset.sh                 # everything, including system packages
#   dev/reset.sh --keep-packages # leave pacman alone
#
# This is NOT the user-facing uninstall. That is:
#
#   ~/.config/omarchy/plugins/ai.bkblab.omavoi/install.sh --remove
#   omarchy plugin remove ai.bkblab.omavoi
#
# and it deliberately stops there. Removing system packages is a different kind
# of act: `-Rns` takes out whatever nothing else needs *at that moment*, and
# someone who stops using Omavoi may well still want whisper.cpp. So the
# packages are torn down here, in a development script, and never from a button
# in the plugin.
#
# One pkexec call, because polkit reports pacman as auth_admin rather than
# auth_admin_keep: every call prompts again.

set -uo pipefail

KEEP_PACKAGES=0
[[ "${1:-}" == "--keep-packages" ]] && KEEP_PACKAGES=1

PLUGIN_ID="ai.bkblab.omavoi"
PLUGIN_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/omarchy/plugins/$PLUGIN_ID"

# Only what Omavoi is the reason for. ggml-cpu is in the list because the
# install is the only reason it is on the machine: pacman reports it as
# `Required By: None`, an optdepend of ggml that nothing pulls in on its own.
# Leaving it behind would hide the one package whose absence breaks the model
# load with an assert instead of an error, which is exactly what the install
# has to get right. ggml itself is a dependency of whisper-cpp, so -Rns takes
# it with them. Deliberately absent:
#   wtype      Omarchy ships it in its base set, and voxtype uses it
#   xdotool    a general tool, and not ours to take
#   uv         explicitly installed by the user, and used for other things
PACKAGES=(whisper-cpp ggml-vulkan ggml-cpu llama-cpp)

say() { printf '  %s\n' "$*"; }
step() { printf '\n%s\n' "$*"; }

step "1. the plugin's own removal — unit, keybinding, service"
if [[ -x "$PLUGIN_DIR/install.sh" ]]; then
  "$PLUGIN_DIR/install.sh" --remove 2>&1 | sed 's/^/  /'
else
  say "no plugin installed, skipping"
  systemctl --user disable --now omavoid.service 2>/dev/null && say "disabled omavoid.service"
  # `rm -f` succeeds on a path that was never there, so the message has to test
  # for the file rather than the command — the same trap install.sh documents.
  unit="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/omavoid.service"
  if [[ -e "$unit" ]]; then
    rm -f "$unit"
    say "removed a stray unit file"
  fi
  systemctl --user daemon-reload
fi
systemctl --user reset-failed omavoid 2>/dev/null

step "2. the plugin"
# Through omarchy, not rm -rf: it also takes the widget out of shell.json,
# which a directory delete leaves behind pointing at nothing.
# -e alone is false for a symlink whose target is gone, and a dev checkout that
# has been moved leaves exactly that. `omarchy plugin add` refuses the id while
# the dangling link is there, so it has to go: -L catches it, and omarchy
# unlinks rather than deletes when the target is a link.
if [[ -e "$PLUGIN_DIR" || -L "$PLUGIN_DIR" ]]; then
  omarchy plugin remove "$PLUGIN_ID" --yes 2>&1 | sed 's/^/  /'
else
  say "not installed, skipping"
fi

step "3. the daemon"
if command -v omavoi >/dev/null; then
  uv tool uninstall omavoi 2>&1 | sed 's/^/  /'
else
  say "not on PATH, skipping"
fi

step "4. config, history, recordings, weights"
for d in "${XDG_CONFIG_HOME:-$HOME/.config}/omavoi" \
         "${XDG_STATE_HOME:-$HOME/.local/state}/omavoi" \
         "${XDG_CACHE_HOME:-$HOME/.cache}/omavoi" \
         "${XDG_DATA_HOME:-$HOME/.local/share}/omavoi"; do
  if [[ -e "$d" ]]; then
    say "removing $(du -sh "$d" 2>/dev/null | cut -f1)  $d"
    rm -rf "$d"
  fi
done
# Weights another tool owns are found and reused, never ours to delete.
say "left alone: ${XDG_DATA_HOME:-$HOME/.local/share}/voxtype"

step "5. uv's build cache, only the entries that are ours alone"
removed=0
for d in "${XDG_CACHE_HOME:-$HOME/.cache}"/uv/archive-v0/*/; do
  [[ -d "$d" ]] || continue
  entries="$(ls -A "$d" 2>/dev/null)"
  [[ "$entries" == *omavoi* ]] || continue
  # Every entry in the directory has to be ours before it can go.
  [[ -z "$(printf '%s\n' "$entries" | grep -v omavoi)" ]] || continue
  rm -rf "$d"; removed=$((removed + 1))
done
say "removed $removed entries"

step "6. system packages"
if (( KEEP_PACKAGES )); then
  say "--keep-packages given, leaving pacman alone"
else
  present=()
  for p in "${PACKAGES[@]}"; do
    pacman -Q "$p" >/dev/null 2>&1 && present+=("$p")
  done
  if (( ${#present[@]} == 0 )); then
    say "none of ${PACKAGES[*]} are installed"
  else
    say "removing: ${present[*]}"
    say "a password dialog will appear — this is the only one"
    pkexec /usr/bin/pacman -Rns --noconfirm "${present[@]}" 2>&1 | sed 's/^/  /'
    code=${PIPESTATUS[0]}
    (( code == 126 || code == 127 )) && say "cancelled, packages left in place"
  fi
fi

step "state"
command -v omavoi >/dev/null && say "!! omavoi is still on PATH" || say "omavoi: gone"
[[ -e "$PLUGIN_DIR" || -L "$PLUGIN_DIR" ]] && say "!! plugin dir remains" \
  || say "plugin: gone"
command -v whisper-server >/dev/null && say "whisper-server: still present" \
  || say "whisper-server: gone"
leftover="$(find "$HOME" -maxdepth 6 -iname '*omavoi*' \
  -not -path '*/Work/*' -not -path '*/.claude*' -not -path '*claude-cli*' 2>/dev/null)"
[[ -z "$leftover" ]] && say "nothing named omavoi outside the source tree" \
  || printf '%s\n' "$leftover" | sed 's/^/  leftover: /'

printf '\nInstall again with:\n'
printf '  omarchy plugin add https://github.com/bkblab/omavoi --enable --yes\n'
