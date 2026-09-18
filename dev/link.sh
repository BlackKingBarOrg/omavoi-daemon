#!/usr/bin/env bash
#
# Point this machine's Omavoi at this checkout, so an edit is live on the next
# daemon start and there is no copy to forget to rebuild.
#
#   dev/link.sh              # link it, and restart the daemon if a unit is there
#   dev/link.sh --real [ref] # install from origin the way a user does (default origin/master)
#   dev/link.sh --check      # report what is running now, change nothing
#
# The daemon's unit runs `%h/.local/bin/omavoi daemon`, and that path is already
# a symlink into whatever uv tool installed. So the whole of this is: make it a
# symlink into the venv this checkout owns instead. The unit does not change,
# the plugin does not change, and every `omavoi` on the command line follows.
#
# It is idempotent on purpose. The plugin's Update screen installs the daemon
# with `uv tool install <pinned commit>`, which replaces the symlink with a real
# file pointing at a snapshot of some other commit — silently, because
# everything still works, just not on your code. Re-running this puts it back.
# That is also why step 2 exists at all: a uv tool copy and this checkout can
# both be installed, and the copy is the one that wins.
#
# Not here: the systemd unit and the keybindings. Those ship with the plugin and
# its install.sh writes them; a development script that also wrote them would be
# a second copy to keep in step. This one only decides which code the unit runs.

set -uo pipefail

MODE=dev
case "${1:-}" in
  --real)  MODE=real ;;
  --check) MODE=check ;;
  "")      ;;
  *)       echo "usage: dev/link.sh [--real [ref]|--check]" >&2; exit 2 ;;
esac
REF="${2:-origin/master}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
BIN="$HOME/.local/bin/omavoi"

say() { printf '  %s\n' "$*"; }
step() { printf '\n%s\n' "$*"; }

[[ -f "$REPO/pyproject.toml" ]] || { echo "not an omavoi checkout: $REPO" >&2; exit 1; }

if [[ "$MODE" == dev ]]; then
  command -v uv >/dev/null || { echo "uv is not installed: https://docs.astral.sh/uv/" >&2; exit 1; }

  step "1. the venv, with the dev group"
  # Installs the project itself editable, so src/ is what gets imported, plus
  # pytest and ruff. One venv for running it and for checking it.
  uv sync --project "$REPO" 2>&1 | sed 's/^/  /'
  (( ${PIPESTATUS[0]} == 0 )) || exit 1

  step "2. any uv tool copy, which would win over this checkout"
  if uv tool list 2>/dev/null | grep -q '^omavoi '; then
    # This also removes ~/.local/bin/omavoi, which step 3 then recreates.
    uv tool uninstall omavoi 2>&1 | sed 's/^/  /'
  else
    say "none installed"
  fi

  step "3. ~/.local/bin/omavoi -> this checkout"
  mkdir -p "$HOME/.local/bin"
  # A real file here is someone else's install and worth naming before it goes;
  # a symlink is either ours already or a uv tool leftover, and is just replaced.
  if [[ -f "$BIN" && ! -L "$BIN" ]]; then
    say "replacing a regular file at $BIN"
  fi
  ln -sfn "$VENV/bin/omavoi" "$BIN"
  say "$BIN -> $(readlink "$BIN")"

  step "4. the daemon"
  if systemctl --user list-unit-files omavoid.service >/dev/null 2>&1 \
     && [[ -n "$(systemctl --user list-unit-files omavoid.service --no-legend 2>/dev/null)" ]]; then
    systemctl --user restart omavoid && say "restarted omavoid"
  else
    say "no omavoid.service — it ships with the plugin, so install that first:"
    say "  omarchy plugin add https://github.com/BlackKingBarOrg/omavoi-shell-plugin --enable --yes"
  fi

elif [[ "$MODE" == real ]]; then
  # What a user gets, so that the difference between "my code is broken" and
  # "this is broken" can be settled by running both. The plugin installs the
  # daemon at one pinned commit rather than at whatever master is, so passing
  # that commit as `ref` is what reproduces a given release exactly; the default
  # is the branch it is pinned from.
  command -v uv >/dev/null || { echo "uv is not installed" >&2; exit 1; }
  url="$(git -C "$REPO" remote get-url origin 2>/dev/null)"
  [[ -n "$url" ]] || { echo "no origin remote on $REPO" >&2; exit 1; }
  # A git URL, not a path: uv resolves ssh and https forms, and an install from
  # the local checkout would be the checkout again under another name.
  url="${url/#git@github.com:/https://github.com/}"
  # --verify and ^{commit}: plain `rev-parse` echoes an unknown ref back on
  # stdout, so a non-empty answer is not the same as a resolved one, and the
  # bad name would have gone to uv as if it were a commit.
  sha="$(git -C "$REPO" rev-parse --verify --quiet "$REF^{commit}")" \
    || { echo "no such commit: $REF" >&2; exit 1; }

  step "1. what this install will not have"
  dirty="$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"
  (( dirty )) && say "$dirty uncommitted change(s) — this installs from the remote, not from here"
  if ! git -C "$REPO" merge-base --is-ancestor "$sha" origin/master 2>/dev/null; then
    say "${REF} (${sha:0:7}) is not on origin/master — uv can only fetch what is pushed"
  fi

  step "2. install from $url at ${sha:0:7}"
  # --force because ~/.local/bin/omavoi is currently the link, and uv will not
  # write over something it did not put there.
  uv tool install --force "git+$url@$sha" 2>&1 | tail -3 | sed 's/^/  /'

  step "3. the daemon"
  systemctl --user restart omavoid 2>/dev/null && say "restarted omavoid" \
    || say "no omavoid.service to restart"
fi

step "state"
# The question this answers is "is the thing running my code", and the honest
# form of it is the import path, not the version string: the version has not
# moved since 0.1.0 and never will per commit.
if [[ -L "$BIN" ]]; then
  target="$(readlink -f "$BIN" 2>/dev/null)"
  if [[ "$target" == "$VENV/bin/omavoi" ]]; then
    say "development: $BIN -> this checkout"
  elif [[ "$target" == "$HOME/.local/share/uv/tools/omavoi/"* ]]; then
    # --real puts this here on purpose, so it is a state and not a fault. A
    # bare !! would cry wolf every time someone compared against a release.
    say "installed: $BIN -> a uv tool copy, not this checkout"
  else
    say "!! $BIN -> $target"
  fi
elif [[ -e "$BIN" ]]; then
  say "!! $BIN is a regular file, not a link to this checkout"
else
  say "!! $BIN does not exist"
fi

if [[ -x "$VENV/bin/python" ]]; then
  resolved="$("$VENV/bin/python" -c 'import omavoi,os;print(os.path.dirname(omavoi.__file__))' 2>/dev/null)"
  [[ "$resolved" == "$REPO/src/omavoi" ]] \
    && say "venv: import omavoi -> $resolved" \
    || say "!! venv: import omavoi -> ${resolved:-failed}"
else
  # git clean -xdf takes .venv with it, and then the symlink dangles and the
  # daemon cannot start. Re-running this script rebuilds it.
  say "!! no venv at $VENV — run dev/link.sh"
fi

pid="$(systemctl --user show omavoid -p MainPID --value 2>/dev/null)"
if [[ -n "$pid" && "$pid" != "0" ]] && [[ -r "/proc/$pid/cmdline" ]]; then
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  if [[ "$cmd" == "$VENV/bin/python "* ]]; then
    say "daemon pid $pid runs this checkout's python"
  elif [[ "$cmd" == "$HOME/.local/share/uv/tools/omavoi/"* ]]; then
    say "daemon pid $pid runs the uv tool copy"
  else
    say "!! daemon pid $pid: $cmd"
  fi
else
  say "daemon not running"
fi

head="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
dirty="$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"
say "checkout at ${head:-?}, $dirty uncommitted change(s)"

# Linking is not pulling. This script points the daemon at the checkout; what
# is *in* the checkout is git's business, and the two are easy to conflate --
# "I ran the script, so I am on the latest" is wrong the moment someone else
# pushes. Read from the tracking ref rather than the network: a --check that
# reached out would be a surprise, and `git fetch` is the user's call.
upstream="$(git -C "$REPO" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"
if [[ -n "$upstream" ]]; then
  behind="$(git -C "$REPO" rev-list --count "HEAD..$upstream" 2>/dev/null)"
  ahead="$(git -C "$REPO" rev-list --count "$upstream..HEAD" 2>/dev/null)"
  if (( behind > 0 && ahead > 0 )); then
    # Diverged is the state worth naming in full: reporting only the behind
    # half reads as "just pull", and a pull here is a merge or a rebase.
    say "diverged from $upstream as of the last fetch: $ahead ahead, $behind behind"
  elif (( behind > 0 )); then
    say "$behind commit(s) behind $upstream as of the last fetch — git pull, then dev/link.sh"
  elif (( ahead > 0 )); then
    say "$ahead commit(s) ahead of $upstream, nothing to pull"
  else
    say "level with $upstream as of the last fetch"
  fi
else
  say "no upstream branch, so nothing to be behind"
fi

printf '\nEdit, then:\n'
printf '  omavoi <anything>                 already your code, new process each time\n'
printf '  systemctl --user restart omavoid  for the daemon\n'
printf '  uv run pytest / uv run ruff check\n'
