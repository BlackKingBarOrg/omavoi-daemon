"""The resident daemon: model stays hot, mic stays open, key stays watched."""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import threading
import time
from typing import Any

from . import asr, config, modes, notify, paths
from .audio import RingCapture
from .history import History
from .hotkey import HotkeyListener, HotkeyUnavailable
from .inject import Injector
from .ipc import ping, request  # noqa: F401  (re-export)
from .pipeline import Pipeline
from .window import active_window

log = logging.getLogger(__name__)

# Every hotkey failure ends up in front of someone who did not cause it and
# cannot see the cause. Both notifications carry this, and the console page it
# names diagnoses all four causes and fixes the two that need no password.
_HOTKEY_ROUTE = ("Open the console with SUPER+ALT+V and look at Settings \u2192 "
                 "hotkey: it names which of the causes this is and fixes the two "
                 "that need no password. From a terminal: `omavoi hotkey check`.")

IDLE, RECORDING, BUSY = "idle", "recording", "transcribing"


class AlreadyRunning(RuntimeError):
    pass


class Daemon:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.sock_path = paths.socket_file()
        self.state_file = paths.state_dir() / "state"

        self.audio = RingCapture(cfg)
        self.backend = asr.build(cfg)
        self.pipeline = Pipeline(cfg, self.backend, Injector(cfg), History(cfg))
        self.forced_mode = ""
        self._mode_hint = "default"
        self.hotkey: HotkeyListener | None = None

        self._lock = threading.Lock()
        self._state = IDLE
        self._mark = 0
        self._started_at = 0.0
        self._stop = threading.Event()
        self._server: socket.socket | None = None
        self._takes = 0
        self._boot = time.time()
        self._last: dict[str, Any] = {}
        # Connections that asked to be told about state changes. The HUD has
        # to appear the instant the key goes down, so it is pushed to, never
        # polled by.
        self._subs: list[socket.socket] = []
        self._subs_lock = threading.Lock()
        self._ticker: threading.Thread | None = None
        self._watcher: threading.Thread | None = None
        self._config_stamp = self._stamp()

    # -- state -------------------------------------------------------------

    def _stamp(self) -> float:
        try:
            return paths.config_file().stat().st_mtime
        except OSError:
            return 0.0

    def _watch_config(self) -> None:
        """Reload when config.toml changes on disk.

        The daemon read its config once at startup, so every edit — from the
        console, from the CLI, from an editor — did nothing until someone
        remembered to reload. Editing settings and having them not apply is
        not a thing to ask people to remember.
        """
        while not self._stop.wait(1.0):
            stamp = self._stamp()
            if stamp == self._config_stamp or stamp == 0.0:
                continue
            # Let a writer finish: config.write replaces the file, but an
            # editor may still be mid-save.
            time.sleep(0.3)
            self._config_stamp = self._stamp()
            log.info("config.toml changed on disk, reloading")
            self.reload()

    def _broadcast(self, payload: dict[str, Any]) -> None:
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
        with self._subs_lock:
            dead = []
            for sock in self._subs:
                try:
                    sock.sendall(line)
                except OSError:
                    dead.append(sock)
            for sock in dead:
                self._subs.remove(sock)
                try:
                    sock.close()
                except OSError:
                    pass

    def _tick(self) -> None:
        """Feed the HUD a level while recording. ~20 Hz is enough to read."""
        while self._state == RECORDING and not self._stop.is_set():
            self._broadcast({
                "event": "level",
                "level": round(self.audio.level(0.08), 4),
                "seconds": round(time.monotonic() - self._started_at, 2),
            })
            time.sleep(0.05)

    def _set_state(self, state: str) -> None:
        previous = self._state
        self._state = state
        # A plain file so waybar/omarchy-shell can poll without a socket client.
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(state + "\n", encoding="utf-8")
        except OSError:
            pass
        if state != previous:
            self._broadcast({"event": "state", "state": state, "mode": self._mode_hint})
            if state == RECORDING:
                self._ticker = threading.Thread(target=self._tick, name="omavoi-level",
                                                daemon=True)
                self._ticker.start()

    def _needed_llms(self) -> set[str] | None:
        """The LLM names the mode in use can reach. None means "cannot tell".

        With window matching on, the mode follows the focused window, so any
        mode's LLM could be wanted a keystroke from now — unloading on that
        signal would thrash a 5 GB server against alt-tab.
        """
        switching = self.cfg.get("switching", {}) or {}
        if switching.get("by_window"):
            return None
        mode = modes.resolve(self.cfg, None, self.forced_mode)
        return {str(step.llm) for step in (mode.steps or []) if getattr(step, "llm", "")}

    def _apply_mode_speech_model(self) -> None:
        """Load the weights the mode in use names, if they differ.

        Done when the mode changes rather than when the take arrives: a swap
        measured 3.7 s from spawn to a usable transcription for large-v3, and
        that belongs to the switch, not to the sentence you are dictating. A
        mode that names nothing leaves whatever is loaded alone.
        """
        if self._state != IDLE:
            # Mid-take. The next mode change or the end of this take gets it.
            return
        mode = modes.resolve(self.cfg, None, self.forced_mode)
        want = str(getattr(mode, "speech_model", "") or "") or str(
            self.cfg["speech"].get("model", "")
        )
        try:
            loaded = str(self.backend.state().get("model", ""))
        except Exception:
            return
        if not want or want == loaded:
            return
        use = getattr(self.backend, "use", None)
        if not callable(use):
            return
        with self._lock:
            self._set_state(BUSY)
        try:
            use(want)
            log.info("mode %s: speech model is now %s", mode.name, want)
        except Exception as exc:
            # Keeping the old weights is better than a mode that cannot hear.
            log.error("mode %s wanted speech model %s: %s", mode.name, want, exc)
        finally:
            with self._lock:
                self._set_state(IDLE)

    def _rebind_hotkey(self) -> None:
        """Put the listener on the key the config now names.

        A reload rebuilt everything except this, so changing the key did
        nothing until the daemon was restarted — and status reported the file's
        value, which made the config look like it had taken effect.
        """
        want_key = str(self.cfg["hotkey"].get("key", ""))
        want_mode = str(self.cfg["hotkey"].get("mode", "push_to_talk"))
        if not self.cfg["hotkey"].get("enabled", True):
            if self.hotkey is not None:
                self.hotkey.stop()
                self.hotkey = None
                log.info("hotkey disabled")
            return
        if self.hotkey is not None:
            if (self.hotkey.key_name == want_key
                    and self.hotkey.mode == want_mode):
                return
            self.hotkey.stop()
            self.hotkey = None
        try:
            self.hotkey = HotkeyListener(
                self.cfg,
                on_press=lambda: self.begin(),
                on_release=lambda: self.end(),
                on_toggle=lambda: self.toggle(),
                on_availability=self._hotkey_availability,
            )
            self.hotkey.start()
            log.info("hotkey rebound to %s (%s)", want_key, want_mode)
        except HotkeyUnavailable as exc:
            # No hotkey at all is the one outcome that looks like a broken
            # microphone, so it is said loudly and left visible in status.
            #
            # A failed rebind is worse than a failed start: the user pressed a
            # key a second ago and the only evidence was a journal line. Now
            # the key goes quiet *and* says so, with the place it can be
            # looked at — a message naming a cause and no route is how the
            # last one of these went unfixed for a week.
            log.error("hotkey unavailable after rebind: %s", exc)
            notify.send(f"Omavoi: {want_key} cannot be used",
                        f"{exc}\n\n{_HOTKEY_ROUTE}", urgency="critical")
            self.hotkey = None

    def capture_key(self, timeout: float = 8.0) -> dict[str, Any]:
        """Read one key press and name it, using the devices already open here.

        The console's "press a key" button used to run `omavoi hotkey capture`
        as a child of the shell, which opens the devices itself — and so it
        failed on exactly the machines where it was most needed: a session that
        joined the `input` group after logging in has no access, while the
        daemon, started through newgrp or simply started later, does. The
        process that can read a key should be the one that reads it.

        Two things this deliberately does:

        The listener is paused for the duration. It is watching for the current
        key on the same devices, so pressing that one while choosing a new one
        would have started a recording nobody asked for.

        And it blocks this socket while it waits. The server handles one
        connection at a time, so a status probe in those seconds waits too —
        acceptable for a deliberate act with the settings page open, and the
        reason the timeout is capped rather than taken on trust.
        """
        from .hotkey import HotkeyUnavailable, capture

        timeout = max(1.0, min(float(timeout), 15.0))
        paused, listener = False, self.hotkey
        if listener is not None:
            listener.stop()
            paused = True
        try:
            name = capture(timeout)
        except HotkeyUnavailable as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            log.exception("capture failed")
            return {"ok": False, "error": str(exc)}
        finally:
            if paused:
                try:
                    listener.start()
                except HotkeyUnavailable as exc:
                    log.error("could not resume the hotkey after capture: %s", exc)
                    self.hotkey = None
        if not name:
            return {"ok": False, "error": f"no key pressed within {timeout:.0f}s"}
        return {"ok": True, "key": name}

    def _hotkey_availability(self, ok: bool, detail: str) -> None:
        """Say when the key stops being readable, and when it comes back.

        Both directions, because a message that only ever appears when things
        break leaves the user unsure whether the fix took.
        """
        if ok:
            notify.send("Omavoi", detail, urgency="low")
        else:
            notify.send("Omavoi: the key stopped working",
                        f"{detail}\n\n{_HOTKEY_ROUTE}", urgency="critical")

    def _release_idle_llms(self) -> None:
        """Stop local LLM servers the current mode does not use.

        A mode is a chain, and leaving one leaves its models resident: the
        server kept its VRAM, which then made the fit check refuse a different
        mode over memory it could have reclaimed.
        """
        if self._state == BUSY:
            # A take is in the middle of using one. The next sweep gets it,
            # and _process runs one as soon as it is done.
            return
        needed = self._needed_llms()
        if needed is None:
            return
        freed = self.pipeline.llms.retain(needed)
        if freed:
            log.info("unloaded llm %s — the mode in use does not name %s",
                     ", ".join(freed), "them" if len(freed) > 1 else "it")

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "state": self._state,
            "pid": os.getpid(),
            "uptime": round(time.time() - self._boot, 1),
            "takes": self._takes,
            "backend": self.backend.describe(),
            # Structured, because "which engine is actually running" cannot be
            # read out of the sentence above, and the config only says which
            # one was asked for.
            "engines": {
                "speech": self.backend.state(),
                "llm": self.pipeline.llms.states(),
            },
            # What the daemon would actually use, from its own copy of the
            # config. The CLI reads the file, and the two differ for about a
            # second after an edit — this is the answer that types.
            "mode": modes.resolve(self.cfg, None, self.forced_mode).name,
            "switching": dict(self.cfg.get("switching", {})),
            "hotkey": {
                "enabled": bool(self.hotkey),
                # What the listener is bound to, not what the file says. These
                # differed silently after a rebind, so the config was checked,
                # found correct, and the key still did nothing.
                "key": (self.hotkey.key_name if self.hotkey
                        else self.cfg["hotkey"]["key"]),
                "mode": (self.hotkey.mode if self.hotkey
                         else self.cfg["hotkey"]["mode"]),
                "configured_key": self.cfg["hotkey"]["key"],
                "devices": self.hotkey.device_names if self.hotkey else [],
            },
            "audio": {
                "healthy": self.audio.healthy,
                "level": round(self.audio.level(), 4),
                "preroll": self.audio.preroll,
                "tail": self.audio.tail,
            },
            "recording_seconds": (
                round(time.monotonic() - self._started_at, 2) if self._state == RECORDING else 0.0
            ),
        }

    # -- recording ---------------------------------------------------------

    def begin(self) -> dict[str, Any]:
        # Asked before the lock. active_window() shells out to hyprctl with a
        # one-second timeout, and holding the lock across a subprocess makes
        # every other caller wait behind it — the key release included, which
        # is the one thing that must not be late.
        window = active_window()
        with self._lock:
            if self._state == RECORDING:
                return {"ok": True, "state": RECORDING, "note": "already recording"}
            if self._state == BUSY:
                return {"ok": False, "state": BUSY, "error": "still transcribing the previous take"}
            if not self.audio.healthy:
                return {"ok": False, "state": self._state, "error": "audio capture is not healthy"}
            # The mark reaches back preroll seconds, so the first syllable
            # spoken before the key registered is already included.
            self._mark = self.audio.mark()
            self._started_at = time.monotonic()
            self._mode_hint = modes.resolve(self.cfg, window, self.forced_mode).name
            self._set_state(RECORDING)
        log.debug("recording started")
        return {"ok": True, "state": RECORDING}

    def end(self, *, discard: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._state != RECORDING:
                return {"ok": False, "state": self._state, "error": "not recording"}
            mark = self._mark
            self._set_state(IDLE if discard else BUSY)
        if discard:
            log.debug("recording cancelled")
            return {"ok": True, "state": IDLE, "discarded": True}

        threading.Thread(target=self._process, args=(mark,), name="omavoi-asr", daemon=True).start()
        return {"ok": True, "state": BUSY}

    def toggle(self) -> dict[str, Any]:
        return self.end() if self._state == RECORDING else self.begin()

    def _process(self, mark: int) -> None:
        try:
            capture = self.audio.take(mark)
            entry = self.pipeline.process(
                capture, window=active_window(), forced_mode=self.forced_mode
            )
            self._takes += 1
            self._last = entry
            self._broadcast({
                "event": "result",
                "text": entry.get("text", ""),
                "rejected": entry.get("rejected", ""),
                "changes": len(entry.get("post", {}).get("changes") or []),
                "warnings": entry.get("warnings", []),
                "mode": entry.get("mode", {}).get("name", ""),
                "seconds": entry.get("total_seconds", 0.0),
            })
        except Exception:
            log.exception("processing failed")
        finally:
            with self._lock:
                self._set_state(IDLE)
            # A mode switch during the take skipped its sweep; this is the
            # first moment the servers are idle again.
            self._release_idle_llms()

    # -- server ------------------------------------------------------------

    def _handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        cmd = str(payload.get("cmd", ""))
        if cmd == "status":
            return self.status()
        if cmd == "setup":
            from . import setup as setup_mod

            return {"ok": True, **setup_mod.check(self.cfg).as_dict()}
        if cmd == "inject":
            # Injection from inside the daemon, on demand. The CLI can already
            # inject, and when the two disagree the difference is the process
            # doing it — not the code, which is shared.
            text = str(payload.get("text", "") or "omavoi daemon test")
            win = active_window()
            outcome = self.pipeline.injector.inject(text, win)
            return {"ok": outcome.ok, "window": win.as_dict(),
                    "inject": outcome.as_dict()}
        if cmd == "last":
            return {"ok": True, "entry": self._last}
        if cmd == "mode":
            name = str(payload.get("name", ""))
            if name and name not in self.cfg.get("modes", {}):
                return {"ok": False, "error": f"no such mode: {name}"}
            self.forced_mode = name
            self._release_idle_llms()
            self._apply_mode_speech_model()
            return {"ok": True, "forced_mode": name}
        if cmd == "start":
            return self.begin()
        if cmd == "stop":
            return self.end()
        if cmd == "cancel":
            return self.end(discard=True)
        if cmd == "toggle":
            return self.toggle()
        if cmd == "capture":
            return self.capture_key(float(payload.get("timeout", 8.0)))
        if cmd == "reload":
            return self.reload()
        if cmd == "quit":
            self._stop.set()
            return {"ok": True, "state": "stopping"}
        return {"ok": False, "error": f"unknown command {cmd!r}"}

    def reload(self) -> dict[str, Any]:
        """Re-read config. Anything but the model can change without a restart."""
        try:
            new = config.load()
        except SystemExit as exc:
            return {"ok": False, "error": str(exc)}

        model_changed = (
            new["speech"] != self.cfg["speech"]
        )
        self.cfg = new
        self._config_stamp = self._stamp()
        # The registry is carried over, not rebuilt: it owns running llama
        # servers, and a reload is not a reason to drop 5 GB of resident
        # weights and start again on the next take.
        llms = self.pipeline.llms
        llms.update(new)
        self.pipeline = Pipeline(new, self.backend, Injector(new), History(new),
                                 registry=llms)
        self._release_idle_llms()
        self._apply_mode_speech_model()
        self._rebind_hotkey()
        log.info("config reloaded%s", "; speech settings changed, restart the daemon" if model_changed else "")
        return {"ok": True, "reloaded": True, "model_restart_required": model_changed}

    def _serve(self) -> None:
        assert self._server is not None
        self._server.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                conn.settimeout(5.0)
                line = conn.makefile("rb").readline()
                payload = json.loads(line) if line else {}
                if str(payload.get("cmd", "")) == "subscribe":
                    # Bounded, not blocking. _broadcast sends under _lock, so
                    # a subscriber that stops reading — a hung console, a
                    # suspended shell — fills its buffer, blocks sendall
                    # forever, and takes the hotkey down with it: begin() and
                    # end() both wait on that same lock. socket.timeout is an
                    # OSError, so _broadcast already treats it as dead and
                    # drops it. A subscriber too slow to keep up is not worth
                    # a frozen daemon.
                    conn.settimeout(2.0)
                    conn.sendall(
                        (json.dumps({"ok": True, **self.status()}) + "\n").encode()
                    )
                    with self._subs_lock:
                        self._subs.append(conn)
                    continue  # the socket stays open and is closed by _broadcast
                reply = self._handle(payload)
            except json.JSONDecodeError as exc:
                reply = {"ok": False, "error": f"not valid JSON: {exc}"}
            except Exception as exc:
                log.exception("command handler error")
                reply = {"ok": False, "error": str(exc)}
            try:
                conn.sendall(json.dumps(reply, ensure_ascii=False).encode() + b"\n")
            except OSError:
                pass
            finally:
                conn.close()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if ping(self.sock_path) is not None:
            raise AlreadyRunning(f"a daemon is already listening on {self.sock_path}")
        # A socket left behind by a crash is safe to clear: nothing answered it.
        self.sock_path.unlink(missing_ok=True)
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.sock_path))
        os.chmod(self.sock_path, 0o600)
        self._server.listen(8)

        log.info("loading %s ...", self.cfg["speech"]["model"])
        self.backend.load()
        log.info("speech ready: %s", self.backend.describe())

        self.audio.start()
        log.info("ring capture running (pre-roll %.2fs, tail %.2fs)",
                 self.audio.preroll, self.audio.tail)

        if self.cfg["hotkey"].get("enabled", True):
            try:
                self.hotkey = HotkeyListener(
                    self.cfg,
                    on_press=lambda: self.begin(),
                    on_release=lambda: self.end(),
                    on_toggle=lambda: self.toggle(),
                    on_availability=self._hotkey_availability,
                )
                self.hotkey.start()
            except HotkeyUnavailable as exc:
                log.error("hotkey unavailable: %s", exc)
                notify.send("Omavoi: hotkey unavailable",
                            f"{exc}\n\n{_HOTKEY_ROUTE}", urgency="critical")
                # The rebind path has always done this and the start path
                # never did, so a listener that failed to open a single device
                # stayed assigned — and status() then reported enabled:true
                # with an empty device list. `omavoi hotkey check` read that
                # and said "the daemon is listening on it:" over a dead key,
                # which is the one thing that command exists not to do.
                self.hotkey = None

        self._watcher = threading.Thread(target=self._watch_config,
                                         name="omavoi-config", daemon=True)
        self._watcher.start()

        self._set_state(IDLE)
        notify.send("Omavoi ready", self.backend.describe(), urgency="low")

    def run(self) -> None:
        self.start()

        def on_signal(signum: int, _frame: Any) -> None:
            if signum == signal.SIGHUP:
                self.reload()
            else:
                self._stop.set()

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGHUP, on_signal)

        thread = threading.Thread(target=self._serve, name="omavoi-ipc", daemon=True)
        thread.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.2)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        log.info("shutting down")
        self._stop.set()
        if self.hotkey is not None:
            self.hotkey.stop()
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()
        self.pipeline.llms.close()
        self.audio.stop()
        if self._server is not None:
            self._server.close()
        self.sock_path.unlink(missing_ok=True)
        self._broadcast({"event": "state", "state": "stopped"})
        with self._subs_lock:
            for sock in self._subs:
                try:
                    sock.close()
                except OSError:
                    pass
            self._subs.clear()
        self._set_state("stopped")
