#!/usr/bin/env python3
"""Check actual text receipt in a disposable foot window on Hyprland.

Run from the checkout: python tools/check_wtype_delivery.py [--compare-legacy]
Temporarily focuses a test terminal; closes it and restores focus afterwards.
No microphone, model, clipboard or shell input is involved. The optional
legacy control demonstrates the old stdin keymap failure with 104 characters.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import tempfile
import termios
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omavoi.inject import Injector


def capture(directory: Path) -> None:
    old = termios.tcgetattr(0)
    try:
        tty.setraw(0)
        with (directory / "received").open("wb", buffering=0) as output:
            (directory / "ready").touch()
            while not (directory / "stop").exists():
                if select.select([0], [], [], .1)[0]:
                    data = os.read(0, 65536)
                    if not data:
                        break
                    output.write(data)
    finally:
        termios.tcsetattr(0, termios.TCSANOW, old)


def hypr(*args: str) -> str:
    return subprocess.check_output(["hyprctl", *args], text=True, timeout=3)


def focus(address: str) -> None:
    # Hyprland 0.56 uses Lua dispatchers; older releases use plain arguments.
    result = subprocess.run(
        ["hyprctl", "dispatch", f'hl.dsp.focus({{window="address:{address}"}})'],
        capture_output=True, timeout=3, check=False)
    if result.returncode:
        hypr("dispatch", "focuswindow", "address:" + address)


def check_delivery(directory: Path, compare_legacy: bool) -> bool:
    original = json.loads(hypr("activewindow", "-j")).get("address")
    app_id = f"omavoi-input-check-{os.getpid()}"
    process = subprocess.Popen(
        ["foot", "--maximized", f"--app-id={app_id}", "--title=Omavoi input check",
         sys.executable, str(Path(__file__).resolve()), "--capture", str(directory)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not (directory / "ready").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("test terminal did not start")
            time.sleep(.05)
        window = next(w for w in json.loads(hypr("clients", "-j"))
                      if w["class"] == app_id and w["pid"] == process.pid)
        focus(window["address"])
        time.sleep(.3)
        cases = [(f"boundary-{n}", "甲" * n + "乙丙丁甲") for n in (99, 100, 101)]
        cases += [
            ("diverse-335", "".join(chr(0x4E00 + i) for i in range(335))),
            ("diverse-1001", "".join(chr(0x4E00 + i) for i in range(1001))),
            ("mixed-unicode", "中文 English 繁體日本語ภาษาไทย🙂👩🏽‍💻e\u0301，。 " * 6),
            ("multiline", "首行" * 80 + "\n\n末行" * 2),
            ("literal-options", "-M ctrl $HOME `literal` 中文 " * 5),
        ]
        if compare_legacy:
            cases.insert(0, ("legacy-control", "甲" * 100 + "乙丙丁甲"))
        passed = True
        for name, text in cases:
            if json.loads(hypr("activewindow", "-j")).get("address") != window["address"]:
                raise RuntimeError("focus left the test terminal; stopping")
            before = (directory / "received").stat().st_size
            start = time.monotonic()
            if name == "legacy-control":
                subprocess.run(["wtype", "-"], input=text.encode(), check=True, timeout=30)
            else:
                Injector({"inject": {}})._wtype(text)
            time.sleep(.15)
            received = (directory / "received").read_bytes()[before:].decode("utf-8")
            matches = received == text.replace("\n", "\r")
            print(json.dumps({"case": name, "sent": len(text), "received": len(received),
                              "matches": matches, "seconds": round(time.monotonic() - start, 3)}), flush=True)
            if name != "legacy-control":
                passed = passed and matches
        return passed
    finally:
        (directory / "stop").touch()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=3)
        if original:
            focus(original)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--capture", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.capture:
        capture(args.capture)
        return 0
    with tempfile.TemporaryDirectory(prefix="omavoi-input-check-") as directory:
        return 0 if check_delivery(Path(directory), args.compare_legacy) else 1


if __name__ == "__main__":
    sys.exit(main())
