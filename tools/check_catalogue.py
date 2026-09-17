#!/usr/bin/env python3
"""Do the catalogue's sources still exist, and are its sizes true?

`size_mb` is not decoration. It decides whether a mode switch is refused —
gpu.needed_mb is computed from it — and it is the denominator of the download
percentage the console shows. Both were wrong in the same direction for
eighteen of the nineteen entries: the ggml and ct2 sizes had been taken from
a source quoting decimal megabytes into a field every consumer reads as
mebibytes, so each was 5% high. ggml:large-v3 claimed 3095 against a real
2952, and the fit gate asked for 147 MiB of VRAM that no model would use.

The symptom had even been worked around without being found: the console
caps the download percentage below 100 because "size_mb is the catalogue's
stated size, not the byte count", which is true and was covering a
systematic unit error rather than rounding.

Nothing in the test suite can catch this — it needs the ground truth, which
is on Hugging Face. So it lives here, and it is worth running before a
release and whenever an entry is added.

Usage: tools/check_catalogue.py [--fix]
Exit 1 if a source is unreachable or a size is more than 2% out.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omavoi import models  # noqa: E402

UA = {"User-Agent": "omavoi-catalogue-check"}
# Rounding and a repo that gains a README are not news; a unit error is.
TOLERANCE = 2.0


def _file_mib(repo: str, filename: str) -> int | None:
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD", headers=UA), timeout=20
        ) as response:
            length = response.headers.get("Content-Length")
            return round(int(length) / 1048576) if length and length.isdigit() else None
    except urllib.error.HTTPError as exc:
        print(f"    unreachable: {url} -> HTTP {exc.code}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"    unreachable: {url} -> {type(exc).__name__}: {exc}")
        return None


def _repo_mib(repo: str) -> int | None:
    """What a ct2 download actually fetches: everything but the originals."""
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=25
        ) as response:
            data = json.load(response)
    except Exception as exc:  # noqa: BLE001
        print(f"    unreachable: {url} -> {type(exc).__name__}: {exc}")
        return None
    total = 0
    for sibling in data.get("siblings") or []:
        name = sibling.get("rfilename", "")
        if name != "model.bin" and name.endswith(
            (".safetensors", ".pt", ".bin.index.json", ".msgpack", ".h5")
        ):
            continue
        if name.startswith(".") or name.lower().startswith("readme"):
            continue
        total += sibling.get("size") or 0
    return round(total / 1048576)


def main(argv: list[str]) -> int:
    fix = "--fix" in argv
    source = Path(__file__).resolve().parent.parent / "src" / "omavoi" / "models.py"
    text = source.read_text(encoding="utf-8")
    problems, fixes = 0, 0

    print(f"  {'key':32s} {'stated':>7s} {'real':>7s} {'off':>6s}")
    for spec in models.CATALOG:
        if not spec.repo:
            continue
        real = _file_mib(spec.repo, spec.filename) if spec.filename else _repo_mib(spec.repo)
        if real is None:
            problems += 1
            continue
        off = (spec.size_mb - real) / real * 100 if real else 0.0
        bad = abs(off) > TOLERANCE
        print(f"  {'FAIL' if bad else 'ok  '} {spec.key:26s} {spec.size_mb:>7} "
              f"{real:>7} {off:>5.0f}%")
        if not bad:
            continue
        problems += 1
        if fix:
            pattern = re.compile(
                rf'(ModelSpec\("{re.escape(spec.id)}",[^)]*?'
                rf'"{re.escape(spec.filename)}",\s*){spec.size_mb}\b'
            )
            text, count = pattern.subn(rf"\g<1>{real}", text, count=1)
            if count:
                fixes += 1

    if fix and fixes:
        source.write_text(text, encoding="utf-8")
        print(f"\n  {fixes} size(s) rewritten in {source}")
    if problems:
        print(f"\n  {problems} problem(s). "
              f"{'Re-run to confirm.' if fix else 'Re-run with --fix to rewrite the sizes.'}")
        return 1
    print("\n  every source resolves and every size is within "
          f"{TOLERANCE:g}% of the real one")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
