"""What the GPU is holding, and whether the next model will fit.

Keeping a speech model and an LLM resident at once is the point of the
daemon, and it is also how a card quietly runs out. Checking before the
spawn is worth doing because the failure otherwise arrives as a llama.cpp
assertion several screens long, and the step then falls through silently.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Holder:
    pid: int
    name: str
    used_mb: int


def _meminfo_mb() -> tuple[int, int]:
    """Total and in-use system memory in MB, from /proc/meminfo.

    MemAvailable rather than MemFree: page cache is reclaimable, and counting
    it as used shows an idle machine at 90%.
    """
    total = avail = 0
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1])
                    if total:
                        break
    except (OSError, ValueError, IndexError):
        return 0, 0
    return total // 1024, max(0, (total - avail) // 1024)


def _rss_mb(pid: int) -> int:
    """Resident memory of one process in MB.

    On a unified-memory GPU the weights are ordinary system pages, so RSS is
    what the model actually costs -- there is no separate pool to query.
    """
    try:
        with open(f"/proc/{pid}/statm", encoding="ascii") as fh:
            resident = int(fh.read().split()[1])
    except (OSError, ValueError, IndexError):
        return 0
    return resident * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)


# fdinfo prints a unit beside each figure, and the kernel is free to pick a
# different one per line.
_DRM_UNITS = {"B": 1 / 1024, "KiB": 1, "MiB": 1024, "GiB": 1024 * 1024}


def _drm_mem_kib(pid: int) -> int:
    """GPU memory one process holds, from the DRM fdinfo the kernel exports.

    RSS is the wrong question for a model on the GPU. Weights uploaded through
    Vulkan become buffer objects, which are not pages of the process that owns
    them: whisper-server shows 111 MB of RSS while holding 1.9 GB of GTT. Ask
    RSS and a 547 MB model reads as free, and the bar drops the whole cost
    into "other programs" -- the one thing it exists to tell apart.

    Summed over regions from drm-total-*, not drm-resident-*. Which region
    holds a buffer is the driver's business and it moves them: idle, xe walks
    the weights out of gtt into system, where it publishes a total and no
    resident figure at all. Reading resident watched the same model swing
    between 1.9 GB and 90 MB depending on how recently anyone had spoken,
    while the totals stayed put. drm-shared-* is a subset rather than a
    further amount, so it is left out; and one process can hold several fds
    against the same client, so the client id is what deduplicates them.

    A unit is required, which is also what separates a memory line from
    drm-total-cycles-*: the engine counters share the prefix, carry no unit,
    and would otherwise add trillions of KiB to the bar.
    """
    try:
        fds = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError:
        return 0
    total = 0
    seen: set[str] = set()
    for fd in fds:
        try:
            with open(f"/proc/{pid}/fdinfo/{fd}", encoding="ascii",
                      errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if "drm-driver" not in text:
            continue
        fields: dict[str, str] = {}
        for line in text.splitlines():
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        # A driver that does not report a client id gets no deduplication
        # rather than having every one of its fds collapse into one.
        client = fields.get("drm-client-id") or f"fd:{fd}"
        if client in seen:
            continue
        seen.add(client)
        for key, value in fields.items():
            if not key.startswith("drm-total-"):
                continue
            parts = value.split()
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            unit = _DRM_UNITS.get(parts[1])
            if unit is None:
                continue
            total += int(int(parts[0]) * unit)
    return total


def _engine_mem_mb(pid: int) -> int:
    """What one engine costs on unified memory: its pages plus its buffers.

    The two do not overlap -- a buffer object is not mapped into the process
    holding it -- so they add rather than one covering the other.
    """
    return _rss_mb(pid) + _drm_mem_kib(pid) // 1024


def _pci_name(slot: str) -> str:
    """The card's marketing name, when pciutils is installed to say it."""
    if not slot or shutil.which("lspci") is None:
        return ""
    try:
        out = subprocess.run(["lspci", "-mm", "-s", slot],
                             capture_output=True, timeout=3, check=False)
        if out.returncode != 0:
            return ""
        # -mm quotes each field: slot, class, vendor, device, then revision
        # and subsystem. Field 3 is the one a person would recognise.
        fields = shlex.split(out.stdout.decode())
        return fields[3] if len(fields) > 3 else ""
    except (subprocess.SubprocessError, OSError, ValueError):
        return ""


def _integrated() -> dict[str, str]:
    """The integrated GPU, when that is all this machine has.

    Returns {} for anything carrying memory of its own, because a pool has a
    real total to report and this fallback would only paper over it:
      - Intel discrete cards publish lmem_total_bytes.
      - amdgpu publishes mem_info_vram_total, but on an APU that is a BIOS
        carve-out the runtime spills past into GTT, so neither number alone
        is the budget. That one wants a case of its own, not this one.
    """
    try:
        cards = sorted(c for c in os.listdir("/sys/class/drm")
                       if c.startswith("card") and "-" not in c)
    except OSError:
        return {}

    found: dict[str, str] = {}
    for card in cards:
        dev = f"/sys/class/drm/{card}/device"
        try:
            entries = set(os.listdir(f"/sys/class/drm/{card}")) | set(os.listdir(dev))
        except OSError:
            continue
        # Any card with its own memory anywhere on the machine disqualifies
        # the whole fallback -- that is the card the models will land on.
        if entries & {"lmem_total_bytes", "mem_info_vram_total"}:
            return {}
        if found:
            continue
        try:
            driver = os.path.basename(os.readlink(f"{dev}/driver"))
        except OSError:
            continue
        slot = ""
        try:
            with open(f"{dev}/uevent", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("PCI_SLOT_NAME="):
                        slot = line.strip().split("=", 1)[1]
                        break
        except OSError:
            pass
        found = {"driver": driver, "slot": slot, "name": _pci_name(slot)}
    return found


def _nvidia_vram() -> dict[str, Any]:
    """Used and total VRAM in MB, or {} when there is no NVIDIA GPU."""
    if shutil.which("nvidia-smi") is None:
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=3, check=False,
        )
        if out.returncode != 0:
            return {}
        line = out.stdout.decode().strip().splitlines()[0]
        name, used, total = (p.strip() for p in line.split(","))
        used_mb, total_mb = int(used), int(total)
        return {"name": name, "used_mb": used_mb, "total_mb": total_mb,
                "free_mb": max(0, total_mb - used_mb)}
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return {}


# Half a second, which is long enough to collapse the nineteen calls one
# `model list` makes into one, and short enough that the fit gate before a
# mode switch still reads a current figure. Measured: nvidia-smi costs 16 ms,
# so `model list --json` was spending 393 ms of its 518 asking the same
# question nineteen times.
_VRAM_TTL = 0.5
_vram_cache: tuple[float, dict[str, Any]] | None = None


def vram(*, fresh: bool = False) -> dict[str, Any]:
    """Used and total memory the GPU draws on, or {} when there is nothing to say.

    Cached for half a second. Pass fresh=True where a stale reading would be
    wrong rather than merely old — nothing does today, and the parameter is
    here so that a caller which needs it does not have to discover the cache
    by being surprised by it.

    Two shapes come back, and callers have to tell them apart. A discrete
    NVIDIA card owns a pool and nvidia-smi reports it. An integrated GPU owns
    nothing: its device memory *is* system RAM -- Vulkan advertises the device
    as UMA and llama.cpp allocates from ordinary pages -- so the honest total
    there is MemTotal, and the payload carries `unified` to say so rather than
    letting a caller print it as if a card had that much to itself.
    """
    global _vram_cache
    now = time.monotonic()
    if not fresh and _vram_cache is not None and now - _vram_cache[0] < _VRAM_TTL:
        return _vram_cache[1]

    info = _nvidia_vram()
    if info:
        _vram_cache = (now, info)
        return info
    card = _integrated()
    if not card:
        _vram_cache = (now, {})
        return {}
    total_mb, used_mb = _meminfo_mb()
    if not total_mb:
        _vram_cache = (now, {})
        return {}
    out = {
        "name": card.get("name") or "integrated GPU",
        "driver": card.get("driver", ""),
        "unified": True,
        "used_mb": used_mb,
        "total_mb": total_mb,
        "free_mb": max(0, total_mb - used_mb),
    }
    _vram_cache = (now, out)
    return out


_apps_cache: tuple[float, list[Any]] | None = None

def compute_apps() -> list[Holder]:
    """Every process nvidia-smi reports holding VRAM, biggest first.

    Cached on the same half-second as vram(), and for the same reason: one
    `model list` asked nvidia-smi eight times for this — once per model that
    would not fit, to name what is holding the memory — at 18 ms each.
    """
    global _apps_cache
    now = time.monotonic()
    if _apps_cache is not None and now - _apps_cache[0] < _VRAM_TTL:
        return _apps_cache[1]
    found = _compute_apps_now()
    _apps_cache = (now, found)
    return found


def _compute_apps_now() -> list[Holder]:
    """The query itself, uncached, so the four exits above stay as written."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=3, check=False,
        )
        if out.returncode != 0:
            return []
        found: list[Holder] = []
        for line in out.stdout.decode().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or not parts[2].isdigit():
                continue
            # nvidia-smi prints the whole command line; the basename is enough
            # to recognise, and the full one is unreadable in a warning.
            name = parts[1].split()[0].rsplit("/", 1)[-1] if parts[1] else "?"
            found.append(Holder(int(parts[0]), name, int(parts[2])))
        found.sort(key=lambda h: -h.used_mb)
        return found
    except (subprocess.SubprocessError, OSError, ValueError):
        return []


def holders(limit: int = 3) -> list[Holder]:
    """The processes holding the most VRAM, biggest first.

    Naming them turns "not enough memory" into something the user can act
    on — it is usually one obvious program, not a mystery.
    """
    return compute_apps()[:limit]


def usage_by_pid() -> dict[int, int]:
    """VRAM per process, for attributing a bar to the models that hold it."""
    return {h.pid: h.used_mb for h in compute_apps()}


def segments(total_used_mb: int, speech_pid: int, llm_pids: set[int]) -> list[dict[str, Any]]:
    """Split the used VRAM into ours-for-speech, ours-for-LLM, and everything else.

    The remainder is computed rather than summed from the other processes:
    nvidia-smi's total includes memory no compute app claims (the display,
    mostly), and a stacked bar that does not add up to the number printed
    beside it is worse than one honest catch-all.
    """
    by_pid = usage_by_pid()
    if shutil.which("nvidia-smi") is None:
        # No compute-app list to ask, so ask the kernel per process. "other" is
        # then the rest of the machine -- the same catch-all as on a card, for
        # the same reason.
        by_pid = {pid: _engine_mem_mb(pid) for pid in {speech_pid, *llm_pids} if pid}
    speech_mb = by_pid.get(speech_pid, 0) if speech_pid else 0
    llm_mb = sum(by_pid.get(p, 0) for p in llm_pids)
    return [
        {"kind": "speech", "used_mb": speech_mb},
        {"kind": "llm", "used_mb": llm_mb},
        {"kind": "other", "used_mb": max(0, int(total_used_mb) - speech_mb - llm_mb)},
    ]


def needed_mb(weights_mb: int, ctx_size: int = 4096, gpu_layers: int = 99) -> int:
    """Roughly what a GGUF model will want on the GPU.

    Weights, plus a KV cache that grows with the context, plus a little for
    the runtime. Deliberately generous: refusing to start something that
    would have just fit is a smaller sin than evicting the speech model.
    """
    if gpu_layers <= 0:
        return 0
    kv_mb = max(128, int(ctx_size / 1024 * 160))
    return int(weights_mb * 1.03) + kv_mb + 256


def fits(weights_mb: int, ctx_size: int = 4096, gpu_layers: int = 99) -> dict[str, Any]:
    """Whether the model fits in what is free right now, and why not."""
    info = vram()
    want = needed_mb(weights_mb, ctx_size, gpu_layers)
    if not info or info.get("unified"):
        # Nothing with a pool of its own to interrogate: either no NVIDIA GPU,
        # or an integrated one whose budget is system RAM. Do not block. The
        # runtime will cope or fail loudly, and overcommitting shared memory
        # costs swap rather than the allocation failure this guard is for --
        # a different conversation, and not one to have by refusing to start.
        return {"known": False, "fits": True, "needed_mb": want}
    free = int(info.get("free_mb", 0))
    ok = free >= want
    out: dict[str, Any] = {
        "known": True, "fits": ok, "needed_mb": want, "free_mb": free,
        "total_mb": info.get("total_mb", 0), "name": info.get("name", ""),
    }
    if not ok:
        out["holders"] = [{"pid": h.pid, "name": h.name, "used_mb": h.used_mb}
                          for h in holders()]
    return out


def fits_chain(wants: list[dict[str, Any]], live: set[str],
               reclaimable_mb: int = 0) -> dict[str, Any]:
    """Whether a whole chain of local models can be resident at once.

    A mode is speech plus zero or more LLM passes, and they all have to stay
    loaded together — swapping per take costs seconds. So the question that
    matters before switching modes is not "does this model fit" but "does
    everything this mode needs fit alongside what is already up".

    `wants` is [{"key", "weights_mb", "ctx_size", "gpu_layers"}]; anything in
    `live` is already resident and so already counted in used_mb.

    `reclaimable_mb` is memory the caller is about to release anyway — the
    local LLM of the mode being left. Counting it as free is the difference
    between "this mode does not fit" and "this mode does not fit beside a
    model that is on its way out".
    """
    info = vram()
    pending = [w for w in wants if w["key"] not in live]
    want = sum(needed_mb(int(w.get("weights_mb", 0)),
                         int(w.get("ctx_size", 4096)),
                         int(w.get("gpu_layers", 99))) for w in pending)
    out: dict[str, Any] = {
        # Unified memory is not knowledge of a pool: claiming otherwise would
        # have the UI print a budget it cannot show a total for.
        "known": bool(info) and not info.get("unified"),
        "fits": True,
        "needed_mb": want,
        "pending": [w["key"] for w in pending],
    }
    if not info or info.get("unified"):
        # As in fits(): no pool of its own to interrogate, so do not block. The
        # runtime will cope or fail loudly, and guessing here would block a
        # working setup.
        return out
    free = int(info.get("free_mb", 0))
    available = free + max(0, int(reclaimable_mb))
    out.update({"fits": want <= available, "free_mb": free,
                "reclaimable_mb": max(0, int(reclaimable_mb)),
                "available_mb": available,
                "total_mb": info.get("total_mb", 0), "name": info.get("name", "")})
    if not out["fits"]:
        out["holders"] = [{"pid": h.pid, "name": h.name, "used_mb": h.used_mb}
                          for h in holders()]
    return out


def explain_shortfall(check: dict[str, Any], model: str) -> str:
    """One line a user can act on."""
    free = check.get("free_mb", 0)
    want = check.get("needed_mb", 0)
    msg = (f"{model} needs about {want / 1024:.1f} GB of VRAM and only "
           f"{free / 1024:.1f} GB is free")
    top = check.get("holders") or []
    if top:
        who = ", ".join(f"{h['name']} {h['used_mb'] / 1024:.1f} GB" for h in top)
        msg += f". Currently held by: {who}"
    return msg
