"""Structured, transactional editing for My dictionary."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import sys
import tomllib
from datetime import UTC, datetime

from .. import config, ipc, modes, paths
from .. import vocabulary as vocab


def etag(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def snapshot(cfg: dict) -> dict:
    data, issues = vocab.migration(cfg)
    candidate = {**cfg, "dictionary": data}
    selected = modes.resolve(candidate)
    index = vocab.Index(candidate, selected.name, selected.rules) if not issues else None
    picked, dropped = index.names.seed_split() if index else ([], [])
    return {"ok": True, "schema_version": 2, "revision": data["revision"], "etag": etag(cfg),
            "entries": sorted(data["entries"], key=lambda e: e.get("updated_at", ""), reverse=True),
            "migration_issues": issues, "migration_pending": not vocab.modern(cfg),
            "modes": list(cfg.get("modes", {})), "active_mode": selected.name,
            "seeded": picked, "dropped": dropped}


def _replace(data: dict, raw: dict, mode_ids: set[str]) -> dict:
    if not isinstance(raw, dict):
        raise vocab.InvalidWord("invalid_entry")
    original = next((e for e in data["entries"] if e["id"] == raw.get("id")), None)
    if raw.get("id") and original is None:
        raise vocab.InvalidWord("not_found", raw["id"])
    # Internal migration metadata is retained from the stored entry only.
    allowed = {"text", "aliases", "enabled", "recognition_hint", "normalize_case", "phonetic", "modes"}
    e = copy.deepcopy(original) if original else vocab.word("")
    e.update({k: v for k, v in raw.items() if k in allowed})
    e["updated_at"] = datetime.now(UTC).isoformat()
    data["entries"] = [v for v in data["entries"] if v["id"] != e["id"]] + [e]
    vocab.validate(data["entries"], mode_ids)
    return e


def apply_edit(cfg: dict, action: str, body: dict) -> dict:
    data, issues = vocab.migration(cfg)
    if issues:
        raise vocab.InvalidWord("migration_required", "\n".join(issues))
    cfg["dictionary"] = data
    if action == "save":
        _replace(data, body.get("entry"), set(cfg["modes"]))
    elif action == "remove":
        previous = next((e for e in data["entries"] if e["id"] == body.get("id")), None)
        if previous is None:
            raise vocab.InvalidWord("not_found")
        data["entries"].remove(previous)
        return {"removed": previous}
    elif action == "restore":
        previous = body.get("entry")
        if not isinstance(previous, dict):
            raise vocab.InvalidWord("invalid_entry")
        data["entries"].append(previous)
        vocab.validate(data["entries"], set(cfg["modes"]))
    elif action == "mode":
        mode = body.get("mode")
        if mode not in cfg["modes"] or not isinstance(body.get("enabled"), bool):
            raise vocab.InvalidWord("invalid_mode")
        cfg["modes"][mode].setdefault("rules", {})["vocabulary"] = body["enabled"]
    else:
        raise vocab.InvalidWord("invalid_action")
    return {}


def mutate(action: str, body: dict) -> dict:
    path = paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".vocabulary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg = config.load()
        if body.get("etag") != etag(cfg):
            raise vocab.InvalidWord("changed_elsewhere")
        running = ipc.ping(timeout=0.5)
        if running and running.get("vocabulary_schema") != 2:
            raise vocab.InvalidWord("restart_required", "Restart omavoid to load the updated speech service.")
        before = path.read_bytes() if path.exists() else None
        was_legacy = not vocab.modern(cfg)
        result = apply_edit(cfg, action, body)
        cfg["dictionary"]["revision"] += 1
        # Catch changes by older config writers that do not take this lock.
        if (path.read_bytes() if path.exists() else None) != before:
            raise vocab.InvalidWord("changed_elsewhere")
        if was_legacy and before is not None:
            backup = path.with_name(path.name + ".before-vocabulary-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))
            with paths.open_private(backup, "w") as stream:
                stream.write(before.decode())
        try:
            config.write(cfg)
        except ValueError as exc:
            raise vocab.InvalidWord("changed_elsewhere") from exc
        result.update(snapshot(cfg))
    try:
        response = ipc.request({"cmd": "reload"}, timeout=3)
        result["activation"] = "active" if response.get("ok") and response.get("vocabulary_schema") == 2 else "pending"
    except (OSError, ValueError):
        result["activation"] = "pending"
    return result


def preview(cfg: dict, body: dict) -> dict:
    from .. import post

    data, issues = vocab.migration(cfg)
    if issues:
        raise vocab.InvalidWord("migration_required", "\n".join(issues))
    cfg = copy.deepcopy(cfg)
    cfg["dictionary"] = data
    draft = body.get("entry")
    if draft:
        _replace(data, draft, set(cfg["modes"]))
    mode_id = body.get("mode") or cfg.get("switching", {}).get("mode", "default")
    if mode_id not in cfg["modes"]:
        raise vocab.InvalidWord("invalid_mode", str(mode_id))
    mode = modes.resolve(cfg, forced=mode_id)
    text = body.get("text", "")
    if not isinstance(text, str) or len(text) > 20000:
        raise vocab.InvalidWord("invalid_text")
    # Same text stage as Pipeline; no model request and no history writes.
    result = post.run(text, cfg, post.Context("", "", mode.rules, mode.name))
    return {"ok": True, "before": text, "after": result.text, "changes": result.word_changes}


def rollback(body: dict) -> dict:
    path = paths.config_file()
    backup = path.parent / str(body.get("backup", ""))
    if backup.parent != path.parent or not backup.name.startswith(path.name + ".before-vocabulary-"):
        raise vocab.InvalidWord("invalid_backup")
    with path.with_suffix(".vocabulary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg = config.load()
        if body.get("etag") != etag(cfg):
            raise vocab.InvalidWord("changed_elsewhere")
        with path.with_suffix(".lock").open("a") as config_lock:
            fcntl.flock(config_lock, fcntl.LOCK_EX)
            if cfg.source_bytes != path.read_bytes():
                raise vocab.InvalidWord("changed_elsewhere")
            old = backup.read_bytes()
            tomllib.loads(old.decode())
            current_backup = path.with_name(
                path.name + ".before-rollback-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))
            with paths.open_private(current_backup, "w") as stream:
                stream.write(path.read_text())
            temp = path.with_suffix(f".rollback.{os.getpid()}.tmp")
            try:
                with paths.open_private(temp, "w") as stream:
                    stream.write(old.decode())
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
    try:
        response = ipc.request({"cmd": "reload"}, timeout=3)
        active = response.get("ok", False)
    except (OSError, ValueError):
        active = False
    return {"ok": True, "activation": "active" if active else "pending", "backup": str(current_backup)}


def cmd_vocabulary(args) -> int:
    try:
        body = json.load(sys.stdin) if args.json_input else {}
        if not isinstance(body, dict):
            raise vocab.InvalidWord("invalid_entry")
        if args.action == "list":
            result = snapshot(config.load())
        elif args.action == "preview":
            result = preview(config.load(), body)
        elif args.action == "rollback":
            result = rollback(body)
        elif args.action == "reload":
            response = ipc.request({"cmd": "reload"}, timeout=3)
            result = {"ok": bool(response.get("ok")),
                      "activation": "active" if response.get("vocabulary_schema") == 2 else "pending"}
        else:
            result = mutate(args.action, body)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"ok": False, "error": getattr(exc, "code", "failed"),
                          "detail": str(exc)}, ensure_ascii=False))
        return 1


def legacy_command(args, kind: str, cfg: dict) -> int:
    """Keep old mutation commands useful without maintaining two stores."""
    data = copy.deepcopy(cfg["dictionary"])
    entries = data["entries"]
    if kind == "dict":
        source = args.heard
        if args.action == "add":
            if not source or not args.meant:
                raise vocab.InvalidWord("invalid_text")
            for e in entries:
                e["aliases"] = [s for s in e["aliases"] if s != source]
                e.get("legacy", {}).get("rule_keys", [])[:] = [
                    s for s in e.get("legacy", {}).get("rule_keys", []) if s != source]
            e = next((e for e in entries if e["text"] == args.meant), None)
            if e is None:
                e = vocab.word(args.meant, recognition_hint=False, normalize_case=False)
                entries.append(e)
            if source != e["text"]:
                e["aliases"].append(source)
            else:
                e["normalize_case"] = True
            e.setdefault("legacy", {}).setdefault("rule_keys", []).append(source)
        elif args.action == "rm":
            found = False
            for e in entries:
                if source in e["aliases"] or source in e.get("legacy", {}).get("rule_keys", []):
                    found = True
                    e["aliases"] = [s for s in e["aliases"] if s != source]
                    if vocab.folded(source) == vocab.folded(e["text"]):
                        e["normalize_case"] = False
                    if "legacy" in e:
                        e["legacy"]["rule_keys"] = [s for s in e["legacy"].get("rule_keys", []) if s != source]
            if not found:
                raise vocab.InvalidWord("not_found", source)
    else:
        if args.action in ("add", "rm") and not args.names:
            raise vocab.InvalidWord("invalid_text")
        for name in args.names if args.action != "enable" else [e["text"] for e in entries]:
            e = next((e for e in entries if e["text"] == name), None)
            if args.action == "add":
                if e is None:
                    e = vocab.word(name, normalize_case=False)
                    entries.append(e)
                e["recognition_hint"] = True
                e.setdefault("legacy", {}).update({"name": True, "group": args.group})
            elif e is None:
                raise vocab.InvalidWord("not_found", name)
            elif args.action == "rm":
                e["recognition_hint"] = False
                e["phonetic"]["enabled"] = False
                e.setdefault("legacy", {})["name"] = False
            elif args.action == "enable":
                e["phonetic"]["enabled"] = True
    data["entries"] = [e for e in entries if e["aliases"] or e["recognition_hint"]
                       or e["normalize_case"] or e["phonetic"]["enabled"]]
    vocab.validate(data["entries"], set(cfg["modes"]))
    # Serialize legacy writes through the same lock and stale-snapshot check.
    path = paths.config_file()
    with path.with_suffix(".vocabulary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if etag(config.load()) != etag(cfg):
            raise vocab.InvalidWord("changed_elsewhere")
        cfg["dictionary"] = data
        data["revision"] += 1
        config.write(cfg)
    print("Saved. Run omavoi reload to apply.")
    return 0
